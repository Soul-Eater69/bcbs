import json
import os

import httpx
from dotenv import load_dotenv

from phoenix.otel import register
from phoenix.evals import LLM
from phoenix.evals.metrics import FaithfulnessEvaluator


# ============================================================
# 0. ENV
# ============================================================

load_dotenv()


def required_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Missing environment variable: {name}"
        )

    return value


# ============================================================
# 1. CONNECT TRACING TO PHOENIX UI
# ============================================================
#
# Evaluator executions will be visible in Phoenix.
#
# batch=False is convenient for notebooks/local testing because
# spans are sent immediately rather than waiting for shutdown.
# ============================================================

tracer_provider = register(
    project_name="jira-epic-evaluator-testing",
    batch=False,
)


# ============================================================
# 2. IDP TOKEN
# ============================================================

def get_idp_token() -> str:

    headers = {
        "Accept": "*/*",
        "ClientId": required_env("IDP_CLIENT_ID"),
        "ClientSecret": required_env("IDP_CLIENT_SECRET"),
        "scope": "profile openid roles permissions",
    }

    body = {
        "username": required_env("IDP_USER"),
        "password": required_env("IDP_PASSWORD"),
    }

    with httpx.Client(
        verify=False,       # LOCAL TEST ONLY
        timeout=30.0,
    ) as client:

        response = client.post(
            required_env("IDP_AUTH_URL"),
            headers=headers,
            json=body,
        )

        response.raise_for_status()

        payload = response.json()

    token = payload.get("jwt_token")

    if not token:
        raise RuntimeError(
            f"IDP response missing jwt_token: {payload}"
        )

    return token


# ============================================================
# 3. CUSTOM HTTP CLIENT
# ============================================================
#
# OpenAI SDK normally calls:
#
#     POST /chat/completions
#
# Your gateway expects:
#
#     POST /api/v1/chatcompletions
#
# So instead of creating another proxy/server, this custom
# client translates the OpenAI request locally.
#
# Phoenix
#    ↓
# OpenAI SDK
#    ↓
# GatewayHTTPClient
#    ↓
# corporate gateway
#
# ============================================================

class GatewayHTTPClient(httpx.Client):

    def __init__(self):

        # OpenAI requires an httpx.Client instance.
        super().__init__(
            timeout=90.0
        )

        # Actual client used to hit your gateway.
        self.gateway_client = httpx.Client(
            verify=False,       # LOCAL TEST ONLY
            timeout=90.0,
        )

        self.token = get_idp_token()

        self.gateway_url = (
            required_env("LLM_BASE_URL").rstrip("/")
            + "/api/v1/chatcompletions"
        )

    def _call_gateway(
        self,
        payload: dict,
    ) -> httpx.Response:

        headers = {
            "Authorization": (
                f"Bearer {self.token}"
            ),
            "app-id": required_env("LLM_APP_ID"),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # ----------------------------------------------------
        # Gateway-specific request fields
        # ----------------------------------------------------

        payload["model"] = required_env(
            "LLM_MODEL"
        )

        payload["api_version"] = (
            "2024-04-01-preview"
        )

        response = self.gateway_client.post(
            self.gateway_url,
            headers=headers,
            json=payload,
        )

        # Token expired/rejected:
        # refresh once and retry.
        if response.status_code == 401:

            self.token = get_idp_token()

            headers["Authorization"] = (
                f"Bearer {self.token}"
            )

            response = self.gateway_client.post(
                self.gateway_url,
                headers=headers,
                json=payload,
            )

        return response

    def send(
        self,
        request: httpx.Request,
        *args,
        **kwargs,
    ) -> httpx.Response:

        # ----------------------------------------------------
        # Request was produced by OpenAI SDK.
        #
        # Example:
        # {
        #   "model": ...,
        #   "messages": ...,
        #   "tools": ...,
        #   ...
        # }
        #
        # Preserve ALL fields Phoenix/OpenAI sends.
        # ----------------------------------------------------

        try:
            body = request.content.decode(
                "utf-8"
            )

            payload = json.loads(body)

        except Exception as exc:

            raise RuntimeError(
                "Could not parse OpenAI request body"
            ) from exc

        # ----------------------------------------------------
        # Call corporate gateway
        # ----------------------------------------------------

        response = self._call_gateway(
            payload
        )

        # Useful during initial debugging
        print(
            "\n[GATEWAY]",
            response.status_code,
            self.gateway_url,
        )

        if response.status_code >= 400:

            print(
                "[GATEWAY ERROR]",
                response.text,
            )

        # ----------------------------------------------------
        # Read gateway response
        # ----------------------------------------------------

        try:
            gateway_payload = (
                response.json()
            )

        except Exception:

            return httpx.Response(
                status_code=response.status_code,
                content=response.content,
                headers={
                    "content-type":
                        response.headers.get(
                            "content-type",
                            "text/plain",
                        )
                },
                request=request,
            )

        # ----------------------------------------------------
        # Your original code showed:
        #
        # choice = payload.get("choice")
        #          or payload["choices"][0]
        #
        # OpenAI SDK expects:
        #
        # "choices": [...]
        #
        # Normalize singular "choice" if your gateway uses it.
        # ----------------------------------------------------

        if (
            "choice" in gateway_payload
            and "choices" not in gateway_payload
        ):

            gateway_payload["choices"] = [
                gateway_payload.pop("choice")
            ]

        # ----------------------------------------------------
        # OpenAI SDK needs an OpenAI-shaped HTTP response.
        # ----------------------------------------------------

        return httpx.Response(
            status_code=response.status_code,
            json=gateway_payload,
            headers={
                "content-type":
                    "application/json"
            },
            request=request,
        )

    def close(self):

        self.gateway_client.close()

        super().close()


# ============================================================
# 4. CREATE THE CUSTOM HTTP CLIENT
# ============================================================

gateway_http_client = GatewayHTTPClient()


# ============================================================
# 5. PHOENIX JUDGE MODEL
# ============================================================
#
# IMPORTANT:
#
# client="openai"
#
# means Phoenix uses the native OpenAI Python SDK.
# No LangChain involved.
#
# base_url itself isn't important to the gateway because our
# custom httpx client intercepts the outgoing request, but it
# must still be a valid URL for OpenAI to build its request.
# ============================================================

judge_llm = LLM(
    provider="openai",
    client="openai",

    model=required_env(
        "LLM_MODEL"
    ),

    api_key="unused",

    base_url=(
        required_env(
            "LLM_BASE_URL"
        ).rstrip("/")
        + "/api/v1"
    ),

    sync_client_kwargs={
        "http_client":
            gateway_http_client
    },
)


# ============================================================
# 6. PHOENIX FAITHFULNESS / HALLUCINATION JUDGE
# ============================================================

faithfulness_evaluator = (
    FaithfulnessEvaluator(
        llm=judge_llm,

        # Keep judge deterministic during benchmarking.
        temperature=0.0,
    )
)


# ============================================================
# 7. SHOW EVALUATOR DESCRIPTION / PROMPT
# ============================================================

print(
    "\n"
    "========== EVALUATOR ==========\n"
)

print(
    faithfulness_evaluator.describe()
)


print(
    "\n"
    "========== PHOENIX PROMPT ==========\n"
)

print(
    faithfulness_evaluator.prompt_template
)


# ============================================================
# 8. TEST DATA
#
# THEME = source of truth / context
# EPIC  = output we're checking for hallucination
# ============================================================

theme = """
Theme Name:
Improve Customer Onboarding

Theme Description:
Reduce customer onboarding friction and automate
manual verification activities.

Expected Outcomes:
- Reduce onboarding time by 25%
- Reduce abandoned registrations
- Reduce manual verification effort
"""


# Deliberately hallucinated example:
#
# Theme DOES NOT mention:
# - facial recognition
# - AWS Rekognition
# - 98% accuracy
#
epic = """
Title:
AI Biometric Customer Verification

Description:
Implement facial recognition using AWS Rekognition
to automate customer identity verification and improve
the onboarding experience.

Success Criteria:
- Reduce onboarding time by 25%
- Facial recognition accuracy must exceed 98%
- AWS Rekognition integration is completed
"""


# ============================================================
# 9. RUN PHOENIX EVALUATOR
# ============================================================

evaluation_input = {

    # This represents what the generator was asked to do.
    "input": """
Generate a Jira Epic from the provided Theme.
""",

    # Authoritative source
    "context": theme,

    # Generated Jira Epic
    "output": epic,
}


scores = faithfulness_evaluator.evaluate(
    evaluation_input
)


score = scores[0]


# ============================================================
# 10. OUTPUT
# ============================================================

print(
    "\n"
    "========== RESULT ==========\n"
)

print(
    "Label       :",
    score.label,
)

print(
    "Score       :",
    score.score,
)

print(
    "Explanation :",
    score.explanation,
)

print(
    "Metadata    :",
    score.metadata,
)


# ============================================================
# 11. TRACE ID
# ============================================================
#
# Phoenix documents that when evaluator tracing is enabled,
# evaluation metadata can include the trace_id.
# ============================================================

if score.metadata:

    trace_id = score.metadata.get(
        "trace_id"
    )

    if trace_id:

        print(
            "\nPhoenix Trace ID:",
            trace_id,
        )


# ============================================================
# 12. CLEANUP
# ============================================================

gateway_http_client.close()
