import json
import os

import httpx
from dotenv import load_dotenv

from phoenix.evals import LLM
from phoenix.evals.metrics import FaithfulnessEvaluator


load_dotenv()


# ============================================================
# 1. GET YOUR CORPORATE JWT
# ============================================================

def get_idp_token() -> str:
    headers = {
        "Accept": "*/*",
        "ClientId": os.environ["IDP_CLIENT_ID"],
        "ClientSecret": os.environ["IDP_CLIENT_SECRET"],
        "scope": "profile openid roles permissions",
    }

    body = {
        "username": os.environ["IDP_USER"],
        "password": os.environ["IDP_PASSWORD"],
    }

    with httpx.Client(
        verify=False,       # local testing only
        timeout=30.0,
    ) as client:

        r = client.post(
            os.environ["IDP_AUTH_URL"],
            headers=headers,
            json=body,
        )

        r.raise_for_status()

        token = r.json().get("jwt_token")

    if not token:
        raise RuntimeError(
            "IDP response missing jwt_token"
        )

    return token


# ============================================================
# 2. VERY SMALL OPENAI -> CORPORATE GATEWAY BRIDGE
# ============================================================

class GatewayTransport(httpx.BaseTransport):
    """
    Phoenix/OpenAI thinks it is calling:

        /api/v1/chat/completions

    Your gateway actually expects:

        /api/v1/chatcompletions

    This transport fixes that locally and also injects:
      - Bearer JWT
      - app-id
      - api_version

    No extra server required.
    """

    def __init__(self):
        self.transport = httpx.HTTPTransport(
            verify=False
        )

        self.token = get_idp_token()

    def _send(
        self,
        request: httpx.Request,
    ) -> httpx.Response:

        # ------------------------------------------
        # Fix gateway path
        # ------------------------------------------

        path = request.url.path

        path = path.replace(
            "/chat/completions",
            "/chatcompletions",
        )

        url = request.url.copy_with(
            path=path
        )

        # ------------------------------------------
        # OpenAI SDK generated JSON body
        # Keep everything Phoenix sends:
        # messages, tools, tool_choice, temperature...
        # ------------------------------------------

        payload = json.loads(
            request.content.decode("utf-8")
        )

        # Corporate gateway-specific field
        payload["api_version"] = (
            "2024-04-01-preview"
        )

        # Ensure corporate deployment/model
        payload["model"] = os.environ[
            "LLM_MODEL"
        ]

        # ------------------------------------------
        # Corporate auth
        # ------------------------------------------

        headers = dict(
            request.headers
        )

        headers["Authorization"] = (
            f"Bearer {self.token}"
        )

        headers["app-id"] = os.environ[
            "LLM_APP_ID"
        ]

        headers["Content-Type"] = (
            "application/json"
        )

        gateway_request = httpx.Request(
            method=request.method,
            url=url,
            headers=headers,
            content=json.dumps(payload),
        )

        response = self.transport.handle_request(
            gateway_request
        )

        response.read()

        return response

    def handle_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:

        response = self._send(request)

        # Token expired? Refresh once.
        if response.status_code == 401:
            self.token = get_idp_token()
            response = self._send(request)

        # ------------------------------------------
        # Your gateway code shows it may return
        # "choice" instead of OpenAI's "choices".
        #
        # Normalize that for Phoenix/OpenAI.
        # ------------------------------------------

        try:
            payload = response.json()

            if (
                "choice" in payload
                and "choices" not in payload
            ):
                payload["choices"] = [
                    payload.pop("choice")
                ]

            return httpx.Response(
                status_code=response.status_code,
                headers=response.headers,
                json=payload,
                request=request,
            )

        except Exception:

            return httpx.Response(
                status_code=response.status_code,
                headers=response.headers,
                content=response.content,
                request=request,
            )

    def close(self):
        self.transport.close()


# ============================================================
# 3. GIVE THE CUSTOM HTTP CLIENT TO PHOENIX
# ============================================================

gateway_http_client = httpx.Client(
    transport=GatewayTransport(),
    timeout=90.0,
)


judge_llm = LLM(
    provider="openai",

    # Explicit: use OpenAI SDK, NOT LangChain
    client="openai",

    model=os.environ["LLM_MODEL"],

    # OpenAI SDK will append /chat/completions.
    # Our transport fixes it to /chatcompletions.
    base_url=(
        os.environ["LLM_BASE_URL"].rstrip("/")
        + "/api/v1"
    ),

    # OpenAI SDK requires an API key.
    # GatewayTransport replaces the Authorization header
    # with your actual IDP JWT.
    api_key="unused",

    sync_client_kwargs={
        "http_client": gateway_http_client
    },
)


# ============================================================
# 4. PHOENIX BUILT-IN FAITHFULNESS JUDGE
# ============================================================

faithfulness = FaithfulnessEvaluator(
    llm=judge_llm,
    temperature=0.0,
)


# ============================================================
# 5. VIEW THE ACTUAL ARIZE PROMPT
# ============================================================

print("\n========== PHOENIX PROMPT ==========\n")

print(
    faithfulness.prompt_template
)


# ============================================================
# 6. TEST YOUR THEME -> EPIC
# ============================================================

theme = """
Theme Name:
Improve Customer Onboarding

Description:
Reduce customer onboarding friction and
automate manual verification.

Success Criteria:
- Reduce onboarding time by 25%
- Reduce abandoned registrations
"""


epic = """
Title:
AI Biometric Customer Verification

Description:
Implement AWS Rekognition facial recognition
to automate identity verification.

Success Criteria:
- Reduce onboarding time by 25%
- Facial recognition accuracy must exceed 98%
"""


result = faithfulness.evaluate(
    {
        # This is basically the generation task/question
        "input": (
            "Generate a Jira Epic from the "
            "provided Theme."
        ),

        # Source of truth
        "context": theme,

        # Thing being judged
        "output": epic,
    }
)


print("\n========== RESULT ==========\n")

score = result[0]

print("Label      :", score.label)
print("Score      :", score.score)
print("Explanation:", score.explanation)


gateway_http_client.close()
