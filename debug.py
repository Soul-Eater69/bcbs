import time
from datetime import datetime
import traceback

from idp_eval import (
    EvaluationFramework,
    CoverageEvaluator,
    FaithfulnessEvaluator,
)


# ------------------------------------------------------------
# 1. Attach diagnostic hook to EXISTING Azure/OpenAI client
# ------------------------------------------------------------

llm = judge._llm
openai_client = llm._sync_client
httpx_client = openai_client._client

state = {
    "http_attempt": 0,
    "last_response_time": None,
}


def trace_http_response(response):
    state["http_attempt"] += 1
    attempt = state["http_attempt"]

    now = time.time()
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    if state["last_response_time"] is None:
        gap = None
    else:
        gap = now - state["last_response_time"]

    state["last_response_time"] = now

    print("\n" + "=" * 80)
    print(f"HTTP RESPONSE #{attempt}")
    print(f"time       : {timestamp}")
    print(f"status     : {response.status_code}")

    if gap is not None:
        print(f"since prior response: {gap:.2f}s")

    # Only print rate-limit / request-id diagnostics.
    interesting_headers = {}

    for key, value in response.headers.items():
        k = key.lower()

        if (
            "retry" in k
            or "ratelimit" in k
            or "rate-limit" in k
            or "request-id" in k
            or "request_id" in k
            or k.startswith("x-ms-")
        ):
            interesting_headers[key] = value

    print("\nRelevant headers:")

    if interesting_headers:
        for key, value in interesting_headers.items():
            print(f"  {key}: {value}")
    else:
        print("  <none>")

    # Body is useful only on failures.
    if response.status_code >= 400:
        try:
            response.read()
            body = response.text
        except Exception as body_exc:
            body = f"<unable to read body: {body_exc}>"

        print("\nResponse body:")
        print(body[:5000])

    print("=" * 80)


# Avoid adding the same hook twice if this cell gets rerun.
hooks = httpx_client.event_hooks.setdefault("response", [])

if trace_http_response not in hooks:
    hooks.append(trace_http_response)


# ------------------------------------------------------------
# 2. Diagnostic framework — NO Excel, NO persistence
# ------------------------------------------------------------

debug_framework = EvaluationFramework(
    judge=judge,
    evaluators=[
        CoverageEvaluator(verbose=True),
        FaithfulnessEvaluator(verbose=True),
    ],
)


# ------------------------------------------------------------
# 3. Run CASE 28 only
# Python index 27 = human case 28
# ------------------------------------------------------------

case_number = 28
case = cases[case_number - 1]

print("\n" + "#" * 80)
print(f"STARTING CASE {case_number}")
print(f"case_id: {case.case_id}")
print("#" * 80)

started = time.perf_counter()

try:
    result = debug_framework.evaluate(
        case,
        metrics=["coverage", "faithfulness"],
        run_name="rate-limit-debug",
        dataset_name="epic_gen_case_28_debug",
    )

    elapsed = time.perf_counter() - started

    print("\n✓ CASE COMPLETED")
    print(f"elapsed: {elapsed:.2f}s")

    for metric, metric_result in result.items():
        print(
            f"{metric}: "
            f"score={metric_result.score} "
            f"label={metric_result.label}"
        )

except Exception as exc:
    elapsed = time.perf_counter() - started

    print("\n" + "!" * 80)
    print("CASE FAILED")
    print(f"elapsed: {elapsed:.2f}s")
    print(f"final exception: {type(exc).__name__}")
    print(f"message: {exc}")
    print("!" * 80)

    print("\nEXCEPTION CHAIN:")

    current = exc
    level = 0
    seen = set()

    while current is not None and id(current) not in seen:
        seen.add(id(current))

        print(
            f"\n[{level}] "
            f"{type(current).__module__}."
            f"{type(current).__name__}"
        )
        print(str(current))

        for attr in (
            "status_code",
            "request_id",
            "code",
            "type",
            "param",
            "current_rate_tokens_per_sec",
            "initial_rate_tokens_per_sec",
            "enforcement_window_seconds",
        ):
            value = getattr(current, attr, None)

            if value is not None:
                print(f"  {attr}: {value}")

        next_exc = current.__cause__

        if next_exc is None:
            next_exc = current.__context__

        current = next_exc
        level += 1

    print("\nTRACEBACK:")
    traceback.print_exc()
