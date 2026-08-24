# ============================================================
# ASYNC + EXCEL PERSISTENCE + AZURE RATE-LIMIT DIAGNOSTICS
#
# - Resumes from case 28
# - Coverage + Faithfulness
# - verbose=True
# - 2 concurrent judge calls
# - Saves successful batches to Excel
# - Prints actual Azure/OpenAI 429 responses
# - DOES NOT add/change retry logic
# ============================================================

import time
import traceback
from datetime import datetime

from idp_eval import (
    EvaluationFramework,
    CoverageEvaluator,
    FaithfulnessEvaluator,
)


# ============================================================
# CONFIG
# ============================================================

START_CASE = 28
STOP_CASE = len(cases)

# Two cases are submitted concurrently.
# Because each case has Coverage + Faithfulness,
# the framework semaphore ensures at most 2 judge calls
# are in flight at once.
MAX_CONCURRENCY = 2

# Keep the persistence batch small.
# If a batch gets a 429, only this small batch needs rerunning.
BATCH_SIZE = 2

EXCEL_PATH = "theme_text_vs_generated_epic_resume_28_50.xlsx"

RUN_NAME = "generated_epic_vs_theme_text"
DATASET_NAME = "epic_gen.parquet"


print("Cases available :", len(cases))
print("Starting at     :", START_CASE)
print("Stopping at     :", STOP_CASE)
print("Concurrency     :", MAX_CONCURRENCY)
print("Batch size      :", BATCH_SIZE)
print("Excel output    :", EXCEL_PATH)


# ============================================================
# 1. GET EXISTING ASYNC OPENAI HTTP CLIENT
# ============================================================

llm = judge._llm

async_openai_client = llm._async_client
async_httpx_client = async_openai_client._client


# ============================================================
# 2. HTTP TRACE STATE
# ============================================================

trace_state = {
    "total_responses": 0,
    "successful_responses": 0,
    "error_responses": 0,
    "rate_limit_responses": 0,
    "last_error_time": None,
}


# ============================================================
# 3. ASYNC HTTP RESPONSE HOOK
#
# We keep successful 200s quiet.
# On 429 / other failures we print:
#   - timestamp
#   - HTTP status
#   - retry headers
#   - rate-limit headers
#   - request IDs
#   - Azure response body
#
# This does NOT retry anything.
# ============================================================

async def trace_async_http_response(response):

    trace_state["total_responses"] += 1

    status = response.status_code

    if status < 400:
        trace_state["successful_responses"] += 1
        return

    trace_state["error_responses"] += 1

    if status == 429:
        trace_state["rate_limit_responses"] += 1

    now = time.time()
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    previous_error = trace_state["last_error_time"]

    if previous_error is None:
        error_gap = None
    else:
        error_gap = now - previous_error

    trace_state["last_error_time"] = now

    print("\n")
    print("!" * 90)
    print("HTTP ERROR RESPONSE")
    print("!" * 90)

    print(f"time          : {timestamp}")
    print(f"status        : {status}")

    try:
        print(f"method        : {response.request.method}")
        print(f"url           : {response.request.url}")
    except Exception:
        pass

    if error_gap is not None:
        print(
            f"since previous error response: "
            f"{error_gap:.2f}s"
        )

    # --------------------------------------------------------
    # Interesting Azure / OpenAI headers
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Read failure body
    # --------------------------------------------------------

    try:

        await response.aread()

        body = response.text

    except Exception as body_exc:

        body = (
            "<unable to read response body: "
            f"{type(body_exc).__name__}: {body_exc}>"
        )

    print("\nResponse body:")
    print(body[:5000])

    print("!" * 90)
    print()


# ============================================================
# 4. REMOVE OLD ASYNC TRACE HOOK COPIES
#
# Important in Jupyter because rerunning the cell creates
# a new function object.
# ============================================================

existing_hooks = async_httpx_client.event_hooks.setdefault(
    "response",
    [],
)

async_httpx_client.event_hooks["response"] = [
    hook
    for hook in existing_hooks
    if getattr(
        hook,
        "__name__",
        "",
    ) != "trace_async_http_response"
]

async_httpx_client.event_hooks["response"].append(
    trace_async_http_response
)

matching_hooks = [
    getattr(hook, "__name__", str(hook))
    for hook in async_httpx_client.event_hooks["response"]
    if getattr(
        hook,
        "__name__",
        "",
    ) == "trace_async_http_response"
]

print("\nDiagnostic async hooks:", matching_hooks)


# ============================================================
# 5. CREATE PERSISTED FRAMEWORK
#
# IMPORTANT:
# This DOES write successful evaluations to Excel.
# ============================================================

framework_async = EvaluationFramework(
    judge=judge,
    evaluators=[
        CoverageEvaluator(verbose=True),
        FaithfulnessEvaluator(verbose=True),
    ],
    output="excel",
    excel_path=EXCEL_PATH,
)


# ============================================================
# 6. EXCEPTION CHAIN PRINTER
# ============================================================

def print_exception_chain(exc):

    print("\nEXCEPTION CHAIN:")

    current = exc
    level = 0
    seen = set()

    while (
        current is not None
        and id(current) not in seen
    ):

        seen.add(id(current))

        print("\n" + "-" * 90)

        print(
            f"[{level}] "
            f"{type(current).__module__}."
            f"{type(current).__name__}"
        )

        print(f"message: {current}")

        attrs = (
            "status_code",
            "request_id",
            "code",
            "type",
            "param",
            "current_rate_tokens_per_sec",
            "initial_rate_tokens_per_sec",
            "enforcement_window_seconds",
        )

        for attr in attrs:

            value = getattr(
                current,
                attr,
                None,
            )

            if value is not None:
                print(f"{attr}: {value}")

        next_exc = current.__cause__

        if next_exc is None:
            next_exc = current.__context__

        current = next_exc
        level += 1


# ============================================================
# 7. CASES TO RUN
#
# case 28 => Python index 27
# ============================================================

resume_cases = cases[
    START_CASE - 1 : STOP_CASE
]

print("\n")
print("#" * 90)
print(
    f"ASYNC PERSISTED RUN: "
    f"CASE {START_CASE} → CASE {STOP_CASE}"
)
print("#" * 90)

print(
    f"Remaining cases: {len(resume_cases)}"
)

print(
    f"Logical judge calls expected: "
    f"{len(resume_cases) * 2}"
)

print(
    f"Maximum simultaneous judge calls: "
    f"{MAX_CONCURRENCY}"
)

print("#" * 90)


# ============================================================
# 8. RUN SMALL ASYNC BATCHES
#
# Each batch runs asynchronously.
#
# Successful batch:
#     results persisted to Excel
#
# Failed batch:
#     stop immediately
#     inspect Azure 429 above
#
# NO custom retry is added here.
# ============================================================

all_results = []

completed_cases = 0

overall_started = time.perf_counter()


for batch_offset in range(
    0,
    len(resume_cases),
    BATCH_SIZE,
):

    batch = resume_cases[
        batch_offset :
        batch_offset + BATCH_SIZE
    ]

    first_case_number = (
        START_CASE + batch_offset
    )

    last_case_number = (
        first_case_number
        + len(batch)
        - 1
    )

    print("\n")
    print("=" * 90)

    print(
        f"ASYNC BATCH: "
        f"CASE {first_case_number}"
        f" → CASE {last_case_number}"
    )

    print("=" * 90)

    batch_started = time.perf_counter()

    try:

        batch_results = await framework_async.a_evaluate_many(
            batch,
            metrics=[
                "coverage",
                "faithfulness",
            ],
            run_name=RUN_NAME,
            dataset_name=DATASET_NAME,
            max_concurrency=MAX_CONCURRENCY,
            show_progress=True,
        )

        all_results.extend(
            batch_results
        )

        completed_cases += len(batch)

        batch_elapsed = (
            time.perf_counter()
            - batch_started
        )

        print("\n")
        print("✓" * 45)

        print(
            f"✓ BATCH COMPLETED "
            f"AND SAVED TO EXCEL"
        )

        print(
            f"cases          : "
            f"{first_case_number}"
            f"–{last_case_number}"
        )

        print(
            f"batch time     : "
            f"{batch_elapsed:.1f}s"
        )

        print(
            f"total persisted: "
            f"{completed_cases}"
            f"/{len(resume_cases)}"
        )

        print(
            f"Excel          : "
            f"{EXCEL_PATH}"
        )

        print("✓" * 45)


    except Exception as exc:

        batch_elapsed = (
            time.perf_counter()
            - batch_started
        )

        print("\n")
        print("!" * 90)

        print(
            f"✗ BATCH FAILED"
        )

        print(
            f"batch cases    : "
            f"{first_case_number}"
            f"–{last_case_number}"
        )

        print(
            f"batch elapsed  : "
            f"{batch_elapsed:.1f}s"
        )

        print(
            f"previously persisted cases: "
            f"{completed_cases}"
        )

        print(
            f"final error    : "
            f"{type(exc).__name__}"
        )

        print(
            f"message        : {exc}"
        )

        print("!" * 90)

        print_exception_chain(exc)

        print("\nFULL TRACEBACK:")
        traceback.print_exc()

        print("\n")
        print("#" * 90)

        print(
            "STOPPING AT FIRST FAILED ASYNC BATCH."
        )

        print(
            "Do NOT immediately rerun."
        )

        print(
            "Inspect the HTTP 429 response above, "
            "especially retry-after / rate-limit headers "
            "and response body."
        )

        print(
            f"If we resume later, rerun from "
            f"case {first_case_number}."
        )

        print("#" * 90)

        break


else:

    overall_elapsed = (
        time.perf_counter()
        - overall_started
    )

    print("\n")
    print("#" * 90)

    print(
        f"✓ ALL CASES "
        f"{START_CASE}–{STOP_CASE} "
        f"COMPLETED"
    )

    print(
        f"Total cases    : "
        f"{len(resume_cases)}"
    )

    print(
        f"Total time     : "
        f"{overall_elapsed:.1f}s"
    )

    print(
        f"HTTP responses : "
        f"{trace_state['total_responses']}"
    )

    print(
        f"HTTP errors    : "
        f"{trace_state['error_responses']}"
    )

    print(
        f"429 responses  : "
        f"{trace_state['rate_limit_responses']}"
    )

    print(
        f"Excel saved to : "
        f"{EXCEL_PATH}"
    )

    print("#" * 90)
