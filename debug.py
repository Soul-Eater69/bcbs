# ============================================================
# AUTO-RESUME UNTIL COMPLETE
#
# - Starts from case 36
# - Uses EXISTING framework_async + existing Excel workbook
# - Runs one case at a time
# - On Azure/PTU 429:
#       wait AT LEAST 180 seconds
#       retry same case
#       repeat until success
# - Then automatically moves to next case
#
# IMPORTANT:
# DO NOT recreate framework_async in this cell.
# It must be the existing framework instance that already
# contains the Excel writer/workbook with cases 30-35 saved.
# ============================================================

import asyncio
import time
import traceback
from datetime import datetime

from phoenix.evals.rate_limiters import (
    RateLimitError as PhoenixRateLimitError,
)


# ============================================================
# OPTIONAL OPENAI / HTTPX OPERATIONAL ERROR TYPES
# ============================================================

try:
    from openai import (
        RateLimitError as OpenAIRateLimitError,
        APITimeoutError,
        APIConnectionError,
        InternalServerError,
    )
except ImportError:
    OpenAIRateLimitError = None
    APITimeoutError = None
    APIConnectionError = None
    InternalServerError = None


try:
    import httpx
except ImportError:
    httpx = None


# ============================================================
# CONFIG
# ============================================================

START_CASE = 36
STOP_CASE = len(cases)

# Azure said retry-after: 30,
# but we deliberately wait longer for PTU recovery.
RATE_LIMIT_WAIT_SECONDS = 180       # 3 minutes

# For temporary connectivity/timeouts/5xx.
OTHER_OPERATIONAL_WAIT_SECONDS = 60

# Small buffer if Azure ever gives a retry-after > our default.
RETRY_AFTER_BUFFER_SECONDS = 5


RUN_NAME = "generated_epic_vs_theme_text"
DATASET_NAME = "epic_gen.parquet"


# ============================================================
# RATE-LIMIT INFORMATION CAPTURED FROM REAL AZURE RESPONSES
# ============================================================

rate_limit_state = {
    "retry_after_seconds": None,
    "status_code": None,
    "message": None,
    "request_id": None,
    "last_429_at": None,
}


# ============================================================
# GET EXISTING ASYNC OPENAI CLIENT
# ============================================================

llm = judge._llm

async_openai_client = llm._async_client
async_httpx_client = async_openai_client._client


# ============================================================
# HTTP HOOK
#
# Captures the ORIGINAL Azure 429 before Phoenix converts it
# into its own RateLimitError.
# ============================================================

async def capture_rate_limit_info(response):

    if response.status_code != 429:
        return

    rate_limit_state["status_code"] = 429
    rate_limit_state["last_429_at"] = time.time()

    # --------------------------------------------------------
    # retry-after
    # --------------------------------------------------------

    retry_after_raw = response.headers.get("retry-after")

    retry_after_seconds = None

    if retry_after_raw is not None:
        try:
            retry_after_seconds = float(retry_after_raw)
        except (TypeError, ValueError):
            retry_after_seconds = None

    rate_limit_state["retry_after_seconds"] = (
        retry_after_seconds
    )

    # --------------------------------------------------------
    # request ID
    # --------------------------------------------------------

    request_id = (
        response.headers.get("x-request-id")
        or response.headers.get("apim-request-id")
        or response.headers.get("request-id")
    )

    rate_limit_state["request_id"] = request_id

    # --------------------------------------------------------
    # Read Azure response body
    # --------------------------------------------------------

    try:
        await response.aread()
        body = response.text
    except Exception:
        body = ""

    rate_limit_state["message"] = body[:2000]

    # --------------------------------------------------------
    # Concise logging
    # --------------------------------------------------------

    print("\n")
    print("!" * 90)
    print("AZURE 429 THROTTLE CAPTURED")
    print("!" * 90)

    print(
        "time        :",
        datetime.now().strftime("%H:%M:%S"),
    )

    print(
        "retry-after :",
        retry_after_raw,
    )

    if request_id:
        print(
            "request-id  :",
            request_id,
        )

    if "Provisioned-Managed" in body:
        print(
            "reason      : "
            "Provisioned-Managed throughput exceeded"
        )

    elif body:
        print(
            "body        :",
            body[:800],
        )

    print("!" * 90)


# ============================================================
# REMOVE OLD COPIES OF OUR HOOK
#
# Also remove previous diagnostic hook from the earlier cell
# so we don't print every response twice.
# ============================================================

existing_hooks = (
    async_httpx_client.event_hooks.setdefault(
        "response",
        [],
    )
)

async_httpx_client.event_hooks["response"] = [
    hook
    for hook in existing_hooks
    if getattr(hook, "__name__", "")
    not in {
        "capture_rate_limit_info",
        "capture_retry_after",
        "trace_async_http_response",
    }
]

async_httpx_client.event_hooks["response"].append(
    capture_rate_limit_info
)

print(
    "Rate-limit hook installed:",
    [
        getattr(h, "__name__", str(h))
        for h
        in async_httpx_client.event_hooks["response"]
    ],
)


# ============================================================
# BUILD OPERATIONAL EXCEPTION TYPES
# ============================================================

operational_types = [
    PhoenixRateLimitError,
]

for exc_type in (
    OpenAIRateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
):
    if isinstance(exc_type, type):
        operational_types.append(exc_type)


if httpx is not None:

    operational_types.extend(
        [
            httpx.TimeoutException,
            httpx.TransportError,
        ]
    )


OPERATIONAL_TYPES = tuple(
    set(operational_types)
)


# ============================================================
# CHECK EXCEPTION CHAIN
# ============================================================

def find_operational_error(exc):

    current = exc
    seen = set()

    while (
        current is not None
        and id(current) not in seen
    ):

        seen.add(id(current))

        if isinstance(
            current,
            OPERATIONAL_TYPES,
        ):
            return current

        if current.__cause__ is not None:
            current = current.__cause__
        else:
            current = current.__context__

    return None


# ============================================================
# DETERMINE WAIT TIME
# ============================================================

def get_wait_seconds(exc):

    operational_error = find_operational_error(exc)

    # --------------------------------------------------------
    # Rate limit / 429
    # --------------------------------------------------------

    is_rate_limit = isinstance(
        operational_error,
        tuple(
            t
            for t in (
                PhoenixRateLimitError,
                OpenAIRateLimitError
                if isinstance(
                    OpenAIRateLimitError,
                    type,
                )
                else None,
            )
            if isinstance(t, type)
        ),
    )

    if (
        is_rate_limit
        or rate_limit_state["status_code"] == 429
    ):

        azure_retry_after = (
            rate_limit_state[
                "retry_after_seconds"
            ]
        )

        # Always wait AT LEAST 3 minutes.
        wait_seconds = (
            RATE_LIMIT_WAIT_SECONDS
        )

        # If Azure ever asks for MORE than 3 minutes,
        # respect the longer value.
        if azure_retry_after is not None:

            wait_seconds = max(
                wait_seconds,
                azure_retry_after
                + RETRY_AFTER_BUFFER_SECONDS,
            )

        return (
            wait_seconds,
            "rate_limit",
        )

    # --------------------------------------------------------
    # Timeout / connection / temporary 5xx
    # --------------------------------------------------------

    return (
        OTHER_OPERATIONAL_WAIT_SECONDS,
        "temporary_provider_error",
    )


# ============================================================
# MAIN AUTO-RESUME LOOP
# ============================================================

print("\n")
print("#" * 90)

print(
    f"AUTO-RESUME RUN: "
    f"CASE {START_CASE} → CASE {STOP_CASE}"
)

print(
    f"Rate-limit cooldown: "
    f"{RATE_LIMIT_WAIT_SECONDS} seconds"
)

print(
    "Runs one case at a time and retries operational "
    "failures until successful."
)

print("#" * 90)


overall_started = time.perf_counter()

completed_this_run = 0
total_retries = 0


for human_case_number in range(
    START_CASE,
    STOP_CASE + 1,
):

    case_index = human_case_number - 1
    case = cases[case_index]

    attempt = 0

    while True:

        attempt += 1

        # Reset response info for this attempt.
        rate_limit_state[
            "retry_after_seconds"
        ] = None

        rate_limit_state[
            "status_code"
        ] = None

        rate_limit_state[
            "message"
        ] = None

        rate_limit_state[
            "request_id"
        ] = None


        print("\n")
        print("=" * 90)

        print(
            f"CASE {human_case_number}"
            f"/{STOP_CASE}"
            f" | attempt {attempt}"
        )

        print(
            f"case_id: {case.case_id}"
        )

        print("=" * 90)


        started = time.perf_counter()


        try:

            # =================================================
            # ONE CASE AT A TIME
            #
            # Once BOTH metrics succeed, a_evaluate()
            # publishes this case to the existing Excel writer.
            # =================================================

            result = await framework_async.a_evaluate(
                case,
                metrics=[
                    "coverage",
                    "faithfulness",
                ],
                run_name=RUN_NAME,
                dataset_name=DATASET_NAME,
                max_concurrency=1,
            )


            elapsed = (
                time.perf_counter()
                - started
            )


            coverage = result[
                "coverage"
            ]

            faithfulness = result[
                "faithfulness"
            ]


            completed_this_run += 1


            print("\n")
            print("✓" * 45)

            print(
                f"✓ CASE {human_case_number} COMPLETE"
            )

            print(
                f"attempts       : {attempt}"
            )

            print(
                f"elapsed        : {elapsed:.1f}s"
            )

            print(
                f"coverage       : "
                f"{coverage.score} "
                f"({coverage.label})"
            )

            print(
                f"faithfulness   : "
                f"{faithfulness.score} "
                f"({faithfulness.label})"
            )

            print(
                "✓ persisted to existing Excel workbook"
            )

            print("✓" * 45)


            # Success -> move to next case.
            break


        except Exception as exc:

            elapsed = (
                time.perf_counter()
                - started
            )


            operational_error = (
                find_operational_error(exc)
            )


            # =================================================
            # REAL BUG / CONFIG / PERSISTENCE ERROR
            #
            # Never infinitely retry these.
            # =================================================

            if operational_error is None:

                print("\n")
                print("X" * 90)

                print(
                    f"NON-OPERATIONAL ERROR "
                    f"ON CASE {human_case_number}"
                )

                print(
                    f"type    : "
                    f"{type(exc).__name__}"
                )

                print(
                    f"message : {exc}"
                )

                print(
                    "This does NOT look like a temporary "
                    "provider failure."
                )

                print(
                    "Stopping instead of retrying a bug forever."
                )

                print("X" * 90)

                traceback.print_exc()

                raise


            # =================================================
            # TEMPORARY OPERATIONAL FAILURE
            # =================================================

            total_retries += 1

            (
                wait_seconds,
                failure_kind,
            ) = get_wait_seconds(exc)


            print("\n")
            print("!" * 90)

            print(
                f"⚠ CASE {human_case_number} "
                f"ATTEMPT {attempt} FAILED TEMPORARILY"
            )

            print(
                f"kind        : {failure_kind}"
            )

            print(
                f"error       : "
                f"{type(operational_error).__name__}"
            )

            print(
                f"message     : "
                f"{operational_error}"
            )

            print(
                f"elapsed     : {elapsed:.1f}s"
            )


            if (
                rate_limit_state[
                    "retry_after_seconds"
                ]
                is not None
            ):

                print(
                    "Azure retry-after: "
                    f"{rate_limit_state['retry_after_seconds']}s"
                )


            if (
                rate_limit_state[
                    "request_id"
                ]
            ):

                print(
                    "request-id  : "
                    f"{rate_limit_state['request_id']}"
                )


            print(
                f"cooldown    : "
                f"{wait_seconds:.0f}s "
                f"({wait_seconds / 60:.1f} minutes)"
            )

            print(
                "action      : "
                "retry SAME case automatically"
            )

            print("!" * 90)


            # =================================================
            # COUNTDOWN
            # =================================================

            remaining = int(
                wait_seconds
            )

            while remaining > 0:

                # Print once per minute,
                # plus final 30 seconds.
                if (
                    remaining % 60 == 0
                    or remaining <= 30
                ):

                    print(
                        f"Retrying case "
                        f"{human_case_number} "
                        f"in {remaining}s..."
                    )

                sleep_for = min(
                    30,
                    remaining,
                )

                await asyncio.sleep(
                    sleep_for
                )

                remaining -= sleep_for


            print(
                f"\nRetrying case "
                f"{human_case_number} now..."
            )


# ============================================================
# FINISHED
# ============================================================

overall_elapsed = (
    time.perf_counter()
    - overall_started
)


print("\n")
print("#" * 90)

print(
    f"✓ ALL CASES "
    f"{START_CASE}–{STOP_CASE} COMPLETE"
)

print(
    f"completed this run : "
    f"{completed_this_run}"
)

print(
    f"temporary retries  : "
    f"{total_retries}"
)

print(
    f"total elapsed      : "
    f"{overall_elapsed / 60:.1f} minutes"
)

print(
    "✓ Results persisted to the existing Excel workbook."
)

print("#" * 90)
