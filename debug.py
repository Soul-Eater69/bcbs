# ============================================================
# RUN 2
# GENERATED EPIC vs context@5
#
# - Uses EXISTING framework_async
# - Therefore appends to the SAME Excel workbook as Run 1
# - Evaluates Coverage + Faithfulness
# - Runs cases 1 -> 50 one at a time
# - Azure/PTU 429 => wait 3 minutes => retry SAME case
# - Keeps retrying operational errors until success
# - Real code/config errors still stop immediately
# ============================================================

import asyncio
import time
import traceback
from datetime import datetime

import numpy as np

from idp_eval import EvaluationCase

from phoenix.evals.rate_limiters import (
    RateLimitError as PhoenixRateLimitError,
)


# ============================================================
# OPTIONAL OPENAI ERROR TYPES
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

START_CASE = 1

RUN_NAME = "generated_epic_vs_context_at_5"
DATASET_NAME = "epic_gen.parquet"

RATE_LIMIT_WAIT_SECONDS = 180        # 3 minutes
OTHER_OPERATIONAL_WAIT_SECONDS = 60  # timeout / connection / temporary 5xx
RETRY_AFTER_BUFFER_SECONDS = 5


# ============================================================
# CONVERT NUMPY / PARQUET VALUES TO NORMAL PYTHON VALUES
# ============================================================

def to_python(value):

    if isinstance(value, np.ndarray):
        return [
            to_python(v)
            for v in value.tolist()
        ]

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(k): to_python(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_python(v)
            for v in value
        ]

    return value


# ============================================================
# VALIDATE DATA
# ============================================================

if "context@5" not in df.columns:
    raise KeyError(
        "Required column 'context@5' was not found in df."
    )


if len(df) != len(cases):
    raise ValueError(
        f"df has {len(df)} rows but cases has {len(cases)} cases. "
        "They must align row-for-row before Run 2."
    )


# ============================================================
# BUILD RUN 2 CASES
#
# IMPORTANT:
# Output stays EXACTLY the same as Run 1.
# Only authoritative context changes:
#
# Run 1:
#     context = theme_text
#
# Run 2:
#     context = context@5
# ============================================================

context5_cases = []


for position, (_, row) in enumerate(df.iterrows()):

    original_case = cases[position]

    context_value = to_python(
        row["context@5"]
    )

    context5_case = EvaluationCase(

        # Keep same original task/input
        input=original_case.input,

        # ====================================================
        # RUN 2 AUTHORITATIVE CONTEXT
        # ====================================================

        context={
            "context_at_5": context_value
        },

        # ====================================================
        # SAME GENERATED EPIC AS RUN 1
        # ====================================================

        output=original_case.output,

        instructions=original_case.instructions,

        # Keep same case ID so Run 1 / Run 2 can be compared
        case_id=original_case.case_id,

        metadata=(
            dict(original_case.metadata)
            if original_case.metadata is not None
            else None
        ),

        retrieved_documents=(
            original_case.retrieved_documents
        ),

        evaluation_scope=(
            original_case.evaluation_scope
        ),
    )

    context5_cases.append(
        context5_case
    )


STOP_CASE = len(context5_cases)


# ============================================================
# SANITY CHECK
# ============================================================

print("=" * 90)
print("RUN 2 SETUP")
print("=" * 90)

print(
    "Run name       :",
    RUN_NAME,
)

print(
    "Dataset        :",
    DATASET_NAME,
)

print(
    "Total cases    :",
    len(context5_cases),
)

print(
    "First case ID  :",
    context5_cases[0].case_id,
)

print(
    "Last case ID   :",
    context5_cases[-1].case_id,
)


# ============================================================
# VERIFY WHICH EXCEL FILE framework_async IS USING
# ============================================================

print("\nExisting framework writers:")

for writer in framework_async._writers:

    print(
        " -",
        type(writer).__name__,
        getattr(
            writer,
            "_path",
            None,
        ),
    )


print(
    "\nIMPORTANT: this run uses the EXISTING framework_async "
    "and therefore the same Excel workbook."
)


# ============================================================
# RATE-LIMIT STATE
# ============================================================

rate_limit_state = {
    "retry_after_seconds": None,
    "status_code": None,
    "message": None,
    "request_id": None,
    "last_429_at": None,
}


# ============================================================
# GET EXISTING ASYNC AZURE/OPENAI HTTP CLIENT
# ============================================================

llm = judge._llm

async_openai_client = (
    llm._async_client
)

async_httpx_client = (
    async_openai_client._client
)


# ============================================================
# HTTP RESPONSE HOOK
#
# Captures original Azure 429 BEFORE Phoenix wraps it.
# ============================================================

async def capture_rate_limit_info(response):

    if response.status_code != 429:
        return

    rate_limit_state[
        "status_code"
    ] = 429

    rate_limit_state[
        "last_429_at"
    ] = time.time()


    # --------------------------------------------------------
    # RETRY-AFTER
    # --------------------------------------------------------

    retry_after_raw = (
        response.headers.get(
            "retry-after"
        )
    )

    retry_after_seconds = None

    if retry_after_raw is not None:

        try:
            retry_after_seconds = float(
                retry_after_raw
            )

        except (
            TypeError,
            ValueError,
        ):
            retry_after_seconds = None


    rate_limit_state[
        "retry_after_seconds"
    ] = retry_after_seconds


    # --------------------------------------------------------
    # REQUEST ID
    # --------------------------------------------------------

    request_id = (
        response.headers.get(
            "x-request-id"
        )
        or response.headers.get(
            "apim-request-id"
        )
        or response.headers.get(
            "request-id"
        )
    )

    rate_limit_state[
        "request_id"
    ] = request_id


    # --------------------------------------------------------
    # RESPONSE BODY
    # --------------------------------------------------------

    try:

        await response.aread()

        body = response.text

    except Exception:

        body = ""


    rate_limit_state[
        "message"
    ] = body[:2000]


    # --------------------------------------------------------
    # LOG
    # --------------------------------------------------------

    print("\n")
    print("!" * 90)

    print(
        "AZURE 429 THROTTLE CAPTURED"
    )

    print("!" * 90)

    print(
        "time        :",
        datetime.now().strftime(
            "%H:%M:%S"
        ),
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


    if (
        "Provisioned-Managed"
        in body
    ):

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
# REMOVE OLD DEBUG HOOKS
#
# Prevent duplicate output if previous Run 1 cell installed
# earlier versions of these hooks.
# ============================================================

existing_hooks = (
    async_httpx_client
    .event_hooks
    .setdefault(
        "response",
        [],
    )
)


async_httpx_client.event_hooks[
    "response"
] = [

    hook

    for hook in existing_hooks

    if getattr(
        hook,
        "__name__",
        "",
    )
    not in {
        "capture_rate_limit_info",
        "capture_retry_after",
        "trace_async_http_response",
    }
]


async_httpx_client.event_hooks[
    "response"
].append(
    capture_rate_limit_info
)


print(
    "\nHTTP response hooks:",
    [
        getattr(
            hook,
            "__name__",
            str(hook),
        )
        for hook
        in async_httpx_client
        .event_hooks["response"]
    ],
)


# ============================================================
# BUILD OPERATIONAL ERROR TYPES
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

    if isinstance(
        exc_type,
        type,
    ):

        operational_types.append(
            exc_type
        )


if httpx is not None:

    operational_types.extend(
        [
            httpx.TimeoutException,
            httpx.TransportError,
        ]
    )


OPERATIONAL_TYPES = tuple(
    set(
        operational_types
    )
)


# ============================================================
# FIND OPERATIONAL ERROR IN EXCEPTION CHAIN
# ============================================================

def find_operational_error(exc):

    current = exc

    seen = set()


    while (
        current is not None
        and id(current) not in seen
    ):

        seen.add(
            id(current)
        )


        if isinstance(
            current,
            OPERATIONAL_TYPES,
        ):

            return current


        if (
            current.__cause__
            is not None
        ):

            current = (
                current.__cause__
            )

        else:

            current = (
                current.__context__
            )


    return None


# ============================================================
# RATE-LIMIT TYPE HELPER
# ============================================================

rate_limit_types = [
    PhoenixRateLimitError,
]


if isinstance(
    OpenAIRateLimitError,
    type,
):

    rate_limit_types.append(
        OpenAIRateLimitError
    )


RATE_LIMIT_TYPES = tuple(
    rate_limit_types
)


# ============================================================
# DETERMINE COOLDOWN
# ============================================================

def get_wait_seconds(exc):

    operational_error = (
        find_operational_error(
            exc
        )
    )


    is_rate_limit = (

        operational_error
        is not None

        and isinstance(
            operational_error,
            RATE_LIMIT_TYPES,
        )

    )


    if (
        is_rate_limit
        or rate_limit_state[
            "status_code"
        ] == 429
    ):

        azure_retry_after = (
            rate_limit_state[
                "retry_after_seconds"
            ]
        )


        # Always wait at least 3 minutes
        wait_seconds = (
            RATE_LIMIT_WAIT_SECONDS
        )


        # If Azure requests MORE than 3 minutes,
        # respect Azure's longer value.
        if (
            azure_retry_after
            is not None
        ):

            wait_seconds = max(
                wait_seconds,
                azure_retry_after
                + RETRY_AFTER_BUFFER_SECONDS,
            )


        return (
            wait_seconds,
            "rate_limit",
        )


    # Timeout / connection / temporary provider failure
    return (
        OTHER_OPERATIONAL_WAIT_SECONDS,
        "temporary_provider_error",
    )


# ============================================================
# START RUN 2
# ============================================================

print("\n")
print("#" * 90)

print(
    "RUN 2: GENERATED EPIC vs context@5"
)

print(
    f"CASES {START_CASE} → {STOP_CASE}"
)

print(
    "Metrics: coverage + faithfulness"
)

print(
    f"PTU throttle cooldown: "
    f"{RATE_LIMIT_WAIT_SECONDS}s "
    f"({RATE_LIMIT_WAIT_SECONDS / 60:.1f} minutes)"
)

print(
    "Operational failures will retry automatically "
    "until the case succeeds."
)

print("#" * 90)


overall_started = (
    time.perf_counter()
)

completed_this_run = 0

total_retries = 0


# ============================================================
# CASE LOOP
# ============================================================

for human_case_number in range(
    START_CASE,
    STOP_CASE + 1,
):

    case_index = (
        human_case_number - 1
    )

    case = (
        context5_cases[
            case_index
        ]
    )


    attempt = 0


    while True:

        attempt += 1


        # ----------------------------------------------------
        # RESET HTTP STATE FOR THIS ATTEMPT
        # ----------------------------------------------------

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
            f"RUN 2 | "
            f"CASE {human_case_number}"
            f"/{STOP_CASE}"
            f" | attempt {attempt}"
        )

        print(
            f"case_id: "
            f"{case.case_id}"
        )

        print("=" * 90)


        started = (
            time.perf_counter()
        )


        try:

            # =================================================
            # RUN ONE CASE
            #
            # max_concurrency=1:
            # only one judge request active at a time.
            #
            # Coverage then Faithfulness.
            # =================================================

            result = (
                await framework_async.a_evaluate(
                    case,
                    metrics=[
                        "coverage",
                        "faithfulness",
                    ],
                    run_name=RUN_NAME,
                    dataset_name=DATASET_NAME,
                    max_concurrency=1,
                )
            )


            elapsed = (
                time.perf_counter()
                - started
            )


            coverage = (
                result["coverage"]
            )

            faithfulness = (
                result[
                    "faithfulness"
                ]
            )


            completed_this_run += 1


            # =================================================
            # SUCCESS
            # =================================================

            print("\n")
            print("✓" * 45)

            print(
                f"✓ RUN 2 CASE "
                f"{human_case_number} COMPLETE"
            )

            print(
                f"attempts       : "
                f"{attempt}"
            )

            print(
                f"elapsed        : "
                f"{elapsed:.1f}s"
            )

            print(
                "coverage       : "
                f"{coverage.score} "
                f"({coverage.label})"
            )

            print(
                "faithfulness   : "
                f"{faithfulness.score} "
                f"({faithfulness.label})"
            )

            print(
                "✓ persisted to existing Excel workbook"
            )

            print(
                f"✓ run_name = {RUN_NAME}"
            )

            print("✓" * 45)


            # Success.
            # Move automatically to next case.
            break


        except Exception as exc:

            elapsed = (
                time.perf_counter()
                - started
            )


            operational_error = (
                find_operational_error(
                    exc
                )
            )


            # =================================================
            # NON-OPERATIONAL ERROR
            #
            # DO NOT RETRY PROGRAMMING / CONFIG / DATA BUGS.
            # =================================================

            if (
                operational_error
                is None
            ):

                print("\n")
                print("X" * 90)

                print(
                    "NON-OPERATIONAL ERROR"
                )

                print(
                    f"Run 2 case : "
                    f"{human_case_number}"
                )

                print(
                    f"case_id    : "
                    f"{case.case_id}"
                )

                print(
                    f"type        : "
                    f"{type(exc).__name__}"
                )

                print(
                    f"message     : "
                    f"{exc}"
                )

                print(
                    "Stopping instead of "
                    "retrying a real bug forever."
                )

                print("X" * 90)


                traceback.print_exc()

                raise


            # =================================================
            # TEMPORARY PROVIDER FAILURE
            # =================================================

            total_retries += 1


            (
                wait_seconds,
                failure_kind,
            ) = get_wait_seconds(
                exc
            )


            print("\n")
            print("!" * 90)

            print(
                f"⚠ RUN 2 CASE "
                f"{human_case_number} "
                f"ATTEMPT {attempt} "
                f"FAILED TEMPORARILY"
            )

            print(
                f"kind        : "
                f"{failure_kind}"
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
                f"elapsed     : "
                f"{elapsed:.1f}s"
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
            # COOLDOWN COUNTDOWN
            # =================================================

            remaining = int(
                wait_seconds
            )


            while remaining > 0:

                if (
                    remaining % 60 == 0
                    or remaining <= 30
                ):

                    print(
                        f"Retrying Run 2 case "
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


                remaining -= (
                    sleep_for
                )


            print(
                f"\nRetrying Run 2 case "
                f"{human_case_number} now..."
            )


# ============================================================
# RUN 2 FINISHED
# ============================================================

overall_elapsed = (
    time.perf_counter()
    - overall_started
)


print("\n")
print("#" * 90)

print(
    "✓ RUN 2 COMPLETE"
)

print(
    f"✓ Cases "
    f"{START_CASE}–{STOP_CASE} complete"
)

print(
    f"completed cases   : "
    f"{completed_this_run}"
)

print(
    f"temporary retries : "
    f"{total_retries}"
)

print(
    f"total elapsed     : "
    f"{overall_elapsed / 60:.1f} minutes"
)

print(
    f"run_name          : "
    f"{RUN_NAME}"
)

print(
    "✓ Results persisted to the SAME existing Excel workbook."
)

print("#" * 90)
