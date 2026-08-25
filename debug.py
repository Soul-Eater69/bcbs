# ============================================================
# RUN 3
# GENERATED EPIC vs (theme_text + context@5)
#
# NEW Excel:
#   generated_epic_vs_theme_plus_context_at_5.xlsx
#
# - Coverage + Faithfulness
# - Cases 1 -> 50
# - One case at a time
# - Operational failure / 429:
#       wait 3 minutes
#       retry SAME case
#       keep retrying until success
# - Programming/config errors still stop immediately
# ============================================================

import asyncio
import time
import traceback
import numpy as np

from idp_eval import EvaluationCase, EvaluationFramework
from idp_eval.evaluators import (
    CoverageEvaluator,
    FaithfulnessEvaluator,
)

from phoenix.evals.rate_limiters import (
    RateLimitError as PhoenixRateLimitError,
)


# ============================================================
# OPTIONAL PROVIDER ERROR TYPES
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

EXCEL_PATH = "generated_epic_vs_theme_plus_context_at_5.xlsx"

RUN_NAME = "generated_epic_vs_theme_plus_context_at_5"

DATASET_NAME = "epic_gen.parquet"

START_CASE = 1

RATE_LIMIT_WAIT_SECONDS = 180       # 3 minutes
OTHER_OPERATIONAL_WAIT_SECONDS = 60


# ============================================================
# CONVERT PARQUET / NUMPY VALUES
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
# VALIDATE REQUIRED DATA
# ============================================================

required_columns = [
    "theme_text",
    "context@5",
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise KeyError(
        f"Missing required columns: {missing_columns}"
    )


if len(df) != len(cases):
    raise ValueError(
        f"df contains {len(df)} rows but cases contains "
        f"{len(cases)} cases. They must align row-for-row."
    )


# ============================================================
# BUILD RUN 3 CASES
#
# SAME generated Epic output.
#
# Authoritative context now contains BOTH:
#
#   1. theme_text
#   2. context@5
# ============================================================

combined_cases = []


for position, (_, row) in enumerate(df.iterrows()):

    original_case = cases[position]

    combined_case = EvaluationCase(

        input=original_case.input,

        # ====================================================
        # RUN 3 CONTEXT
        # ====================================================

        context={
            "theme_text": to_python(
                row["theme_text"]
            ),
            "context_at_5": to_python(
                row["context@5"]
            ),
        },

        # ====================================================
        # SAME GENERATED EPIC OUTPUT AS RUNS 1 AND 2
        # ====================================================

        output=original_case.output,

        instructions=original_case.instructions,

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

    combined_cases.append(
        combined_case
    )


STOP_CASE = len(combined_cases)


# ============================================================
# CREATE A NEW FRAMEWORK
#
# IMPORTANT:
# This gives Run 3 its OWN Excel workbook.
# ============================================================

framework_run3 = EvaluationFramework(

    judge=judge,

    evaluators=[
        CoverageEvaluator(
            verbose=True
        ),
        FaithfulnessEvaluator(
            verbose=True
        ),
    ],

    output="excel",

    excel_path=EXCEL_PATH,
)


# ============================================================
# SANITY CHECK
# ============================================================

print("=" * 90)

print("RUN 3 SETUP")

print("=" * 90)

print(
    "Run name      :",
    RUN_NAME,
)

print(
    "Excel         :",
    EXCEL_PATH,
)

print(
    "Dataset       :",
    DATASET_NAME,
)

print(
    "Total cases   :",
    len(combined_cases),
)

print(
    "First case ID :",
    combined_cases[0].case_id,
)

print(
    "Last case ID  :",
    combined_cases[-1].case_id,
)

print(
    "\nContext used:"
)

print(
    "  theme_text + context@5"
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
    set(operational_types)
)


# ============================================================
# FIND TEMPORARY ERROR ANYWHERE IN EXCEPTION CHAIN
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

        if current.__cause__ is not None:
            current = current.__cause__
        else:
            current = current.__context__

    return None


# ============================================================
# RATE-LIMIT TYPE CHECK
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
# RUN
# ============================================================

print("\n")
print("#" * 90)

print(
    "RUN 3: GENERATED EPIC "
    "vs THEME TEXT + CONTEXT@5"
)

print(
    f"Cases: {START_CASE} -> {STOP_CASE}"
)

print(
    "Metrics: coverage + faithfulness"
)

print(
    "429 throttle cooldown: "
    f"{RATE_LIMIT_WAIT_SECONDS}s "
    f"({RATE_LIMIT_WAIT_SECONDS / 60:.0f} minutes)"
)

print(
    "Operational errors retry automatically "
    "until the case succeeds."
)

print("#" * 90)


overall_started = time.perf_counter()

completed_this_run = 0
total_retries = 0


# ============================================================
# ONE CASE AT A TIME
# ============================================================

for human_case_number in range(
    START_CASE,
    STOP_CASE + 1,
):

    case = combined_cases[
        human_case_number - 1
    ]

    attempt = 0


    while True:

        attempt += 1

        print("\n")
        print("=" * 90)

        print(
            f"RUN 3 | CASE "
            f"{human_case_number}/{STOP_CASE} "
            f"| attempt {attempt}"
        )

        print(
            f"case_id: {case.case_id}"
        )

        print("=" * 90)


        started = time.perf_counter()


        try:

            # =================================================
            # ONE CASE
            #
            # Coverage + Faithfulness
            #
            # max_concurrency=1 protects PTU capacity.
            # =================================================

            result = await framework_run3.a_evaluate(

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


            # =================================================
            # SUCCESS
            # =================================================

            print("\n")
            print("✓" * 45)

            print(
                f"✓ RUN 3 CASE "
                f"{human_case_number} COMPLETE"
            )

            print(
                f"attempts       : {attempt}"
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
                f"✓ saved to {EXCEL_PATH}"
            )

            print("✓" * 45)


            # Success -> next case
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
            # REAL ERROR
            # =================================================

            if operational_error is None:

                print("\n")
                print("X" * 90)

                print(
                    f"NON-OPERATIONAL ERROR "
                    f"ON RUN 3 CASE "
                    f"{human_case_number}"
                )

                print(
                    f"case_id : "
                    f"{case.case_id}"
                )

                print(
                    f"type    : "
                    f"{type(exc).__name__}"
                )

                print(
                    f"message : {exc}"
                )

                print(
                    "Stopping because this does not "
                    "look like a temporary provider error."
                )

                print("X" * 90)

                traceback.print_exc()

                raise


            # =================================================
            # TEMPORARY ERROR
            # =================================================

            total_retries += 1


            if isinstance(
                operational_error,
                RATE_LIMIT_TYPES,
            ):

                wait_seconds = (
                    RATE_LIMIT_WAIT_SECONDS
                )

                failure_kind = (
                    "rate_limit"
                )

            else:

                wait_seconds = (
                    OTHER_OPERATIONAL_WAIT_SECONDS
                )

                failure_kind = (
                    "temporary_provider_error"
                )


            print("\n")
            print("!" * 90)

            print(
                f"⚠ RUN 3 CASE "
                f"{human_case_number} "
                f"ATTEMPT {attempt} "
                f"FAILED TEMPORARILY"
            )

            print(
                f"kind     : "
                f"{failure_kind}"
            )

            print(
                f"error    : "
                f"{type(operational_error).__name__}"
            )

            print(
                f"message  : "
                f"{operational_error}"
            )

            print(
                f"elapsed  : "
                f"{elapsed:.1f}s"
            )

            print(
                f"cooldown : "
                f"{wait_seconds}s "
                f"({wait_seconds / 60:.1f} minutes)"
            )

            print(
                "action   : retry same case automatically"
            )

            print("!" * 90)


            # =================================================
            # COUNTDOWN
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


                remaining -= (
                    sleep_for
                )


            print(
                f"\nRetrying Run 3 case "
                f"{human_case_number} now..."
            )


# ============================================================
# COMPLETE
# ============================================================

overall_elapsed = (
    time.perf_counter()
    - overall_started
)


print("\n")
print("#" * 90)

print(
    "✓ RUN 3 COMPLETE"
)

print(
    f"✓ Cases 1-{STOP_CASE} complete"
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
    f"Excel             : "
    f"{EXCEL_PATH}"
)

print(
    f"run_name          : "
    f"{RUN_NAME}"
)

print("#" * 90)
