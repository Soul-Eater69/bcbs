import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from idp_eval import (
    CoverageEvaluator,
    FaithfulnessEvaluator,
    EvaluationCase,
    EvaluationFramework,
    create_azure_judge,
)

from idp_eval.judges import AzureJudgeConfig


# =========================================================
# 1. CONFIG
# =========================================================

PARQUET_PATH = "../epic_gen.parquet"

EXCEL_PATH = "generated_epic_vs_theme_text_v4_exhaustive.xlsx"

# This is the source Epic key / row identifier in your parquet.
EPIC_KEY_COLUMN = "id"

RUN_NAME = "generated_epic_vs_theme_text_v4_exhaustive"
DATASET_NAME = "epic_gen.parquet"

# Retry failed operational/provider calls every 3 minutes.
RETRY_INTERVAL_SECONDS = 180


# =========================================================
# 2. LOAD ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# 3. CREATE AZURE JUDGE
# =========================================================

azure_config = AzureJudgeConfig(
    model=os.environ["IDP_MODEL"],
    azure_endpoint=os.environ["IDP_AZURE_ENDPOINT"],
    tenant_id=os.environ["IDP_AZURE_TENANT_ID"],
    client_id=os.environ["IDP_AZURE_CLIENT_ID"],
    client_secret=os.environ["IDP_AZURE_CLIENT_SECRET"],

    api_version="2024-12-01-preview",

    # Long timeout for GPT-5 / exhaustive evaluations
    timeout=180,

    proxy_url="http://bcproxy.hcscint.net:8080",

    verify_ssl=False,

    reasoning_effort=None,
)

judge = create_azure_judge(
    config=azure_config
)

print("Azure judge created")


# =========================================================
# 4. HELPER
#    CONVERT PARQUET / NUMPY VALUES TO NORMAL PYTHON
# =========================================================

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


# =========================================================
# 5. BUILD GENERATED EPIC OUTPUT
#
#    OUTPUT BEING EVALUATED:
#      - title
#      - description
#      - success criteria
# =========================================================

def build_generated_epic(row):
    return {
        "epic_title": to_python(
            row["gen_epic_title_context@5"]
        ),

        "epic_description": to_python(
            row["gen_epic_description_context@5"]
        ),

        "epic_success_criteria": to_python(
            row["gen_epic_successCriteria_context@5"]
        ),
    }


# =========================================================
# 6. LOAD PARQUET
# =========================================================

all_df = pd.read_parquet(
    PARQUET_PATH
)

# FULL RUN
df = all_df

# ---------------------------------------------------------
# For another small test instead, use:
#
# df = all_df.head(5)
# ---------------------------------------------------------

print(
    f"Loaded {len(df)} rows"
)


# =========================================================
# 7. BUILD EVALUATION CASES
#
#    RUN 1:
#
#       CONTEXT  = theme_text
#
#       OUTPUT   = generated Epic
#                  title
#                  description
#                  success criteria
#
# =========================================================

cases = []

for index, row in df.iterrows():

    # -----------------------------------------------------
    # CASE ID
    # -----------------------------------------------------

    if (
        EPIC_KEY_COLUMN in df.columns
        and pd.notna(row[EPIC_KEY_COLUMN])
    ):
        epic_key = str(
            to_python(
                row[EPIC_KEY_COLUMN]
            )
        )
    else:
        epic_key = str(index)

    # -----------------------------------------------------
    # BUILD CASE
    # -----------------------------------------------------

    case = EvaluationCase(
        case_id=epic_key,

        input=(
            "Generate an Epic from the supplied "
            "business theme."
        ),

        # =================================================
        # AUTHORITATIVE SOURCE FOR RUN 1
        # =================================================

        context={
            "theme_text": to_python(
                row["theme_text"]
            )
        },

        # =================================================
        # GENERATED EPIC BEING EVALUATED
        # =================================================

        output=build_generated_epic(
            row
        ),
    )

    cases.append(case)


print(
    f"Built {len(cases)} evaluation cases"
)

print("\nFirst case IDs:")

for case in cases[:5]:
    print(
        case.case_id
    )


# =========================================================
# 8. CREATE EVALUATION FRAMEWORK
#
#    max_items=None
#        -> exhaustive extraction
#
#    verbose=False
#        -> no full per-item audit trail exposed
#
#    reason_mode="overall"
#        -> one clean semantic explanation
#
# =========================================================

framework = EvaluationFramework(
    judge=judge,

    evaluators=[
        CoverageEvaluator(
            max_items=None,
            reason_mode="overall",
            verbose=False,
        ),

        FaithfulnessEvaluator(
            max_items=None,
            reason_mode="overall",
            verbose=False,
        ),
    ],

    # Excel checkpoint/output
    output="excel",

    excel_path=EXCEL_PATH,

    # Required for:
    # - restart/resume
    # - retry_until_complete
    resume=True,

    # Visible columns in Excel
    report_fields=[
        "context",
        "output",
    ],
)


# =========================================================
# 9. RUN EVALUATION
#
#    THEME_TEXT VS GENERATED EPIC
#
#    If Azure/provider fails:
#
#       error is checkpointed
#           ↓
#       finish current round
#           ↓
#       wait 180 sec
#           ↓
#       retry failed metrics only
#           ↓
#       successful metrics are skipped
#           ↓
#       repeat until complete
#
# =========================================================

results = framework.evaluate_many(
    cases,

    metrics=[
        "coverage",
        "faithfulness",
    ],

    run_name=RUN_NAME,

    dataset_name=DATASET_NAME,

    # Convert recognized operational/provider failures
    # into resumable error results.
    on_error="continue",

    # Keep retrying failed operational evaluations
    # until all are complete.
    retry_until_complete=True,

    # 3-minute cooldown between retry rounds.
    retry_interval_seconds=RETRY_INTERVAL_SECONDS,

    show_progress=True,
)


# =========================================================
# 10. FINAL SUMMARY
# =========================================================

print("\n")
print("=" * 80)
print("RUN 1 COMPLETE")
print("=" * 80)

print(
    f"Cases evaluated: {len(results)}"
)

print(
    f"Excel file: {EXCEL_PATH}"
)


# =========================================================
# 11. OPTIONAL SCORE SUMMARY
# =========================================================

coverage_scores = []
faithfulness_scores = []

for result in results:

    coverage = result["coverage"]
    faithfulness = result["faithfulness"]

    if coverage.score is not None:
        coverage_scores.append(
            coverage.score
        )

    if faithfulness.score is not None:
        faithfulness_scores.append(
            faithfulness.score
        )


if coverage_scores:
    print(
        "Average Coverage:",
        round(
            sum(coverage_scores)
            / len(coverage_scores),
            4,
        ),
    )


if faithfulness_scores:
    print(
        "Average Faithfulness:",
        round(
            sum(faithfulness_scores)
            / len(faithfulness_scores),
            4,
        ),
    )


print("=" * 80)
