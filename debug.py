import numpy as np
import pandas as pd

from idp_eval import (
    CoverageEvaluator,
    FaithfulnessEvaluator,
    EvaluationCase,
    EvaluationFramework,
)

# =========================================================
# CONFIG
# =========================================================

PARQUET_PATH = "epic_gen.parquet"

EXCEL_PATH = "generated_epic_vs_theme_text_v4_test5.xlsx"

# If your parquet has epic_id, it will be used.
# Otherwise dataframe index will be used.
EPIC_ID_COLUMN = "epic_id"


# =========================================================
# HELPER: CONVERT NUMPY/PARQUET VALUES TO NORMAL PYTHON
# =========================================================

def to_python(value):
    if isinstance(value, np.ndarray):
        return [to_python(v) for v in value.tolist()]

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(k): to_python(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [to_python(v) for v in value]

    return value


# =========================================================
# GENERATED EPIC OUTPUT
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
# LOAD ONLY FIRST 5 ROWS
# =========================================================

df = pd.read_parquet(PARQUET_PATH).head(5)

print(f"Loaded {len(df)} rows for test run")


# =========================================================
# BUILD CASES
#
# RUN 1:
# generated epic VS theme_text
# =========================================================

cases = []

for index, row in df.iterrows():

    if EPIC_ID_COLUMN in df.columns and pd.notna(row[EPIC_ID_COLUMN]):
        case_id = str(row[EPIC_ID_COLUMN])
    else:
        case_id = str(index)

    case = EvaluationCase(
        case_id=case_id,

        input="Generate an Epic from the supplied business theme.",

        # Authoritative source for RUN 1
        context={
            "theme_text": to_python(
                row["theme_text"]
            )
        },

        # Generated Epic being evaluated
        output=build_generated_epic(row),
    )

    cases.append(case)


print(f"Built {len(cases)} evaluation cases")


# =========================================================
# FRAMEWORK
#
# IMPORTANT:
# max_items=None = EXHAUSTIVE MODE
#
# It does NOT mean 5 evaluator items.
#
# We are testing only 5 dataframe rows while allowing
# Coverage/Faithfulness to extract all material atomic
# items/claims from each row.
# =========================================================

framework = EvaluationFramework(
    judge=judge,  # use your already configured judge

    evaluators=[
        CoverageEvaluator(
            max_items=None,
            reason_mode="overall",
            verbose=True,
        ),

        FaithfulnessEvaluator(
            max_items=None,
            reason_mode="overall",
            verbose=True,
        ),
    ],

    # Phoenix + Excel
    output="both",

    excel_path=EXCEL_PATH,

    # Allows safe restart/resume
    resume=True,

    # Keep actual source/output visible in Excel
    report_fields=[
        "context",
        "output",
    ],
)


# =========================================================
# RUN EVALUATION
# =========================================================

results = framework.evaluate_many(
    cases,

    metrics=[
        "coverage",
        "faithfulness",
    ],

    run_name="generated_epic_vs_theme_text_v4_exhaustive_test5",

    dataset_name="epic_gen.parquet",

    # Provider failures get recorded without losing
    # already-completed cases.
    on_error="continue",

    show_progress=True,
)


# =========================================================
# PRINT RESULTS
# =========================================================

print("\n================ RESULTS ================\n")

for case, result in zip(cases, results):

    print(f"Case: {case.case_id}")

    coverage = result["coverage"]
    faithfulness = result["faithfulness"]

    print(
        "Coverage:",
        coverage.score,
        "|",
        coverage.label,
    )

    print(
        "Coverage explanation:",
        coverage.explanation,
    )

    print(
        "Faithfulness:",
        faithfulness.score,
        "|",
        faithfulness.label,
    )

    print(
        "Faithfulness explanation:",
        faithfulness.explanation,
    )

    print("-" * 80)


print("\nTest run complete.")
print(f"Excel saved to: {EXCEL_PATH}")
