from pathlib import Path

import pandas as pd

from idp_eval import (
    CoverageEvaluator,
    EvaluationCase,
    EvaluationFramework,
    FaithfulnessEvaluator,
)


# ============================================================
# 1. LOAD DATA
# ============================================================

PARQUET_PATH = Path("../epic_gen.parquet")

df = pd.read_parquet(PARQUET_PATH).fillna("")

print("Rows:", len(df))


# ============================================================
# 2. BUILD EVALUATION CASES
#
# FIRST RUN:
# context = theme_text
# output  = generated Epic from context@5 generation
# ============================================================

cases = []

for row_index, row in df.iterrows():

    # Prefer UUID because projectKey may repeat across rows.
    case_id = str(row.get("uuid", row_index))

    case = EvaluationCase(
        case_id=case_id,

        # Authoritative source for this first experiment
        context={
            "theme_text": row.get("theme_text", ""),
        },

        # Generated Epic being evaluated
        output={
            "epic_title": row.get(
                "gen_epic_title_context@5",
                "",
            ),
            "epic_description": row.get(
                "gen_epic_description_context@5",
                "",
            ),
            "epic_success_criteria": row.get(
                "gen_epic_successCriteria_context@5",
                "",
            ),
        },
    )

    cases.append(case)


print("Evaluation cases:", len(cases))


# ============================================================
# 3. FRAMEWORK
# ============================================================

framework = EvaluationFramework(
    judge=judge,
    evaluators=[
        CoverageEvaluator,
        FaithfulnessEvaluator,
    ],
    output="excel",
    excel_path="theme_text_vs_generated_epic.xlsx",
)


# ============================================================
# 4. RUN EVALUATION
# ============================================================

results = framework.evaluate_many(
    cases,
    metrics=[
        "coverage",
        "faithfulness",
    ],
    run_name="generated_epic_vs_theme_text",
    dataset_name="epic_gen.parquet",

    # Use this if your progress-bar change is already in your branch.
    show_progress=True,
)


# ============================================================
# 5. BUILD SIMPLE RESULTS DATAFRAME
# ============================================================

rows = []

for case, result in zip(cases, results):

    coverage = result["coverage"]
    faithfulness = result["faithfulness"]

    rows.append(
        {
            "case_id": case.case_id,

            "coverage_score": coverage.score,
            "coverage_label": coverage.label,
            "coverage_explanation": coverage.explanation,

            "faithfulness_score": faithfulness.score,
            "faithfulness_label": faithfulness.label,
            "faithfulness_explanation": faithfulness.explanation,

            # Optional explicit hallucination rate
            "hallucination_rate": (
                None
                if faithfulness.score is None
                else 1.0 - faithfulness.score
            ),
        }
    )


evaluation_df = pd.DataFrame(rows)

display(evaluation_df)


# ============================================================
# 6. OVERALL AVERAGES
# ============================================================

print()
print("OVERALL RESULTS")
print("=" * 70)

print(
    "Average Coverage:",
    evaluation_df["coverage_score"].mean(),
)

print(
    "Average Faithfulness:",
    evaluation_df["faithfulness_score"].mean(),
)

print(
    "Average Hallucination Rate:",
    evaluation_df["hallucination_rate"].mean(),
)
