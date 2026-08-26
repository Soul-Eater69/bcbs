# =========================================================
# RETRIEVAL EVAL
# THEME_TEXT -> THEME_ONLY_AT_5
# METRIC: RELEVANCE@5
# =========================================================

from idp_eval import (
    EvaluationCase,
    EvaluationFramework,
    RelevanceAtKEvaluator,
)


# =========================================================
# CONFIG
# =========================================================

RELEVANCE_EXCEL_PATH = "theme_retrieval_relevance_at_5.xlsx"

RELEVANCE_RUN_NAME = "theme_only_retrieval_relevance_at_5"

DATASET_NAME = "epic_gen.parquet"


# =========================================================
# BUILD RETRIEVAL EVALUATION CASES
# =========================================================

relevance_cases = []

for index, row in all_df.iterrows():

    # -----------------------------------------------------
    # CASE ID
    # -----------------------------------------------------

    if (
        EPIC_KEY_COLUMN in all_df.columns
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
    # RETRIEVED THEMES
    # rank order must be preserved
    # -----------------------------------------------------

    retrieved_themes = to_python(
        row["theme_only_at_5"]
    )

    # -----------------------------------------------------
    # CASE
    #
    # input               = retrieval query
    # retrieved_documents = ranked retrieved themes
    # -----------------------------------------------------

    case = EvaluationCase(
        case_id=epic_key,

        # QUERY / INFORMATION NEED
        input=to_python(
            row["theme_text"]
        ),

        # TOP-5 RETRIEVED HISTORICAL THEMES
        retrieved_documents=retrieved_themes,
    )

    relevance_cases.append(case)


print(
    f"Built {len(relevance_cases)} retrieval evaluation cases"
)

print("\nFirst case:")
print("Case ID:", relevance_cases[0].case_id)
print("Query:", relevance_cases[0].input)
print(
    "Retrieved themes:",
    len(relevance_cases[0].retrieved_documents),
)


# =========================================================
# FRAMEWORK
# =========================================================

relevance_framework = EvaluationFramework(
    judge=judge,

    evaluators=[
        RelevanceAtKEvaluator(
            k=5,
            verbose=False,
        ),
    ],

    output="excel",

    excel_path=RELEVANCE_EXCEL_PATH,

    resume=True,

    report_fields=[
        "input",
        "retrieved_documents",
    ],
)


# =========================================================
# RUN RELEVANCE@5
# =========================================================

relevance_results = relevance_framework.evaluate_many(
    relevance_cases,

    metrics=[
        "relevance_at_5",
    ],

    run_name=RELEVANCE_RUN_NAME,

    dataset_name=DATASET_NAME,

    on_error="continue",

    # Keep retrying Azure/provider operational failures
    retry_until_complete=True,

    # 3 minutes between retry rounds
    retry_interval_seconds=180,

    show_progress=True,
)


# =========================================================
# SUMMARY
# =========================================================

scores = [
    result["relevance_at_5"].score
    for result in relevance_results
    if result["relevance_at_5"].score is not None
]

print("\n" + "=" * 80)
print("THEME RETRIEVAL RELEVANCE@5 COMPLETE")
print("=" * 80)

print(
    f"Cases evaluated: {len(relevance_results)}"
)

if scores:
    print(
        "Average Relevance@5:",
        round(
            sum(scores) / len(scores),
            4,
        ),
    )

print(
    f"Excel file: {RELEVANCE_EXCEL_PATH}"
)

print("=" * 80)
