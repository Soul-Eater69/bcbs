from idp_eval import (
    EvaluationFramework,
    NDCGAtKEvaluator,
)

framework = EvaluationFramework(
    judge=judge,
    evaluators=[
        NDCGAtKEvaluator(
            k=5,
            verbose=True,
        ),
    ],

    output="excel",
    excel_path="ndcg_at_5.xlsx",
    resume=True,

    report_fields=[
        "input",
        "retrieved_documents",
    ],
)

results = framework.evaluate_many(
    cases,
    metrics=["ndcg_at_5"],
    run_name="ndcg_at_5",
    dataset_name="epic_gen.parquet",
    on_error="continue",
    show_progress=True,
)

for case, result in zip(cases, results):
    ndcg = result["ndcg_at_5"]

    print("Case:", case.case_id)
    print("nDCG@5:", ndcg.score)
    print("Label:", ndcg.label)
    print("Explanation:", ndcg.explanation)
    print("-" * 80)
