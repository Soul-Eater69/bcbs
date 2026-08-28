"""One-off updater for the five L3 experiment notebooks.

Changes only dataset/evaluation population behavior:
- run every Theme from epic_gen.csv (no nrows cap),
- treat blank/no-L3 Jira rows as unlabeled, not negative examples,
- keep end-to-end metrics for labeled Epics,
- add a selector-only score for fully retrievable GT Epics,
- flag no-candidate / partial-GT cases as selector-ineligible.

Prompt text and experiment payload definitions are intentionally untouched.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "01_theme_stage.ipynb",
    ROOT / "02_full_context.ipynb",
    ROOT / "03_no_theme_description.ipynb",
    ROOT / "04_no_theme.ipynb",
    ROOT / "05_full_with_hierarchy.ipynb",
]

ALL_THEME_IDS = '''THEME_IDS = (
    pd.read_csv(
        THEME_PATH,
        usecols=["key"],
        encoding="cp1252",
        encoding_errors="replace",
        dtype=str,
    )["key"]
    .dropna()
    .str.strip()
    .loc[lambda values: values.ne("")]
    .drop_duplicates()
    .tolist()
)
'''

EVALUATION_CELL = '''def ground_truth_by_epic():
    # Ground truth is loaded only after every prediction has been produced.
    ground_truth_frame = read_table(GROUND_TRUTH_PATH).copy()
    ground_truth_frame["l3_capability_id"] = (
        ground_truth_frame["l3_capability_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Blank/no-capability rows are unlabeled examples, not negative labels.
    ground_truth_frame = ground_truth_frame.loc[
        ground_truth_frame["l3_capability_id"].ne("")
    ]

    return {
        epic_key: set(group["l3_capability_id"])
        for epic_key, group in ground_truth_frame.groupby(
            "epic_key",
            sort=False,
        )
    }


def evaluate_predictions(prediction_frame):
    truth_by_epic = ground_truth_by_epic()
    result_rows = []

    for row in prediction_frame.to_dict(orient="records"):
        predicted_ids = set(json.loads(row["predicted_l3_ids"]))
        available_candidate_ids = set(
            json.loads(row["available_candidate_l3_ids"])
        )
        truth_ids = truth_by_epic.get(row["epic_key"])

        if truth_ids is None:
            # Keep for diagnostics, but do not score an unlabeled Epic.
            if row["status"] != "error":
                row["status"] = "missing_ground_truth"
            metrics = {
                "exact_match": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "predicted_count": len(predicted_ids),
                "truth_count": None,
            }
            available_truth_ids = None
            availability_fraction = None
        else:
            available_truth_ids = truth_ids & available_candidate_ids
            availability_fraction = (
                len(available_truth_ids) / len(truth_ids)
                if truth_ids
                else 1.0
            )
            if row["status"] == "error":
                metrics = {
                    "exact_match": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "predicted_count": len(predicted_ids),
                    "truth_count": len(truth_ids),
                }
            else:
                # End-to-end score includes candidate-retrieval misses.
                metrics = score_sets(predicted_ids, truth_ids)

        row["ground_truth_l3_ids"] = (
            json.dumps(sorted(truth_ids))
            if truth_ids is not None
            else None
        )
        row["gt_available_candidate_l3_ids"] = (
            json.dumps(sorted(available_truth_ids))
            if available_truth_ids is not None
            else None
        )
        row["gt_candidate_available_count"] = (
            len(available_truth_ids)
            if available_truth_ids is not None
            else None
        )
        row["gt_candidate_availability_fraction"] = availability_fraction

        # Fair prompt/selector comparison: every GT ID must have been offered.
        selector_eligible = (
            truth_ids is not None
            and row["status"] == "ok"
            and bool(available_candidate_ids)
            and availability_fraction == 1.0
        )
        row["selector_eligible"] = selector_eligible

        if truth_ids is None:
            exclusion_reason = "missing_ground_truth"
        elif row["status"] == "error":
            exclusion_reason = "error"
        elif not row.get("stage_ids") or row.get("stage_ids") == "[]":
            exclusion_reason = "no_stage"
        elif not available_candidate_ids:
            exclusion_reason = "no_candidates"
        elif availability_fraction != 1.0:
            exclusion_reason = "gt_not_fully_retrievable"
        else:
            exclusion_reason = ""
        row["selector_exclusion_reason"] = exclusion_reason

        for metric_name in ("exact_match", "precision", "recall", "f1"):
            row[f"selector_{metric_name}"] = (
                metrics[metric_name]
                if selector_eligible
                else None
            )

        row.update(metrics)
        result_rows.append(row)

    return pd.DataFrame(result_rows)


def evaluation_summary(result_frame):
    # End-to-end: every labeled, non-error Epic.
    labeled = result_frame.loc[result_frame["exact_match"].notna()]

    # Selector-only: only rows where all GT L3s were actually candidates.
    selector = result_frame.loc[result_frame["selector_eligible"]]

    def summarize_scope(frame, scope):
        return {
            "scope": scope,
            "evaluated_epics": len(frame),
            "exact_match_accuracy": (
                frame["exact_match"].mean() if len(frame) else 0.0
            ),
            "mean_precision": (
                frame["precision"].mean() if len(frame) else 0.0
            ),
            "mean_recall": (
                frame["recall"].mean() if len(frame) else 0.0
            ),
            "mean_f1": (
                frame["f1"].mean() if len(frame) else 0.0
            ),
        }

    summary = pd.DataFrame(
        [
            summarize_scope(labeled, "end_to_end_labeled"),
            summarize_scope(
                selector,
                "selector_only_fully_retrievable",
            ),
        ]
    )

    diagnostics = pd.DataFrame(
        [
            {
                "prediction_rows": len(result_frame),
                "labeled_epics": len(labeled),
                "selector_eligible_epics": len(selector),
                "missing_ground_truth": int(
                    (
                        result_frame["status"]
                        == "missing_ground_truth"
                    ).sum()
                ),
                "no_candidates": int(
                    (
                        result_frame["selector_exclusion_reason"]
                        == "no_candidates"
                    ).sum()
                ),
                "no_stage": int(
                    (
                        result_frame["selector_exclusion_reason"]
                        == "no_stage"
                    ).sum()
                ),
                "partial_or_zero_gt_availability": int(
                    (
                        result_frame[
                            "gt_candidate_availability_fraction"
                        ].notna()
                        & result_frame[
                            "gt_candidate_availability_fraction"
                        ].lt(1.0)
                    ).sum()
                ),
                "errors": int(
                    (result_frame["status"] == "error").sum()
                ),
            }
        ]
    )
    return summary, diagnostics


results = evaluate_predictions(predictions)
summary, diagnostics = evaluation_summary(results)

print("Evaluation summary")
display(summary)

print("Dataset / retrieval diagnostics")
display(diagnostics)

display(results.head(20))

output_path = save_results_excel(
    results,
    EXPERIMENT_NAME,
    "results",
)
print(f"Saved {output_path}")
'''

EVAL_MARKDOWN = '''## Ground-truth evaluation

Ground truth enters only in this post-prediction section.

Two views are reported:
- **end_to_end_labeled**: every Epic with non-empty Jira L3 ground truth;
- **selector_only_fully_retrievable**: only Epics where every GT L3 was present in the supplied candidate population.

Blank/no-L3 ground-truth rows are treated as unlabeled and are not scored. No-candidate and partial/zero-GT-availability rows remain visible as retrieval diagnostics but are excluded from the selector-only prompt comparison.'''


def source_text(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def replace_source(cell, source):
    cell["source"] = source


def patch_theme_population(source):
    start = source.index("THEME_IDS =")
    end = source.index("VALUE_STREAM_STAGE_FIELD_ID", start)
    return source[:start] + ALL_THEME_IDS + "\n" + source[end:]


def patch_predict_for_stage(source):
    # Avoid an unnecessary LLM call if a Stage has no candidate L3s.
    if "if not candidates:" in source or "if not candidate_rows:" in source:
        return source

    candidates_line = "    candidates = candidate_rows_for_stage(stage_id)\n"
    if candidates_line in source:
        insertion = candidates_line + '''    if not candidates:\n        return {\n            "stage": stage,\n            "candidates": [],\n            "user_prompt": build_user_prompt(theme, epic, stage, []),\n            "raw_response": None,\n            "selections": [],\n        }\n'''
        return source.replace(candidates_line, insertion, 1)

    candidate_rows_line = "    candidate_rows = candidate_rows_for_stage(stage_id)\n"
    if candidate_rows_line in source:
        insertion = candidate_rows_line + '''    if not candidate_rows:\n        return {\n            "stage": stage,\n            "candidates": [],\n            "user_prompt": build_user_prompt(theme, epic, stage, []),\n            "raw_response": None,\n            "selections": [],\n        }\n'''
        return source.replace(candidate_rows_line, insertion, 1)

    raise AssertionError("Could not find candidate retrieval in predict_for_stage")


def patch_notebook(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))

    config_cell = next(
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "THEME_IDS =" in source_text(cell)
        and "VALUE_STREAM_STAGE_FIELD_ID" in source_text(cell)
    )
    replace_source(
        config_cell,
        patch_theme_population(source_text(config_cell)),
    )

    prediction_cell = next(
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "def predict_for_stage" in source_text(cell)
    )
    replace_source(
        prediction_cell,
        patch_predict_for_stage(source_text(prediction_cell)),
    )

    eval_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code"
        and "def ground_truth_by_epic" in source_text(cell)
    )
    replace_source(notebook["cells"][eval_index], EVALUATION_CELL)

    if eval_index > 0 and notebook["cells"][eval_index - 1]["cell_type"] == "markdown":
        replace_source(notebook["cells"][eval_index - 1], EVAL_MARKDOWN)

    path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def system_prompt_from(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n\n".join(
        source_text(cell)
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name)
                and target.id == "SYSTEM_PROMPT"
                for target in node.targets
            ):
                return ast.literal_eval(node.value)
    raise AssertionError(f"SYSTEM_PROMPT missing in {path.name}")


def validate_notebooks():
    prompts = []
    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile(
                    source_text(cell),
                    f"{path.name}:cell-{index}",
                    "exec",
                )
        all_code = "\n".join(
            source_text(cell)
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        assert "nrows=25" not in all_code
        assert "nrows = 25" not in all_code
        assert "selector_eligible" in all_code
        assert "selector_only_fully_retrievable" in all_code
        assert "gt_candidate_availability_fraction" in all_code
        prompts.append(system_prompt_from(path))

    assert len(set(prompts)) == 1, "SYSTEM_PROMPT changed across experiments"


if __name__ == "__main__":
    for notebook_path in NOTEBOOKS:
        patch_notebook(notebook_path)
    validate_notebooks()
    print("Updated and validated all five L3 experiment notebooks.")
