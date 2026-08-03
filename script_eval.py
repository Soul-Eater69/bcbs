from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION — UPDATE THESE FILE PATHS
# ============================================================

BATCH_FILES = {
    "batch_1": "batch_1_results.json",
    "batch_2": "batch_2_results.json",
    "batch_3": "batch_3_results.json",
}

OUTPUT_DIR = Path("stage_coverage_eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STAGE_FIELDS = [
    "stage_name",
    "stage_description",
    "entrance_criteria",
    "exit_criteria",
]

EPIC_FIELDS = [
    "title",
    "description",
    "success_criteria",
]


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def extract_records(data: Any) -> list[dict]:
    """
    Supports:
    1. [record, record, ...]
    2. {"results": [...]}
    3. {"data": [...]}
    4. {"EPIC-1": {...}, "EPIC-2": {...}}
    5. A single result object
    """
    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON list or dictionary, got {type(data).__name__}."
        )

    possible_list_keys = [
        "results",
        "data",
        "items",
        "records",
        "evaluations",
        "outputs",
    ]

    for key in possible_list_keys:
        if isinstance(data.get(key), list):
            return data[key]

    if "epic_id" in data and "stage_field_counts" in data:
        return [data]

    if data and all(isinstance(value, dict) for value in data.values()):
        records = []

        for epic_id, result in data.items():
            record = dict(result)
            record.setdefault("epic_id", str(epic_id))
            records.append(record)

        return records

    raise ValueError(
        "Could not identify the records inside the JSON file. "
        "Inspect the file's top-level structure."
    )


def safe_integer(value: Any) -> int | float:
    """Return a non-negative integer or NaN."""
    if value is None or isinstance(value, bool):
        return np.nan

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(numeric) or not numeric.is_integer():
        return np.nan

    return int(numeric)


def safe_boolean(value: Any) -> bool | float:
    """Convert common Boolean formats or return NaN."""
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {"true", "yes", "1"}:
            return True

        if normalized in {"false", "no", "0"}:
            return False

    if value in {0, 1}:
        return bool(value)

    return np.nan


def safe_divide(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator <= 0:
        return np.nan

    return numerator / denominator


def classify_consistency(
    coverage_range: float,
    number_of_runs: int,
) -> str:
    if number_of_runs < len(BATCH_FILES):
        return "missing_run"

    if pd.isna(coverage_range):
        return "not_evaluable"

    if coverage_range <= 0.10:
        return "stable"

    if coverage_range <= 0.25:
        return "moderate_variation"

    return "unstable"


def round_numeric_columns(
    dataframe: pd.DataFrame,
    digits: int = 3,
) -> pd.DataFrame:
    result = dataframe.copy()
    numeric_columns = result.select_dtypes(include=[np.number]).columns
    result[numeric_columns] = result[numeric_columns].round(digits)
    return result


def print_table(title: str, dataframe: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    if dataframe.empty:
        print("No data available.")
    else:
        print(round_numeric_columns(dataframe).to_string(index=False))


# ============================================================
# 3. LOAD AND NORMALIZE THE THREE BATCHES
# ============================================================

detail_rows = []
evidence_rows = []
issues = []
batch_metadata = []

for batch_name, file_path in BATCH_FILES.items():
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {batch_name}: {path.resolve()}"
        )

    with path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    records = extract_records(raw_data)

    raw_epic_ids = [
        str(record.get("epic_id", "")).strip()
        for record in records
        if isinstance(record, dict)
    ]

    valid_ids = [epic_id for epic_id in raw_epic_ids if epic_id]
    duplicate_ids = [
        epic_id
        for epic_id, count in Counter(valid_ids).items()
        if count > 1
    ]

    batch_metadata.append(
        {
            "batch": batch_name,
            "file_path": str(path),
            "records_loaded": len(records),
            "unique_nonempty_epic_ids": len(set(valid_ids)),
            "missing_epic_ids": len(records) - len(valid_ids),
            "duplicate_epic_id_count": len(duplicate_ids),
        }
    )

    for duplicate_id in duplicate_ids:
        issues.append(
            {
                "batch": batch_name,
                "epic_id": duplicate_id,
                "issue_type": "duplicate_epic_id",
                "details": "The epic ID appears more than once in this batch.",
            }
        )

    for row_number, record in enumerate(records):
        if not isinstance(record, dict):
            issues.append(
                {
                    "batch": batch_name,
                    "epic_id": f"ROW-{row_number}",
                    "issue_type": "invalid_record",
                    "details": "The record is not a JSON object.",
                }
            )
            continue

        epic_id = str(record.get("epic_id", "")).strip()

        if not epic_id:
            epic_id = f"MISSING-ID-ROW-{row_number}"
            issues.append(
                {
                    "batch": batch_name,
                    "epic_id": epic_id,
                    "issue_type": "missing_epic_id",
                    "details": "The record does not contain a valid epic_id.",
                }
            )

        counts = record.get("stage_field_counts") or {}
        locations = record.get("stage_usage_locations") or {}
        evidence = record.get("evidence") or {}

        row = {
            "batch": batch_name,
            "epic_id": epic_id,
        }

        all_evidence_locations = set()

        for stage_field in STAGE_FIELDS:
            count_block = counts.get(stage_field) or {}

            total_items = safe_integer(count_block.get("total_items"))
            covered_items = safe_integer(count_block.get("covered_items"))

            row[f"{stage_field}_total_items"] = total_items
            row[f"{stage_field}_covered_items"] = covered_items
            row[f"{stage_field}_coverage"] = safe_divide(
                covered_items,
                total_items,
            )

            if pd.isna(total_items) or pd.isna(covered_items):
                issues.append(
                    {
                        "batch": batch_name,
                        "epic_id": epic_id,
                        "issue_type": "missing_or_invalid_count",
                        "details": (
                            f"{stage_field} has an invalid total_items "
                            "or covered_items value."
                        ),
                    }
                )
            else:
                if total_items < 0 or covered_items < 0:
                    issues.append(
                        {
                            "batch": batch_name,
                            "epic_id": epic_id,
                            "issue_type": "negative_count",
                            "details": (
                                f"{stage_field} contains a negative count."
                            ),
                        }
                    )

                if covered_items > total_items:
                    issues.append(
                        {
                            "batch": batch_name,
                            "epic_id": epic_id,
                            "issue_type": "covered_greater_than_total",
                            "details": (
                                f"{stage_field}: covered_items="
                                f"{covered_items}, total_items={total_items}."
                            ),
                        }
                    )

            stage_evidence = evidence.get(stage_field) or []

            if not isinstance(stage_evidence, list):
                issues.append(
                    {
                        "batch": batch_name,
                        "epic_id": epic_id,
                        "issue_type": "invalid_evidence_array",
                        "details": (
                            f"Evidence for {stage_field} is not a list."
                        ),
                    }
                )
                stage_evidence = []

            valid_evidence_count = 0

            for evidence_number, evidence_item in enumerate(stage_evidence):
                if not isinstance(evidence_item, dict):
                    issues.append(
                        {
                            "batch": batch_name,
                            "epic_id": epic_id,
                            "issue_type": "invalid_evidence_item",
                            "details": (
                                f"{stage_field} evidence item "
                                f"{evidence_number} is not an object."
                            ),
                        }
                    )
                    continue

                epic_field = evidence_item.get("epic_field")
                text = evidence_item.get("text")

                if epic_field not in EPIC_FIELDS:
                    issues.append(
                        {
                            "batch": batch_name,
                            "epic_id": epic_id,
                            "issue_type": "invalid_evidence_location",
                            "details": (
                                f"{stage_field} evidence uses invalid "
                                f"epic_field={epic_field!r}."
                            ),
                        }
                    )
                    continue

                if not isinstance(text, str) or not text.strip():
                    issues.append(
                        {
                            "batch": batch_name,
                            "epic_id": epic_id,
                            "issue_type": "empty_evidence_text",
                            "details": (
                                f"{stage_field} contains empty evidence text."
                            ),
                        }
                    )
                    continue

                valid_evidence_count += 1
                all_evidence_locations.add(epic_field)

                evidence_rows.append(
                    {
                        "batch": batch_name,
                        "epic_id": epic_id,
                        "stage_field": stage_field,
                        "epic_field": epic_field,
                        "evidence_text": text,
                    }
                )

            row[f"{stage_field}_evidence_count"] = valid_evidence_count

            if not pd.isna(covered_items):
                if covered_items > 0 and valid_evidence_count == 0:
                    issues.append(
                        {
                            "batch": batch_name,
                            "epic_id": epic_id,
                            "issue_type": "covered_without_evidence",
                            "details": (
                                f"{stage_field} has covered_items > 0 "
                                "but no valid evidence."
                            ),
                        }
                    )

                if covered_items == 0 and valid_evidence_count > 0:
                    issues.append(
                        {
                            "batch": batch_name,
                            "epic_id": epic_id,
                            "issue_type": "evidence_with_zero_coverage",
                            "details": (
                                f"{stage_field} has evidence but "
                                "covered_items is 0."
                            ),
                        }
                    )

        total_columns = [
            f"{stage_field}_total_items"
            for stage_field in STAGE_FIELDS
        ]
        covered_columns = [
            f"{stage_field}_covered_items"
            for stage_field in STAGE_FIELDS
        ]

        total_values = [row[column] for column in total_columns]
        covered_values = [row[column] for column in covered_columns]

        row["overall_total_items"] = (
            np.nan
            if all(pd.isna(value) for value in total_values)
            else np.nansum(total_values)
        )

        row["overall_covered_items"] = (
            np.nan
            if all(pd.isna(value) for value in covered_values)
            else np.nansum(covered_values)
        )

        row["overall_coverage"] = safe_divide(
            row["overall_covered_items"],
            row["overall_total_items"],
        )

        for epic_field in EPIC_FIELDS:
            supplied_flag = safe_boolean(locations.get(epic_field))
            row[f"location_{epic_field}"] = supplied_flag

            if pd.isna(supplied_flag):
                issues.append(
                    {
                        "batch": batch_name,
                        "epic_id": epic_id,
                        "issue_type": "missing_or_invalid_location_flag",
                        "details": (
                            f"stage_usage_locations.{epic_field} "
                            "is missing or invalid."
                        ),
                    }
                )
                continue

            expected_flag = epic_field in all_evidence_locations

            if supplied_flag != expected_flag:
                issues.append(
                    {
                        "batch": batch_name,
                        "epic_id": epic_id,
                        "issue_type": "location_evidence_mismatch",
                        "details": (
                            f"{epic_field}: supplied={supplied_flag}, "
                            f"expected_from_evidence={expected_flag}."
                        ),
                    }
                )

        detail_rows.append(row)


detail_df = pd.DataFrame(detail_rows)
evidence_df = pd.DataFrame(evidence_rows)
issues_df = pd.DataFrame(issues)
batch_metadata_df = pd.DataFrame(batch_metadata)


# ============================================================
# 4. DATA-QUALITY SUMMARY
# ============================================================

if issues_df.empty:
    issue_counts = pd.DataFrame(index=batch_metadata_df["batch"])
else:
    issue_counts = (
        issues_df
        .groupby(["batch", "issue_type"])
        .size()
        .unstack(fill_value=0)
    )

data_quality_summary = (
    batch_metadata_df
    .set_index("batch")
    .join(issue_counts, how="left")
    .fillna(0)
    .reset_index()
)

issue_columns = [
    column
    for column in data_quality_summary.columns
    if column not in {
        "batch",
        "file_path",
        "records_loaded",
        "unique_nonempty_epic_ids",
        "missing_epic_ids",
        "duplicate_epic_id_count",
    }
]

if issue_columns:
    data_quality_summary["total_validation_issues"] = (
        data_quality_summary[issue_columns].sum(axis=1)
    )
else:
    data_quality_summary["total_validation_issues"] = 0


# ============================================================
# 5. BATCH-LEVEL COVERAGE SUMMARY
# ============================================================

batch_summary_rows = []

for batch_name, group in detail_df.groupby("batch", sort=False):
    coverage = group["overall_coverage"].dropna()

    total_stage_items = group["overall_total_items"].sum(min_count=1)
    covered_stage_items = group["overall_covered_items"].sum(min_count=1)

    batch_summary_rows.append(
        {
            "batch": batch_name,
            "epics": group["epic_id"].nunique(),
            "total_stage_items": total_stage_items,
            "covered_stage_items": covered_stage_items,
            "micro_overall_coverage": safe_divide(
                covered_stage_items,
                total_stage_items,
            ),
            "mean_epic_coverage": coverage.mean(),
            "median_epic_coverage": coverage.median(),
            "coverage_std": coverage.std(ddof=0),
            "minimum_epic_coverage": coverage.min(),
            "maximum_epic_coverage": coverage.max(),
            "zero_coverage_epics": int(
                np.isclose(coverage, 0.0).sum()
            ),
            "full_coverage_epics": int(
                np.isclose(coverage, 1.0).sum()
            ),
        }
    )

batch_summary_df = pd.DataFrame(batch_summary_rows)


# ============================================================
# 6. COVERAGE BY STAGE FIELD
# ============================================================

field_summary_rows = []

for batch_name, group in detail_df.groupby("batch", sort=False):
    for stage_field in STAGE_FIELDS:
        total_column = f"{stage_field}_total_items"
        covered_column = f"{stage_field}_covered_items"
        coverage_column = f"{stage_field}_coverage"

        total_items = group[total_column].sum(min_count=1)
        covered_items = group[covered_column].sum(min_count=1)
        epic_coverage = group[coverage_column].dropna()

        field_summary_rows.append(
            {
                "batch": batch_name,
                "stage_field": stage_field,
                "total_items": total_items,
                "covered_items": covered_items,
                "micro_coverage": safe_divide(
                    covered_items,
                    total_items,
                ),
                "mean_epic_coverage": epic_coverage.mean(),
                "median_epic_coverage": epic_coverage.median(),
                "coverage_std": epic_coverage.std(ddof=0),
                "epics_with_field_available": int(
                    (group[total_column] > 0).sum()
                ),
                "epics_with_field_used": int(
                    (group[covered_column] > 0).sum()
                ),
                "field_used_epic_rate": (
                    (group[covered_column] > 0).mean()
                ),
                "mean_total_items_per_epic": group[total_column].mean(),
            }
        )

field_summary_df = pd.DataFrame(field_summary_rows)


# ============================================================
# 7. WHERE STAGE INFORMATION APPEARS
# ============================================================

location_summary_rows = []

for batch_name, group in detail_df.groupby("batch", sort=False):
    for epic_field in EPIC_FIELDS:
        column = f"location_{epic_field}"
        valid_flags = group[column].dropna()

        location_summary_rows.append(
            {
                "batch": batch_name,
                "epic_field": epic_field,
                "valid_epics": len(valid_flags),
                "epics_with_stage_usage": int(
                    valid_flags.astype(bool).sum()
                ),
                "usage_rate": (
                    valid_flags.astype(bool).mean()
                    if len(valid_flags)
                    else np.nan
                ),
            }
        )

location_summary_df = pd.DataFrame(location_summary_rows)


# ============================================================
# 8. THREE-RUN CONSISTENCY BY EPIC
# ============================================================

# Duplicates are already reported. Keep the first occurrence for consistency.
clean_detail_df = (
    detail_df
    .sort_values(["batch", "epic_id"])
    .drop_duplicates(["batch", "epic_id"], keep="first")
)

consistency_rows = []

for epic_id, group in clean_detail_df.groupby("epic_id"):
    coverage_values = group["overall_coverage"].dropna()
    runs_present = group["batch"].nunique()

    consistency_row = {
        "epic_id": epic_id,
        "runs_present": runs_present,
        "mean_overall_coverage": coverage_values.mean(),
        "coverage_std": coverage_values.std(ddof=0),
        "minimum_coverage": coverage_values.min(),
        "maximum_coverage": coverage_values.max(),
        "coverage_range": (
            coverage_values.max() - coverage_values.min()
            if len(coverage_values)
            else np.nan
        ),
    }

    for batch_name in BATCH_FILES:
        batch_values = group.loc[
            group["batch"] == batch_name,
            "overall_coverage",
        ]

        consistency_row[f"{batch_name}_coverage"] = (
            batch_values.iloc[0]
            if not batch_values.empty
            else np.nan
        )

    for stage_field in STAGE_FIELDS:
        field_coverage = group[
            f"{stage_field}_coverage"
        ].dropna()

        total_item_counts = group[
            f"{stage_field}_total_items"
        ].dropna()

        consistency_row[f"{stage_field}_mean_coverage"] = (
            field_coverage.mean()
        )

        consistency_row[f"{stage_field}_coverage_range"] = (
            field_coverage.max() - field_coverage.min()
            if len(field_coverage)
            else np.nan
        )

        consistency_row[f"{stage_field}_total_items_range"] = (
            total_item_counts.max() - total_item_counts.min()
            if len(total_item_counts)
            else np.nan
        )

    for epic_field in EPIC_FIELDS:
        location_values = group[
            f"location_{epic_field}"
        ].dropna()

        boolean_values = [
            bool(value)
            for value in location_values
        ]

        consistency_row[f"{epic_field}_true_rate"] = (
            np.mean(boolean_values)
            if boolean_values
            else np.nan
        )

        consistency_row[f"{epic_field}_unanimous"] = (
            len(boolean_values) == len(BATCH_FILES)
            and len(set(boolean_values)) == 1
        )

    consistency_row["consistency_category"] = classify_consistency(
        consistency_row["coverage_range"],
        runs_present,
    )

    consistency_rows.append(consistency_row)

consistency_df = pd.DataFrame(consistency_rows)


# ============================================================
# 9. OVERALL CONSISTENCY SUMMARY
# ============================================================

all_three_runs = consistency_df[
    consistency_df["runs_present"] == len(BATCH_FILES)
].copy()

consistency_overview_df = pd.DataFrame(
    [
        {
            "unique_epics_seen": consistency_df["epic_id"].nunique(),
            "epics_present_in_all_three_runs": len(all_three_runs),
            "epics_missing_at_least_one_run": int(
                (
                    consistency_df["runs_present"]
                    < len(BATCH_FILES)
                ).sum()
            ),
            "stable_epics": int(
                (
                    all_three_runs["consistency_category"]
                    == "stable"
                ).sum()
            ),
            "moderate_variation_epics": int(
                (
                    all_three_runs["consistency_category"]
                    == "moderate_variation"
                ).sum()
            ),
            "unstable_epics": int(
                (
                    all_three_runs["consistency_category"]
                    == "unstable"
                ).sum()
            ),
            "stable_epic_rate": (
                (
                    all_three_runs["consistency_category"]
                    == "stable"
                ).mean()
                if len(all_three_runs)
                else np.nan
            ),
            "mean_coverage_range": (
                all_three_runs["coverage_range"].mean()
            ),
            "median_coverage_range": (
                all_three_runs["coverage_range"].median()
            ),
            "maximum_coverage_range": (
                all_three_runs["coverage_range"].max()
            ),
            "mean_run_standard_deviation": (
                all_three_runs["coverage_std"].mean()
            ),
        }
    ]
)


# ============================================================
# 10. FIELD-LEVEL RUN CONSISTENCY
# ============================================================

field_consistency_rows = []

for stage_field in STAGE_FIELDS:
    coverage_range_column = f"{stage_field}_coverage_range"
    item_range_column = f"{stage_field}_total_items_range"

    valid_coverage_ranges = all_three_runs[
        coverage_range_column
    ].dropna()

    valid_item_ranges = all_three_runs[
        item_range_column
    ].dropna()

    field_consistency_rows.append(
        {
            "stage_field": stage_field,
            "epics_evaluated": len(valid_coverage_ranges),
            "mean_coverage_range": valid_coverage_ranges.mean(),
            "median_coverage_range": valid_coverage_ranges.median(),
            "maximum_coverage_range": valid_coverage_ranges.max(),
            "exact_coverage_agreement_rate": (
                np.isclose(valid_coverage_ranges, 0.0).mean()
                if len(valid_coverage_ranges)
                else np.nan
            ),
            "mean_total_item_count_range": valid_item_ranges.mean(),
            "total_item_count_agreement_rate": (
                np.isclose(valid_item_ranges, 0.0).mean()
                if len(valid_item_ranges)
                else np.nan
            ),
        }
    )

field_consistency_df = pd.DataFrame(field_consistency_rows)


# ============================================================
# 11. LOCATION CONSISTENCY
# ============================================================

location_consistency_rows = []

for epic_field in EPIC_FIELDS:
    unanimous_column = f"{epic_field}_unanimous"
    true_rate_column = f"{epic_field}_true_rate"

    location_consistency_rows.append(
        {
            "epic_field": epic_field,
            "unanimous_location_rate": (
                all_three_runs[unanimous_column].mean()
                if len(all_three_runs)
                else np.nan
            ),
            "average_true_rate": (
                all_three_runs[true_rate_column].mean()
                if len(all_three_runs)
                else np.nan
            ),
            "epics_with_location_disagreement": int(
                (~all_three_runs[unanimous_column]).sum()
            ),
        }
    )

location_consistency_df = pd.DataFrame(
    location_consistency_rows
)


# ============================================================
# 12. OUTLIERS
# ============================================================

top_inconsistent_df = (
    consistency_df
    .sort_values(
        ["coverage_range", "coverage_std"],
        ascending=False,
    )
    .head(15)
)

lowest_coverage_df = (
    consistency_df
    .sort_values(
        ["mean_overall_coverage", "coverage_range"],
        ascending=[True, False],
    )
    .head(15)
)

highest_coverage_df = (
    consistency_df
    .sort_values(
        ["mean_overall_coverage", "coverage_range"],
        ascending=[False, False],
    )
    .head(15)
)

top_outlier_ids = set(top_inconsistent_df["epic_id"])

if not evidence_df.empty:
    outlier_evidence_df = evidence_df[
        evidence_df["epic_id"].isin(top_outlier_ids)
    ].sort_values(
        ["epic_id", "batch", "stage_field", "epic_field"]
    )
else:
    outlier_evidence_df = pd.DataFrame()


# ============================================================
# 13. SAVE ALL TABLES
# ============================================================

tables_to_save = {
    "data_quality_summary.csv": data_quality_summary,
    "validation_issues.csv": issues_df,
    "batch_summary.csv": batch_summary_df,
    "field_summary.csv": field_summary_df,
    "location_summary.csv": location_summary_df,
    "consistency_overview.csv": consistency_overview_df,
    "epic_run_consistency.csv": consistency_df,
    "field_consistency_summary.csv": field_consistency_df,
    "location_consistency_summary.csv": location_consistency_df,
    "top_15_inconsistent_epics.csv": top_inconsistent_df,
    "lowest_15_coverage_epics.csv": lowest_coverage_df,
    "highest_15_coverage_epics.csv": highest_coverage_df,
    "outlier_evidence.csv": outlier_evidence_df,
    "all_normalized_results.csv": detail_df,
}

for filename, dataframe in tables_to_save.items():
    dataframe.to_csv(OUTPUT_DIR / filename, index=False)


# ============================================================
# 14. GENERATE BASIC EDA CHARTS
# ============================================================

# Chart 1: Overall coverage distribution by batch
fig, ax = plt.subplots(figsize=(9, 6))

boxplot_values = []
boxplot_labels = []

for batch_name in BATCH_FILES:
    values = detail_df.loc[
        detail_df["batch"] == batch_name,
        "overall_coverage",
    ].dropna()

    boxplot_values.append(values)
    boxplot_labels.append(batch_name)

ax.boxplot(boxplot_values, tick_labels=boxplot_labels)
ax.set_title("Overall Stage Coverage by Batch")
ax.set_xlabel("Batch")
ax.set_ylabel("Overall Coverage")
ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "overall_coverage_by_batch.png",
    dpi=160,
)
plt.show()


# Chart 2: Micro coverage by stage field and batch
field_chart = field_summary_df.pivot(
    index="stage_field",
    columns="batch",
    values="micro_coverage",
)

fig, ax = plt.subplots(figsize=(10, 6))
field_chart.plot(kind="bar", ax=ax)
ax.set_title("Stage-Field Coverage by Batch")
ax.set_xlabel("Stage Field")
ax.set_ylabel("Micro Coverage")
ax.set_ylim(0, 1.05)
ax.tick_params(axis="x", rotation=25)
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "coverage_by_stage_field.png",
    dpi=160,
)
plt.show()


# Chart 3: Usage-location rate by batch
location_chart = location_summary_df.pivot(
    index="epic_field",
    columns="batch",
    values="usage_rate",
)

fig, ax = plt.subplots(figsize=(9, 6))
location_chart.plot(kind="bar", ax=ax)
ax.set_title("Where Stage Information Appears")
ax.set_xlabel("Epic Field")
ax.set_ylabel("Epic Usage Rate")
ax.set_ylim(0, 1.05)
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "stage_usage_locations.png",
    dpi=160,
)
plt.show()


# Chart 4: Distribution of run-to-run coverage ranges
if not all_three_runs.empty:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(
        all_three_runs["coverage_range"].dropna(),
        bins=10,
    )
    ax.set_title("Run-to-Run Overall Coverage Variation")
    ax.set_xlabel("Maximum Coverage − Minimum Coverage")
    ax.set_ylabel("Number of Epics")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "coverage_range_distribution.png",
        dpi=160,
    )
    plt.show()


# Chart 5: Most inconsistent epics
if not top_inconsistent_df.empty:
    chart_data = (
        top_inconsistent_df
        .sort_values("coverage_range")
        .set_index("epic_id")["coverage_range"]
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    chart_data.plot(kind="barh", ax=ax)
    ax.set_title("Most Inconsistent Epics Across Three Runs")
    ax.set_xlabel("Coverage Range")
    ax.set_ylabel("Epic ID")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "most_inconsistent_epics.png",
        dpi=160,
    )
    plt.show()


# ============================================================
# 15. PRINT THE IMPORTANT RESULTS
# ============================================================

print_table(
    "DATA QUALITY SUMMARY",
    data_quality_summary,
)

print_table(
    "BATCH COVERAGE SUMMARY",
    batch_summary_df,
)

print_table(
    "COVERAGE BY STAGE FIELD",
    field_summary_df,
)

print_table(
    "STAGE-USAGE LOCATION SUMMARY",
    location_summary_df,
)

print_table(
    "THREE-RUN CONSISTENCY OVERVIEW",
    consistency_overview_df,
)

print_table(
    "FIELD-LEVEL CONSISTENCY",
    field_consistency_df,
)

print_table(
    "LOCATION CONSISTENCY",
    location_consistency_df,
)

important_outlier_columns = [
    "epic_id",
    "runs_present",
    "batch_1_coverage",
    "batch_2_coverage",
    "batch_3_coverage",
    "mean_overall_coverage",
    "coverage_std",
    "coverage_range",
    "consistency_category",
]

available_outlier_columns = [
    column
    for column in important_outlier_columns
    if column in top_inconsistent_df.columns
]

print_table(
    "TOP 15 MOST INCONSISTENT EPICS",
    top_inconsistent_df[available_outlier_columns],
)

print(f"\nAll outputs were saved to: {OUTPUT_DIR.resolve()}")
