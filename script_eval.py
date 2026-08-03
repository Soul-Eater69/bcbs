import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


# ============================================================
# UPDATE ONLY THESE PATHS
# ============================================================

FILES = {
    "run_1": "run_1.json",
    "run_2": "run_2.json",
    "run_3": "run_3.json",
}

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
# HELPERS
# ============================================================

def extract_records(data):
    """
    Supports:
    - A direct list of results
    - {"results": [...]}
    - {"data": [...]}
    - {"items": [...]}
    - A dictionary keyed by epic_id
    """

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in [
            "results",
            "data",
            "items",
            "records",
            "evaluations",
            "outputs",
        ]:
            if isinstance(data.get(key), list):
                return data[key]

        if "epic_id" in data and "stage_field_counts" in data:
            return [data]

        if data and all(isinstance(value, dict) for value in data.values()):
            records = []

            for epic_id, value in data.items():
                record = dict(value)
                record.setdefault("epic_id", str(epic_id))
                records.append(record)

            return records

    raise ValueError(
        "Could not recognize the JSON structure. "
        "Please print the top-level keys separately."
    )


def safe_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def safe_coverage(covered, total):
    if pd.isna(covered) or pd.isna(total) or total <= 0:
        return np.nan

    return covered / total


# ============================================================
# LOAD AND NORMALIZE
# ============================================================

rows = []
evidence_rows = []
schema_details = {}
validation_issues = []

for run_name, file_path in FILES.items():
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    records = extract_records(raw)

    schema_details[run_name] = {
        "top_level_type": type(raw).__name__,
        "top_level_keys": (
            list(raw.keys())[:20]
            if isinstance(raw, dict)
            else None
        ),
        "record_count": len(records),
        "first_record_keys": (
            list(records[0].keys())
            if records and isinstance(records[0], dict)
            else None
        ),
    }

    epic_ids = []

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            validation_issues.append(
                {
                    "run": run_name,
                    "epic_id": f"row_{index}",
                    "issue": "record_is_not_an_object",
                }
            )
            continue

        epic_id = str(record.get("epic_id", "")).strip()

        if not epic_id:
            epic_id = f"missing_id_row_{index}"
            validation_issues.append(
                {
                    "run": run_name,
                    "epic_id": epic_id,
                    "issue": "missing_epic_id",
                }
            )

        epic_ids.append(epic_id)

        counts = record.get("stage_field_counts", {})
        locations = record.get("stage_usage_locations", {})
        evidence = record.get("evidence", {})

        row = {
            "run": run_name,
            "epic_id": epic_id,
        }

        overall_total = 0
        overall_covered = 0
        has_valid_counts = False

        for stage_field in STAGE_FIELDS:
            field_counts = counts.get(stage_field, {})

            total_items = safe_number(
                field_counts.get("total_items")
            )
            covered_items = safe_number(
                field_counts.get("covered_items")
            )

            row[f"{stage_field}_total"] = total_items
            row[f"{stage_field}_covered"] = covered_items
            row[f"{stage_field}_coverage"] = safe_coverage(
                covered_items,
                total_items,
            )

            if not pd.isna(total_items) and not pd.isna(covered_items):
                has_valid_counts = True
                overall_total += total_items
                overall_covered += covered_items

                if covered_items > total_items:
                    validation_issues.append(
                        {
                            "run": run_name,
                            "epic_id": epic_id,
                            "issue": (
                                f"{stage_field}_covered_greater_than_total"
                            ),
                        }
                    )
            else:
                validation_issues.append(
                    {
                        "run": run_name,
                        "epic_id": epic_id,
                        "issue": f"{stage_field}_missing_or_invalid_counts",
                    }
                )

            field_evidence = evidence.get(stage_field, [])

            if not isinstance(field_evidence, list):
                validation_issues.append(
                    {
                        "run": run_name,
                        "epic_id": epic_id,
                        "issue": f"{stage_field}_evidence_not_list",
                    }
                )
                field_evidence = []

            row[f"{stage_field}_evidence_count"] = len(field_evidence)

            for item in field_evidence:
                if not isinstance(item, dict):
                    continue

                epic_field = item.get("epic_field")
                text = item.get("text", "")

                evidence_rows.append(
                    {
                        "run": run_name,
                        "epic_id": epic_id,
                        "stage_field": stage_field,
                        "epic_field": epic_field,
                        "text_length": (
                            len(text)
                            if isinstance(text, str)
                            else np.nan
                        ),
                    }
                )

            if not pd.isna(covered_items):
                if covered_items > 0 and len(field_evidence) == 0:
                    validation_issues.append(
                        {
                            "run": run_name,
                            "epic_id": epic_id,
                            "issue": f"{stage_field}_covered_without_evidence",
                        }
                    )

                if covered_items == 0 and len(field_evidence) > 0:
                    validation_issues.append(
                        {
                            "run": run_name,
                            "epic_id": epic_id,
                            "issue": f"{stage_field}_evidence_with_zero_covered",
                        }
                    )

        row["overall_total"] = (
            overall_total if has_valid_counts else np.nan
        )
        row["overall_covered"] = (
            overall_covered if has_valid_counts else np.nan
        )
        row["overall_coverage"] = safe_coverage(
            row["overall_covered"],
            row["overall_total"],
        )

        for epic_field in EPIC_FIELDS:
            row[f"used_in_{epic_field}"] = locations.get(epic_field)

        rows.append(row)

    duplicate_ids = [
        epic_id
        for epic_id, count in Counter(epic_ids).items()
        if count > 1
    ]

    schema_details[run_name]["duplicate_epic_ids"] = duplicate_ids


df = pd.DataFrame(rows)
evidence_df = pd.DataFrame(evidence_rows)
issues_df = pd.DataFrame(validation_issues)


# ============================================================
# 1. FILE AND SCHEMA INFORMATION
# ============================================================

print("\n" + "=" * 90)
print("A. FILE AND SCHEMA INFORMATION")
print("=" * 90)

print(json.dumps(schema_details, indent=2))


# ============================================================
# 2. DATA QUALITY INFORMATION
# ============================================================

print("\n" + "=" * 90)
print("B. DATA QUALITY SUMMARY")
print("=" * 90)

if issues_df.empty:
    print("No validation issues detected.")
else:
    issue_summary = (
        issues_df.groupby(["run", "issue"])
        .size()
        .reset_index(name="count")
    )

    print(issue_summary.to_string(index=False))


# ============================================================
# 3. EPIC ID OVERLAP ACROSS RUNS
# ============================================================

print("\n" + "=" * 90)
print("C. EPIC ID OVERLAP")
print("=" * 90)

id_sets = {
    run_name: set(
        df.loc[df["run"] == run_name, "epic_id"]
    )
    for run_name in FILES
}

common_ids = set.intersection(*id_sets.values())
all_ids = set.union(*id_sets.values())

overlap_summary = {
    "unique_epics_across_all_runs": len(all_ids),
    "epics_present_in_all_three_runs": len(common_ids),
    "epics_missing_from_at_least_one_run": len(all_ids - common_ids),
    "run_counts": {
        run_name: len(epic_ids)
        for run_name, epic_ids in id_sets.items()
    },
}

print(json.dumps(overlap_summary, indent=2))

for run_name, epic_ids in id_sets.items():
    missing = sorted(all_ids - epic_ids)

    print(
        f"\nMissing from {run_name}:",
        missing[:20],
        "..." if len(missing) > 20 else "",
    )


# ============================================================
# 4. OVERALL COVERAGE BY RUN
# ============================================================

print("\n" + "=" * 90)
print("D. OVERALL COVERAGE BY RUN")
print("=" * 90)

overall_rows = []

for run_name, group in df.groupby("run"):
    coverage = group["overall_coverage"].dropna()

    total_items = group["overall_total"].sum()
    covered_items = group["overall_covered"].sum()

    overall_rows.append(
        {
            "run": run_name,
            "epics": group["epic_id"].nunique(),
            "total_stage_items": total_items,
            "covered_stage_items": covered_items,
            "micro_coverage": (
                covered_items / total_items
                if total_items > 0
                else np.nan
            ),
            "mean_epic_coverage": coverage.mean(),
            "median_epic_coverage": coverage.median(),
            "std_epic_coverage": coverage.std(ddof=0),
            "minimum_coverage": coverage.min(),
            "maximum_coverage": coverage.max(),
            "zero_coverage_epics": int(
                np.isclose(coverage, 0).sum()
            ),
            "full_coverage_epics": int(
                np.isclose(coverage, 1).sum()
            ),
        }
    )

overall_summary = pd.DataFrame(overall_rows).round(3)
print(overall_summary.to_string(index=False))


# ============================================================
# 5. COVERAGE BY STAGE FIELD
# ============================================================

print("\n" + "=" * 90)
print("E. COVERAGE BY STAGE FIELD")
print("=" * 90)

field_rows = []

for run_name, group in df.groupby("run"):
    for stage_field in STAGE_FIELDS:
        total = group[f"{stage_field}_total"].sum()
        covered = group[f"{stage_field}_covered"].sum()
        epic_coverage = group[
            f"{stage_field}_coverage"
        ].dropna()

        field_rows.append(
            {
                "run": run_name,
                "stage_field": stage_field,
                "total_items": total,
                "covered_items": covered,
                "micro_coverage": (
                    covered / total
                    if total > 0
                    else np.nan
                ),
                "mean_epic_coverage": epic_coverage.mean(),
                "median_epic_coverage": epic_coverage.median(),
                "coverage_std": epic_coverage.std(ddof=0),
                "mean_total_items_per_epic": (
                    group[f"{stage_field}_total"].mean()
                ),
                "epics_where_field_was_used": int(
                    (
                        group[f"{stage_field}_covered"] > 0
                    ).sum()
                ),
            }
        )

field_summary = pd.DataFrame(field_rows).round(3)
print(field_summary.to_string(index=False))


# ============================================================
# 6. WHERE STAGE INFORMATION APPEARS
# ============================================================

print("\n" + "=" * 90)
print("F. STAGE USAGE LOCATIONS")
print("=" * 90)

location_rows = []

for run_name, group in df.groupby("run"):
    for epic_field in EPIC_FIELDS:
        values = group[
            f"used_in_{epic_field}"
        ].dropna()

        true_count = sum(value is True for value in values)

        location_rows.append(
            {
                "run": run_name,
                "epic_field": epic_field,
                "valid_records": len(values),
                "true_count": true_count,
                "usage_rate": (
                    true_count / len(values)
                    if len(values) > 0
                    else np.nan
                ),
            }
        )

location_summary = pd.DataFrame(location_rows).round(3)
print(location_summary.to_string(index=False))


# ============================================================
# 7. EVIDENCE DISTRIBUTION
# ============================================================

print("\n" + "=" * 90)
print("G. EVIDENCE DISTRIBUTION")
print("=" * 90)

if evidence_df.empty:
    print("No evidence records found.")
else:
    evidence_summary = (
        evidence_df.groupby(
            ["run", "stage_field", "epic_field"],
            dropna=False,
        )
        .agg(
            evidence_count=("epic_id", "size"),
            unique_epics=("epic_id", "nunique"),
            mean_text_length=("text_length", "mean"),
        )
        .reset_index()
        .round(2)
    )

    print(evidence_summary.to_string(index=False))


# ============================================================
# 8. RUN-TO-RUN CONSISTENCY
# ============================================================

print("\n" + "=" * 90)
print("H. RUN-TO-RUN CONSISTENCY")
print("=" * 90)

common_df = df[df["epic_id"].isin(common_ids)].copy()

coverage_pivot = common_df.pivot_table(
    index="epic_id",
    columns="run",
    values="overall_coverage",
    aggfunc="first",
)

coverage_pivot["mean_coverage"] = coverage_pivot.mean(axis=1)
coverage_pivot["coverage_std"] = coverage_pivot[
    list(FILES.keys())
].std(axis=1, ddof=0)
coverage_pivot["coverage_range"] = (
    coverage_pivot[list(FILES.keys())].max(axis=1)
    - coverage_pivot[list(FILES.keys())].min(axis=1)
)

coverage_pivot["consistency"] = pd.cut(
    coverage_pivot["coverage_range"],
    bins=[-0.001, 0.10, 0.25, np.inf],
    labels=[
        "stable",
        "moderate_variation",
        "unstable",
    ],
)

consistency_counts = (
    coverage_pivot["consistency"]
    .value_counts(dropna=False)
    .to_dict()
)

consistency_summary = {
    "epics_compared": len(coverage_pivot),
    "mean_coverage_range": round(
        coverage_pivot["coverage_range"].mean(),
        3,
    ),
    "median_coverage_range": round(
        coverage_pivot["coverage_range"].median(),
        3,
    ),
    "maximum_coverage_range": round(
        coverage_pivot["coverage_range"].max(),
        3,
    ),
    "mean_run_standard_deviation": round(
        coverage_pivot["coverage_std"].mean(),
        3,
    ),
    "consistency_counts": {
        str(key): int(value)
        for key, value in consistency_counts.items()
    },
}

print(json.dumps(consistency_summary, indent=2))


# ============================================================
# 9. FIELD-LEVEL CONSISTENCY
# ============================================================

print("\n" + "=" * 90)
print("I. FIELD-LEVEL CONSISTENCY")
print("=" * 90)

field_consistency_rows = []

for stage_field in STAGE_FIELDS:
    field_coverage_pivot = common_df.pivot_table(
        index="epic_id",
        columns="run",
        values=f"{stage_field}_coverage",
        aggfunc="first",
    )

    total_items_pivot = common_df.pivot_table(
        index="epic_id",
        columns="run",
        values=f"{stage_field}_total",
        aggfunc="first",
    )

    field_ranges = (
        field_coverage_pivot.max(axis=1)
        - field_coverage_pivot.min(axis=1)
    )

    item_count_ranges = (
        total_items_pivot.max(axis=1)
        - total_items_pivot.min(axis=1)
    )

    field_consistency_rows.append(
        {
            "stage_field": stage_field,
            "mean_coverage_range": field_ranges.mean(),
            "median_coverage_range": field_ranges.median(),
            "maximum_coverage_range": field_ranges.max(),
            "exact_coverage_agreement_rate": (
                np.isclose(field_ranges, 0).mean()
            ),
            "mean_total_item_count_range": (
                item_count_ranges.mean()
            ),
            "exact_total_item_agreement_rate": (
                np.isclose(item_count_ranges, 0).mean()
            ),
        }
    )

field_consistency_summary = pd.DataFrame(
    field_consistency_rows
).round(3)

print(field_consistency_summary.to_string(index=False))


# ============================================================
# 10. MOST INCONSISTENT EPICS
# ============================================================

print("\n" + "=" * 90)
print("J. TOP 15 MOST INCONSISTENT EPICS")
print("=" * 90)

top_inconsistent = (
    coverage_pivot
    .sort_values(
        ["coverage_range", "coverage_std"],
        ascending=False,
    )
    .head(15)
    .reset_index()
    .round(3)
)

print(top_inconsistent.to_string(index=False))


# ============================================================
# 11. LOWEST AND HIGHEST COVERAGE EPICS
# ============================================================

print("\n" + "=" * 90)
print("K. LOWEST 10 MEAN-COVERAGE EPICS")
print("=" * 90)

lowest = (
    coverage_pivot
    .sort_values("mean_coverage")
    .head(10)
    .reset_index()
    .round(3)
)

print(lowest.to_string(index=False))


print("\n" + "=" * 90)
print("L. HIGHEST 10 MEAN-COVERAGE EPICS")
print("=" * 90)

highest = (
    coverage_pivot
    .sort_values("mean_coverage", ascending=False)
    .head(10)
    .reset_index()
    .round(3)
)

print(highest.to_string(index=False))
