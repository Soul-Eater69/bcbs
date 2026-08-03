import json
from collections import Counter

import pandas as pd


# ---------------------------------------------------------
# CHANGE THIS ONLY IF YOUR LIST HAS A DIFFERENT VARIABLE NAME
# ---------------------------------------------------------
records = results


# Convert Pydantic objects to dictionaries when necessary
def to_dict(item):
    if hasattr(item, "model_dump"):
        return item.model_dump()

    if isinstance(item, dict):
        return item

    raise TypeError(f"Unsupported record type: {type(item)}")


records = [to_dict(record) for record in records]

print(f"Total epics analyzed: {len(records)}")


# =========================================================
# SECTION 1: OVERALL STAGE COVERAGE
# =========================================================

overall_rows = []

for record in records:
    overall_rows.append(
        {
            "epic_id": record.get("epic_id"),
            "overall_stage_coverage": record.get(
                "overall_stage_coverage", 0
            ),
        }
    )

overall_df = pd.DataFrame(overall_rows)

overall_df["overall_stage_coverage"] = pd.to_numeric(
    overall_df["overall_stage_coverage"],
    errors="coerce",
)


def coverage_bucket(score):
    if pd.isna(score):
        return "Missing"
    if score == 0:
        return "No coverage"
    if score <= 0.25:
        return "Low: 0-25%"
    if score <= 0.50:
        return "Moderate: 25-50%"
    if score <= 0.75:
        return "High: 50-75%"
    return "Very high: 75-100%"


overall_df["coverage_bucket"] = overall_df[
    "overall_stage_coverage"
].apply(coverage_bucket)

print("\n" + "=" * 70)
print("SECTION 1: OVERALL STAGE COVERAGE")
print("=" * 70)

print(
    pd.DataFrame(
        {
            "metric": [
                "Total epics",
                "Mean coverage",
                "Median coverage",
                "Minimum coverage",
                "Maximum coverage",
                "Epics with zero coverage",
                "Epics above 25%",
                "Epics above 50%",
                "Epics above 75%",
            ],
            "value": [
                len(overall_df),
                round(
                    overall_df["overall_stage_coverage"].mean(),
                    3,
                ),
                round(
                    overall_df["overall_stage_coverage"].median(),
                    3,
                ),
                round(
                    overall_df["overall_stage_coverage"].min(),
                    3,
                ),
                round(
                    overall_df["overall_stage_coverage"].max(),
                    3,
                ),
                int(
                    (
                        overall_df["overall_stage_coverage"] == 0
                    ).sum()
                ),
                int(
                    (
                        overall_df["overall_stage_coverage"] > 0.25
                    ).sum()
                ),
                int(
                    (
                        overall_df["overall_stage_coverage"] > 0.50
                    ).sum()
                ),
                int(
                    (
                        overall_df["overall_stage_coverage"] > 0.75
                    ).sum()
                ),
            ],
        }
    ).to_string(index=False)
)

print("\nCoverage buckets:")

bucket_summary = (
    overall_df["coverage_bucket"]
    .value_counts()
    .rename_axis("coverage_bucket")
    .reset_index(name="epic_count")
)

bucket_summary["percentage"] = (
    bucket_summary["epic_count"] / len(overall_df) * 100
).round(2)

print(bucket_summary.to_string(index=False))


# =========================================================
# SECTION 2: WHICH STAGE FIELDS ARE USED MOST?
# =========================================================

stage_fields = [
    "stage_name",
    "stage_description",
    "entrance_criteria",
    "exit_criteria",
]

field_rows = []

for record in records:
    field_coverage = record.get(
        "coverage_by_stage_field", {}
    )

    for field in stage_fields:
        field_rows.append(
            {
                "epic_id": record.get("epic_id"),
                "stage_field": field,
                "coverage": field_coverage.get(field, 0),
            }
        )

field_df = pd.DataFrame(field_rows)

field_df["coverage"] = pd.to_numeric(
    field_df["coverage"],
    errors="coerce",
)

field_summary = (
    field_df.groupby("stage_field")
    .agg(
        epic_count=("epic_id", "count"),
        mean_coverage=("coverage", "mean"),
        median_coverage=("coverage", "median"),
        zero_coverage_count=(
            "coverage",
            lambda values: int((values == 0).sum()),
        ),
        positive_coverage_count=(
            "coverage",
            lambda values: int((values > 0).sum()),
        ),
        above_50_percent_count=(
            "coverage",
            lambda values: int((values > 0.5).sum()),
        ),
    )
    .reset_index()
)

field_summary["zero_coverage_percentage"] = (
    field_summary["zero_coverage_count"]
    / field_summary["epic_count"]
    * 100
).round(2)

field_summary["positive_coverage_percentage"] = (
    field_summary["positive_coverage_count"]
    / field_summary["epic_count"]
    * 100
).round(2)

field_summary["mean_coverage"] = field_summary[
    "mean_coverage"
].round(3)

field_summary["median_coverage"] = field_summary[
    "median_coverage"
].round(3)

field_summary = field_summary.sort_values(
    "mean_coverage",
    ascending=False,
)

print("\n" + "=" * 70)
print("SECTION 2: COVERAGE BY STAGE FIELD")
print("=" * 70)

print(field_summary.to_string(index=False))


# =========================================================
# SECTION 3: WHERE IS STAGE INFORMATION USED?
# =========================================================

epic_fields = [
    "title",
    "description",
    "success_criteria",
]

location_rows = []

for record in records:
    locations = record.get("stage_usage_locations", {})

    location_rows.append(
        {
            "epic_id": record.get("epic_id"),
            "title": bool(locations.get("title", False)),
            "description": bool(
                locations.get("description", False)
            ),
            "success_criteria": bool(
                locations.get("success_criteria", False)
            ),
        }
    )

location_df = pd.DataFrame(location_rows)

location_summary_rows = []

for field in epic_fields:
    used_count = int(location_df[field].sum())

    location_summary_rows.append(
        {
            "epic_field": field,
            "epics_using_stage_context": used_count,
            "percentage_of_epics": round(
                used_count / len(location_df) * 100,
                2,
            ),
        }
    )

location_summary = pd.DataFrame(location_summary_rows)

print("\n" + "=" * 70)
print("SECTION 3: STAGE USAGE BY EPIC FIELD")
print("=" * 70)

print(location_summary.to_string(index=False))


def get_location_pattern(row):
    used_locations = [
        field
        for field in epic_fields
        if row[field]
    ]

    if not used_locations:
        return "No stage usage"

    return " + ".join(used_locations)


location_df["location_pattern"] = location_df.apply(
    get_location_pattern,
    axis=1,
)

pattern_summary = (
    location_df["location_pattern"]
    .value_counts()
    .rename_axis("usage_pattern")
    .reset_index(name="epic_count")
)

pattern_summary["percentage"] = (
    pattern_summary["epic_count"]
    / len(location_df)
    * 100
).round(2)

print("\nUsage-location combinations:")
print(pattern_summary.to_string(index=False))


# =========================================================
# SECTION 4: STAGE FIELD -> EPIC FIELD MAPPING
# =========================================================

mapping_records = []

for record in records:
    epic_id = record.get("epic_id")
    evidence = record.get("evidence", {})

    for stage_field in stage_fields:
        evidence_items = evidence.get(stage_field, []) or []

        for item in evidence_items:
            epic_field = item.get("epic_field")

            if epic_field not in epic_fields:
                continue

            mapping_records.append(
                {
                    "epic_id": epic_id,
                    "stage_field": stage_field,
                    "epic_field": epic_field,
                    "evidence_text": item.get("text", ""),
                }
            )

mapping_df = pd.DataFrame(mapping_records)

print("\n" + "=" * 70)
print("SECTION 4: STAGE FIELD TO EPIC FIELD MAPPING")
print("=" * 70)

if mapping_df.empty:
    print("No evidence mappings were found.")
else:
    unique_mapping_df = mapping_df.drop_duplicates(
        subset=[
            "epic_id",
            "stage_field",
            "epic_field",
        ]
    )

    mapping_table = pd.crosstab(
        unique_mapping_df["stage_field"],
        unique_mapping_df["epic_field"],
    )

    mapping_table = mapping_table.reindex(
        index=stage_fields,
        columns=epic_fields,
        fill_value=0,
    )

    print(mapping_table.to_string())

    print(
        "\nEach number represents the number of epics where "
        "that stage field was found in that epic field."
    )


# =========================================================
# SECTION 5: WHICH STAGE FIELDS ARE UNUSED?
# =========================================================

unused_rows = []

for field in stage_fields:
    field_records = field_df[
        field_df["stage_field"] == field
    ]

    unused_count = int(
        (field_records["coverage"] == 0).sum()
    )

    unused_rows.append(
        {
            "stage_field": field,
            "unused_epic_count": unused_count,
            "unused_percentage": round(
                unused_count / len(records) * 100,
                2,
            ),
        }
    )

unused_summary = pd.DataFrame(unused_rows).sort_values(
    "unused_percentage",
    ascending=False,
)

print("\n" + "=" * 70)
print("SECTION 5: UNUSED STAGE FIELDS")
print("=" * 70)

print(unused_summary.to_string(index=False))

zero_coverage_epics = overall_df[
    overall_df["overall_stage_coverage"] == 0
][["epic_id", "overall_stage_coverage"]]

print("\nEpics with zero overall stage coverage:")

if zero_coverage_epics.empty:
    print("None")
else:
    print(zero_coverage_epics.to_string(index=False))


# =========================================================
# SECTION 6: COVERAGE BY ACTUAL STAGE NAME
# Runs only if stage_name exists in each record
# =========================================================

stage_name_rows = []

for record in records:
    stage_name = (
        record.get("stage_name")
        or record.get("stage", {}).get("name")
    )

    if stage_name:
        stage_name_rows.append(
            {
                "epic_id": record.get("epic_id"),
                "stage_name": stage_name,
                "overall_stage_coverage": record.get(
                    "overall_stage_coverage", 0
                ),
            }
        )

print("\n" + "=" * 70)
print("SECTION 6: COVERAGE BY ACTUAL STAGE")
print("=" * 70)

if not stage_name_rows:
    print(
        "Skipped because the actual stage name is not present "
        "in the evaluator-output records."
    )
else:
    stage_name_df = pd.DataFrame(stage_name_rows)

    coverage_by_stage = (
        stage_name_df.groupby("stage_name")
        .agg(
            epic_count=("epic_id", "count"),
            mean_coverage=(
                "overall_stage_coverage",
                "mean",
            ),
            median_coverage=(
                "overall_stage_coverage",
                "median",
            ),
            minimum_coverage=(
                "overall_stage_coverage",
                "min",
            ),
            maximum_coverage=(
                "overall_stage_coverage",
                "max",
            ),
        )
        .reset_index()
        .sort_values(
            "mean_coverage",
            ascending=False,
        )
        .round(3)
    )

    print(coverage_by_stage.to_string(index=False))


# =========================================================
# SECTION 7: SAME EVIDENCE USED FOR MULTIPLE STAGE FIELDS
# =========================================================

print("\n" + "=" * 70)
print("SECTION 7: REUSED EVIDENCE")
print("=" * 70)

if mapping_df.empty:
    print("No evidence was available.")
else:
    mapping_df["normalized_evidence"] = (
        mapping_df["evidence_text"]
        .fillna("")
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    reused_evidence = (
        mapping_df[
            mapping_df["normalized_evidence"] != ""
        ]
        .groupby(
            [
                "epic_id",
                "epic_field",
                "normalized_evidence",
            ]
        )
        .agg(
            stage_field_count=(
                "stage_field",
                "nunique",
            ),
            stage_fields=(
                "stage_field",
                lambda values: ", ".join(
                    sorted(set(values))
                ),
            ),
        )
        .reset_index()
    )

    reused_evidence = reused_evidence[
        reused_evidence["stage_field_count"] > 1
    ]

    if reused_evidence.empty:
        print(
            "No identical epic evidence was assigned to "
            "multiple stage fields."
        )
    else:
        print(
            reused_evidence[
                [
                    "epic_id",
                    "epic_field",
                    "stage_fields",
                    "normalized_evidence",
                ]
            ].to_string(index=False)
        )

        print(
            f"\nTotal repeated-evidence cases: "
            f"{len(reused_evidence)}"
        )


# =========================================================
# SECTION 8: EVALUATOR CONSISTENCY CHECKS
# =========================================================

validation_issues = []

for record in records:
    epic_id = record.get("epic_id")
    overall_score = record.get(
        "overall_stage_coverage"
    )

    field_coverage = record.get(
        "coverage_by_stage_field", {}
    )

    evidence = record.get("evidence", {})
    locations = record.get(
        "stage_usage_locations", {}
    )

    if not isinstance(overall_score, (int, float)):
        validation_issues.append(
            {
                "epic_id": epic_id,
                "issue": "Overall coverage is not numeric",
            }
        )
    elif not 0 <= overall_score <= 1:
        validation_issues.append(
            {
                "epic_id": epic_id,
                "issue": "Overall coverage outside 0-1",
            }
        )

    evidence_locations = set()

    for stage_field in stage_fields:
        score = field_coverage.get(stage_field, 0)
        items = evidence.get(stage_field, []) or []

        if not isinstance(score, (int, float)):
            validation_issues.append(
                {
                    "epic_id": epic_id,
                    "issue": (
                        f"{stage_field} coverage is not numeric"
                    ),
                }
            )
            continue

        if not 0 <= score <= 1:
            validation_issues.append(
                {
                    "epic_id": epic_id,
                    "issue": (
                        f"{stage_field} coverage outside 0-1"
                    ),
                }
            )

        if score > 0 and len(items) == 0:
            validation_issues.append(
                {
                    "epic_id": epic_id,
                    "issue": (
                        f"{stage_field} has positive coverage "
                        "but no evidence"
                    ),
                }
            )

        if score == 0 and len(items) > 0:
            validation_issues.append(
                {
                    "epic_id": epic_id,
                    "issue": (
                        f"{stage_field} has zero coverage "
                        "but contains evidence"
                    ),
                }
            )

        for item in items:
            location = item.get("epic_field")

            if location:
                evidence_locations.add(location)

    for epic_field in epic_fields:
        expected = epic_field in evidence_locations
        supplied = bool(
            locations.get(epic_field, False)
        )

        if expected != supplied:
            validation_issues.append(
                {
                    "epic_id": epic_id,
                    "issue": (
                        f"Location mismatch for {epic_field}: "
                        f"expected={expected}, supplied={supplied}"
                    ),
                }
            )

issues_df = pd.DataFrame(validation_issues)

print("\n" + "=" * 70)
print("SECTION 8: EVALUATOR VALIDATION")
print("=" * 70)

print(f"Total records: {len(records)}")

if issues_df.empty:
    print("No structural inconsistencies found.")
else:
    print(
        f"Epics with issues: "
        f"{issues_df['epic_id'].nunique()}"
    )
    print(f"Total issues: {len(issues_df)}")
    print("\nIssues:")
    print(issues_df.to_string(index=False))


# =========================================================
# SECTION 9: TOP AND BOTTOM EPICS
# =========================================================

print("\n" + "=" * 70)
print("SECTION 9: TOP AND BOTTOM COVERAGE EPICS")
print("=" * 70)

print("\nHighest-coverage epics:")
print(
    overall_df.nlargest(
        5,
        "overall_stage_coverage",
    )[
        [
            "epic_id",
            "overall_stage_coverage",
            "coverage_bucket",
        ]
    ].to_string(index=False)
)

print("\nLowest-coverage epics:")
print(
    overall_df.nsmallest(
        5,
        "overall_stage_coverage",
    )[
        [
            "epic_id",
            "overall_stage_coverage",
            "coverage_bucket",
        ]
    ].to_string(index=False)
)


# =========================================================
# SAVE ALL MAIN TABLES INTO ONE EXCEL FILE
# =========================================================

output_file = "stage_coverage_eda.xlsx"

with pd.ExcelWriter(output_file) as writer:
    overall_df.to_excel(
        writer,
        sheet_name="overall_epics",
        index=False,
    )

    bucket_summary.to_excel(
        writer,
        sheet_name="coverage_buckets",
        index=False,
    )

    field_summary.to_excel(
        writer,
        sheet_name="stage_fields",
        index=False,
    )

    location_summary.to_excel(
        writer,
        sheet_name="usage_locations",
        index=False,
    )

    pattern_summary.to_excel(
        writer,
        sheet_name="location_patterns",
        index=False,
    )

    unused_summary.to_excel(
        writer,
        sheet_name="unused_fields",
        index=False,
    )

    if not mapping_df.empty:
        mapping_df.to_excel(
            writer,
            sheet_name="evidence_mapping",
            index=False,
        )

    if not issues_df.empty:
        issues_df.to_excel(
            writer,
            sheet_name="validation_issues",
            index=False,
        )

print(f"\nSaved complete analysis to: {output_file}")
