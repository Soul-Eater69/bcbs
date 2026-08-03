# Stage Context Coverage in Human-Written Epics

## Executive Summary

This experiment evaluates how much information from a workflow stage is reflected in existing human-written epics.

Each evaluation compares four stage fields:

- Stage name
- Stage description
- Entrance criteria
- Exit criteria

against three epic fields:

- Epic title
- Epic description
- Epic success criteria

The analysis covers **three batches of 50 evaluations each**, for a total of **150 evaluation records**. Because some epic IDs appear in more than one batch, these results should be described as 150 evaluations rather than 150 unique epics.

Across the three batches, stage information was detectable in most epics, but the amount of reflected information was generally limited.

Key findings:

- Approximately **70%** of evaluations contained at least some stage information.
- **61.3%** of evaluations had no more than 25% stage coverage.
- Only **20%** of evaluations reflected more than half of the available stage information.
- The **stage name** was the most strongly reflected stage field.
- **Entrance criteria** were consistently the least reflected field.
- Stage information appeared most often in **epic success criteria**.
- Batch 1 had substantially lower coverage than batches 2 and 3.

---

## 1. Objective

The objective of this experiment is to answer:

> How much of the available stage context is semantically reflected in the human-written epic, and where does that information appear?

---

## 2. Evaluation Method

For each epic, the evaluator received:

### Stage context

- Stage name
- Stage description
- Entrance criteria
- Exit criteria

### Human-written epic

- Title
- Description
- Success criteria

The evaluator returned:

- Overall stage coverage
- Coverage by stage field
- Epic locations containing stage information
- Exact supporting evidence from the epic

### Coverage interpretation

A score of:

- `0.0` means no meaningful information from the field was reflected.
- `0.5` means approximately half of the meaningful information was reflected.
- `1.0` means all or nearly all meaningful information was reflected.

The overall score represents:

> Meaningful stage information reflected anywhere in the epic divided by total meaningful stage information available.

The title, description, and success criteria are evidence locations. They are not independently averaged to calculate overall coverage.

---

## 3. Overall Coverage Distribution

### Combined results across all batches

| Coverage bucket | Evaluation count | Percentage |
|---|---:|---:|
| No coverage | 45 | 30.0% |
| Low: greater than 0% to 25% | 47 | 31.3% |
| Moderate: greater than 25% to 50% | 28 | 18.7% |
| High: greater than 50% to 75% | 25 | 16.7% |
| Very high: greater than 75% to 100% | 5 | 3.3% |
| **Total** | **150** | **100%** |

### Interpretation

Stage information was found in **105 of 150 evaluations**, or approximately **70%**.

However, detected usage was usually limited:

- **92 evaluations, or 61.3%, had coverage of 25% or less.**
- Only **30 evaluations, or 20%, had coverage above 50%.**
- Only **5 evaluations, or 3.3%, had coverage above 75%.**

This indicates that stage context is usually used selectively rather than comprehensively.

---

## 4. Comparison Across Batches

| Metric | Batch 1 | Batch 2 | Batch 3 |
|---|---:|---:|---:|
| Evaluation count | 50 | 50 | 50 |
| Approximate mean coverage | 19.0% | 30.2% | 30.3% |
| No coverage | 38% | 26% | 26% |
| Low coverage | 36% | 32% | 26% |
| Moderate coverage | 14% | 16% | 26% |
| High coverage | 12% | 20% | 18% |
| Very high coverage | 0% | 6% | 4% |

### Interpretation

Batch 1 showed much lower stage coverage than batches 2 and 3.

Batches 2 and 3 were very similar in average coverage, both around 30%. This suggests that the lower result in batch 1 may be related to its epic or stage composition rather than the general behavior of the evaluator.

Possible explanations include:

- Different stages represented in each batch
- Different epic domains
- Different levels of detail in the stage context
- Different writing styles in the human-authored epics
- Evaluator variability

The next analysis should compare repeated epic IDs across batches and examine whether the same epic received stable scores.

---

## 5. Coverage by Stage Field

| Stage field | Batch 1 average coverage | Batch 2 average coverage | Batch 3 average coverage | Combined average coverage |
|---|---:|---:|---:|---:|
| Stage name | 35.5% | 45.0% | 48.5% | **43.0%** |
| Stage description | 21.2% | 32.4% | 33.0% | **28.8%** |
| Entrance criteria | 13.1% | 17.9% | 18.9% | **16.6%** |
| Exit criteria | 18.2% | 33.3% | 25.0% | **25.5%** |

### Ranking

1. Stage name: **43.0%**
2. Stage description: **28.8%**
3. Exit criteria: **25.5%**
4. Entrance criteria: **16.6%**

### Interpretation

#### Stage name

The stage name had the highest average coverage in every batch. The combined value of 43.0% means that, across all evaluations, an average of 43% of the meaningful information in the stage-name field was reflected somewhere in the epic. It does not mean that 43% of the epic came from the stage name, and it does not directly mean that 43% of epics used the stage name.

#### Stage description

The stage description had the second-highest coverage. It appears to provide broader context that can influence both the epic description and success criteria.

#### Exit criteria

Exit criteria were reflected more often than entrance criteria. This is expected because exit criteria describe what must be achieved before work is considered complete, which naturally overlaps with epic success criteria.

#### Entrance criteria

Entrance criteria had the lowest coverage in all three batches, and their median coverage was zero. This suggests that workflow prerequisites are usually not restated in the epic.

Low entrance-criteria coverage should not automatically be treated as a defect. Entrance criteria may describe process state rather than the work the epic is intended to deliver.

---

## 6. Where Stage Information Appears in the Epic

| Epic field | Batch 1 | Batch 2 | Batch 3 | Combined usage rate |
|---|---:|---:|---:|---:|
| Title | 30% | 32% | 34% | **32.0%** |
| Description | 38% | 42% | 48% | **42.7%** |
| Success criteria | 42% | 54% | 62% | **52.7%** |

Across 150 evaluations:

| Epic field | Evaluations containing stage information |
|---|---:|
| Title | 48 |
| Description | 64 |
| Success criteria | 79 |

### Interpretation

Stage information appeared most frequently in epic success criteria.

This suggests that stage context is more likely to shape:

- Completion expectations
- Validation conditions
- Required outcomes
- Acceptance conditions

than to shape the epic title.

The title had the lowest usage rate, which is reasonable because titles are short and often capture only the business topic rather than stage-specific detail.

The increase in success-criteria usage from 42% in batch 1 to 62% in batch 3 should be investigated further. It may reflect differences in batch composition or evaluator consistency.

---

## 7. Main Findings

### 7.1 Stage context is present but selectively used

Most epics contain some semantic overlap with their stage. However, the majority reflect only a small portion of the available stage context.

### 7.2 Strong stage coverage is uncommon

Only one in five evaluations reflected more than half of the available stage information.

### 7.3 Stage name provides the strongest signal

The stage name was the most consistently reflected field. It may be the most useful compact stage input for identifying the general purpose of an epic.

### 7.4 Entrance criteria provide the weakest signal

Entrance criteria were rarely reflected in epic text. They may be more useful for workflow control, readiness checks, or upstream validation than for epic generation.

### 7.5 Exit criteria align with success criteria

Exit criteria were more visible than entrance criteria and commonly appeared in epic success criteria. This indicates a meaningful relationship between stage completion rules and epic completion expectations.

### 7.6 Batch composition matters

The difference between batch 1 and batches 2 and 3 is large enough that a single combined average may hide meaningful variation.

---

## 8. Implications for Epic Generation

If stage context will be used as input for generating new epics, the current results suggest the following:

### Stage name

Use it to anchor the epic's primary purpose or lifecycle position.

### Stage description

Use it to shape the epic description and define the scope of work.

### Entrance criteria

Treat these mainly as prerequisites, assumptions, or readiness checks. They should not necessarily be copied into the epic.

### Exit criteria

Use these heavily when creating epic success criteria, because they define completion and expected outcomes.

A generation prompt should not force every stage field into every epic. Instead, it should use each field according to its natural function.

---

## 9. Limitations

### Semantic reflection is not causal attribution

The analysis shows that stage information is reflected in the epic. It does not prove that the human author used the stage as the source.

### LLM-generated scores require auditing

Coverage scores are based on semantic judgments made by an LLM. Exact evidence improves auditability, but the evaluator may still produce false positives or miss valid paraphrases.

### Overall decimal scores may be difficult to reproduce

When the evaluator directly returns a score such as `0.222`, the arithmetic is not fully auditable unless total and covered information counts are also returned.

A stronger design would have the evaluator return:

- Total meaningful items
- Covered meaningful items
- Exact evidence

and calculate coverage in code.

### Repeated epic IDs

Some epic IDs appear across multiple batches. The combined results therefore represent evaluation records, not necessarily unique epics.

### Batch composition

Differences in stages, domains, or text characteristics may explain variation between batches.

---

## 10. Recommended Next Analyses

### 10.1 Evaluator stability across batches

For epic IDs that appear in more than one batch:

- Compare overall coverage scores
- Compare field-level scores
- Compare usage locations
- Compare cited evidence

This will show whether the evaluator is consistent.

### 10.2 Stage-field-to-epic-field mapping

Create a matrix showing how often each stage field appears in each epic field:

| Stage source | Epic title | Epic description | Epic success criteria |
|---|---:|---:|---:|
| Stage name |  |  |  |
| Stage description |  |  |  |
| Entrance criteria |  |  |  |
| Exit criteria |  |  |  |

This will show the strongest semantic pathways.

### 10.3 Manual review

Review:

- Five highest-coverage evaluations
- Five lowest-coverage evaluations
- Five random evaluations
- Repeated epic IDs with inconsistent scores

For each, verify whether the evidence genuinely supports the assigned score.

### 10.4 Deterministic score calculation

Update the evaluator to return item counts and calculate coverage in code:

```text
field coverage = covered items / total items
overall coverage = total covered items / total available items
```

This will make the scoring reproducible.

### 10.5 Compare results by stage or domain

Group results by:

- Actual stage name
- Epic domain
- Epic type
- Business area

This may explain why some batches have higher coverage.

---

## 11. Conclusion

Across three batches containing 150 stage-to-epic evaluations, stage context was detectable in approximately 70% of cases. However, usage was generally limited: 61.3% of evaluations reflected no more than one-quarter of the available stage information, while only 20% reflected more than half.

The stage name was the most strongly reflected field, followed by the stage description and exit criteria. Entrance criteria had the weakest representation. Stage information appeared most frequently in epic success criteria, suggesting that stage context contributes more strongly to completion expectations than to titles.

Overall, the experiment indicates that human-written epics reflect stage context selectively rather than comprehensively. This supports using stage context as structured guidance during epic generation, while allowing different stage fields to influence different parts of the epic according to their natural purpose.
