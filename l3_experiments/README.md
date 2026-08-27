# L3 Capability Experiments

Controlled ablation experiments for L3 business capability selection.

## Files

- `common.py` — shared gateway call, JSON parsing, candidate validation, metrics, and Excel export.
- `01_theme_stage.ipynb` — Theme business needs + Theme description + Stage + L3.
- `02_full_context.ipynb` — Theme + Epic description/success criteria + Stage + L3.
- `03_no_theme_description.ipynb` — Theme business needs + Epic + Stage + L3.
- `04_no_theme.ipynb` — Epic + Stage + L3 only.
- `05_full_with_hierarchy.ipynb` — full context + L1/L2 hierarchy.

The same production system prompt is used in all five notebooks. Only the user-context payload changes.

## Candidate L3 fields

Experiments 1–4 send:

- `capability_id`
- `capability_name`
- `capability_description`
- `capability_tier`

Experiment 5 adds:

- `level_1_name`
- `level_2_name`

## Inputs

The notebooks expect your existing exports/data sources for Theme, Epic, Value Stream Stage, Stage→Capability mapping, capability master, and Jira L3 ground truth. Update the file-path constants in the first configuration cell if your filenames differ.

## Evaluation

Each experiment reports Epic-level exact-set match, precision, recall, and F1 against Jira ground truth, and writes an Excel result workbook under `results/`.

## Fair comparison

Keep the model, temperature/settings, candidate ordering, system prompt, dataset, and ground truth fixed across all five experiments. Only the context fields supplied in the user prompt should vary.
