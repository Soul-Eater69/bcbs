# L3 Capability Experiments

Controlled ablation experiments for L3 business capability selection.

## Files

- `common.py` — shared gateway call, JSON parsing, candidate validation, metrics, and Excel export.
- `01_theme_stage.ipynb` — Theme business needs + Theme description + Stage + L3.
- `02_full_context.ipynb` — Theme + Epic description/success criteria + Stage + L3.
- `03_no_theme_description.ipynb` — Theme business needs + Epic + Stage + L3.
- `04_no_theme.ipynb` — Epic + Stage + L3 only.
- `05_full_with_hierarchy.ipynb` — full context + L1/L2 hierarchy.

The production `SYSTEM_PROMPT` is byte-identical in all five notebooks. Only the
intended user-context fields change.

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

Input paths are explicit in the first configuration cell of every notebook. By
default the notebooks look in `L3_EXPERIMENT_DATA_DIR` (or the current directory)
for:

- `epic_gen.csv`
- `value_stream_stages.xlsx`
- `stage_capability_map.xlsx`
- `capability_master.xlsx`
- `jira_ground_truth.csv`

Set the corresponding `L3_THEME_PATH`, `L3_STAGE_PATH`,
`L3_STAGE_CAPABILITY_MAP_PATH`, `L3_CAPABILITY_MASTER_PATH`, or
`L3_GROUND_TRUTH_PATH` environment variable when an export has another name or
location.

Jira and the internal LLM gateway continue to use the existing environment-based
configuration (`JIRA_BASE_URL`, Jira authentication variables, and
`IDP_GATEWAY_FACTORY`). No credentials are stored in the notebooks.

## Single-example inspection

Set `INSPECTION_THEME_ID` and `INSPECTION_EPIC_KEY` in a notebook's configuration
cell, then run through the inspection section. It displays the actual system
prompt, serialized user payload, ordered candidates, raw model response, and
validated prediction before batch execution. Ground truth is not loaded until
the separate evaluation section.

## Evaluation

Each experiment reports Epic-level exact-set match, precision, recall, and F1 against Jira ground truth, and writes an Excel result workbook under `results/`.

Evaluation also reports the union of candidate IDs available across each Epic's
retrieved stages, the number of ground-truth IDs present in that union, and the
resulting availability fraction. These diagnostics are calculated only after
prediction and are never supplied to the model.

## Fair comparison

Keep the model, temperature/settings, candidate ordering, system prompt, dataset, and ground truth fixed across all five experiments. Only the context fields supplied in the user prompt should vary.
