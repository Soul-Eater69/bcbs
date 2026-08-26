# Notebook Setup and Run Guide

## 1. Clone the repository

```bash
git clone https://github.com/hcsc-core/idp-workitem-gen-api.git
cd idp-workitem-gen-api
```

If the repository is already cloned:

```bash
cd idp-workitem-gen-api
git fetch origin
```

## 2. Switch to the evaluation branch

```bash
git checkout IDP-6157-LLM-AS-JUDGE-EVALS
git pull origin IDP-6157-LLM-AS-JUDGE-EVALS
```

## 3. Create the UV virtual environment

Run this from the project root, where `pyproject.toml` and `uv.lock` are located:

```bash
uv venv
```

## 4. Activate the environment

### Windows Command Prompt

```bat
.venv\Scripts\activate
```

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
source .venv/bin/activate
```

## 5. Install the project and dependencies

```bash
uv sync --extra dev
```

If Jupyter does not detect the environment as a notebook kernel:

```bash
uv pip install ipykernel
python -m ipykernel install --user --name idp-workitem-gen-api --display-name "idp-workitem-gen-api"
```

## 6. Place/open the notebooks

The notebook folder can be stored inside or outside the repository.

Example:

```text
workspace/
├── idp-workitem-gen-api/
│   ├── .venv/
│   ├── pyproject.toml
│   ├── uv.lock
│   └── idp_eval/
└── notebooks/
    ├── coverage_validation.ipynb
    ├── faithfulness_validation.ipynb
    ├── instruction_adherence_validation.ipynb
    ├── relevance_at_k_validation.ipynb
    └── ndcg_at_k_validation.ipynb
```

## 7. Select the notebook kernel

In VS Code/Jupyter, select:

```text
idp-workitem-gen-api
```

or select the Python interpreter from:

```text
idp-workitem-gen-api/.venv
```

## 8. Environment configuration

The required environment configuration will be provided separately.

Set the provided environment values before running the notebooks.

## 9. Run the notebooks

Open each notebook and run all cells from top to bottom.

Recommended order:

```text
coverage_validation.ipynb
faithfulness_validation.ipynb
instruction_adherence_validation.ipynb
relevance_at_k_validation.ipynb
ndcg_at_k_validation.ipynb
```

Each notebook contains its own test cases and writes the evaluation results to Excel.

## Quick Start

```bash
git clone https://github.com/hcsc-core/idp-workitem-gen-api.git
cd idp-workitem-gen-api
git checkout IDP-6157-LLM-AS-JUDGE-EVALS
git pull origin IDP-6157-LLM-AS-JUDGE-EVALS

uv venv
```

Activate `.venv`, then run:

```bash
uv sync --extra dev
```

Set the provided environment configuration, select the project `.venv` as the notebook kernel, and run all cells.
