import ast
import json
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    REPOSITORY_ROOT / "01_theme_stage.ipynb",
    REPOSITORY_ROOT / "02_full_context.ipynb",
    REPOSITORY_ROOT / "03_no_theme_description.ipynb",
    REPOSITORY_ROOT / "04_no_theme.ipynb",
    REPOSITORY_ROOT / "05_full_with_hierarchy.ipynb",
]

EXPECTED_PAYLOAD_PATHS = {
    "01_theme_stage.ipynb": {
        "task",
        "theme.business_needs",
        "theme.description",
        "value_stream_stage.stage_id",
        "value_stream_stage.stage_name",
        "value_stream_stage.stage_description",
        "value_stream_stage.entrance_criteria",
        "value_stream_stage.exit_criteria",
        "candidate_l3_capabilities",
        "selection_instruction",
    },
    "02_full_context.ipynb": {
        "task",
        "theme.business_needs",
        "theme.description",
        "epic.description",
        "epic.success_criteria",
        "value_stream_stage.stage_id",
        "value_stream_stage.stage_name",
        "value_stream_stage.stage_description",
        "value_stream_stage.entrance_criteria",
        "value_stream_stage.exit_criteria",
        "candidate_l3_capabilities",
        "selection_instruction",
    },
    "03_no_theme_description.ipynb": {
        "task",
        "theme.business_needs",
        "epic.description",
        "epic.success_criteria",
        "value_stream_stage.stage_id",
        "value_stream_stage.stage_name",
        "value_stream_stage.stage_description",
        "value_stream_stage.entrance_criteria",
        "value_stream_stage.exit_criteria",
        "candidate_l3_capabilities",
        "selection_instruction",
    },
    "04_no_theme.ipynb": {
        "task",
        "epic.description",
        "epic.success_criteria",
        "value_stream_stage.stage_id",
        "value_stream_stage.stage_name",
        "value_stream_stage.stage_description",
        "value_stream_stage.entrance_criteria",
        "value_stream_stage.exit_criteria",
        "candidate_l3_capabilities",
        "selection_instruction",
    },
    "05_full_with_hierarchy.ipynb": {
        "task",
        "theme.business_needs",
        "theme.description",
        "epic.description",
        "epic.success_criteria",
        "value_stream_stage.stage_id",
        "value_stream_stage.stage_name",
        "value_stream_stage.stage_description",
        "value_stream_stage.entrance_criteria",
        "value_stream_stage.exit_criteria",
        "candidate_l3_capabilities",
        "selection_instruction",
    },
}

BASE_CANDIDATE_FIELDS = [
    "capability_id",
    "capability_name",
    "capability_description",
    "capability_tier",
]


def load_notebook(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_text(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def code_source(notebook):
    return "\n\n".join(
        source_text(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )


def markdown_source(notebook):
    return "\n\n".join(
        source_text(cell)
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )


def assigned_string(source, variable_name):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == variable_name
                for target in node.targets
            ):
                return ast.literal_eval(node.value)
    raise AssertionError(f"{variable_name} was not assigned")


def extracted_prompt_builder(source):
    tree = ast.parse(source)
    builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_user_prompt"
    )
    module = ast.Module(
        body=[ast.Import(names=[ast.alias(name="json")]), builder],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), "<prompt_builder>", "exec"), namespace)
    return namespace["build_user_prompt"]


def leaf_paths(value, prefix=""):
    paths = set()
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(child, dict):
            paths.update(leaf_paths(child, path))
        else:
            paths.add(path)
    return paths


def sample_prompt_inputs():
    theme = {
        "theme_business_needs": "Need",
        "theme_description": "Theme description",
    }
    epic = {
        "key": "EPIC-1",
        "description": "Epic description",
        "success_criteria": "Success criteria",
    }
    stage = {
        "stage_id": "STAGE-1",
        "stage_name": "Stage",
        "stage_description": "Stage description",
        "entrance_criteria": "Entrance",
        "exit_criteria": "Exit",
    }
    candidates = [
        {
            "capability_id": "CAP1",
            "capability_name": "Capability",
            "capability_description": "Definition",
            "capability_tier": "Core",
        }
    ]
    return theme, epic, stage, candidates


def test_all_notebooks_are_valid_json_with_compilable_code_cells():
    for path in NOTEBOOKS:
        notebook = load_notebook(path)
        assert notebook["nbformat"] == 4
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile(source_text(cell), f"{path.name}:cell-{index}", "exec")


def test_system_prompt_is_byte_identical_and_contains_required_contract():
    prompts = [
        assigned_string(code_source(load_notebook(path)), "SYSTEM_PROMPT")
        for path in NOTEBOOKS
    ]
    assert len(set(prompts)) == 1
    prompt = prompts[0]
    required_phrases = [
        "Epic success criteria",
        "Epic description",
        "Value Stream Stage context",
        "Theme business needs",
        "Theme description",
        "capability_description",
        "capability_tier",
        "0 to 3",
        "direct evidence",
        "JSON only",
        "no more than three",
    ]
    for phrase in required_phrases:
        assert phrase in prompt


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_user_prompt_contains_exact_experiment_context(path):
    notebook = load_notebook(path)
    builder = extracted_prompt_builder(code_source(notebook))
    prompt = json.loads(builder(*sample_prompt_inputs()))
    assert leaf_paths(prompt) == EXPECTED_PAYLOAD_PATHS[path.name]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_candidate_schema_matches_experiment_definition(path):
    notebook = load_notebook(path)
    source = code_source(notebook)
    candidate_fields = assigned_string(source, "CANDIDATE_FIELDS")
    expected = BASE_CANDIDATE_FIELDS.copy()
    if path.name == "05_full_with_hierarchy.ipynb":
        expected.extend(["level_1_name", "level_2_name"])
    assert candidate_fields == expected
    assert "Capability Level" not in source


def test_notebooks_include_inspection_before_batch_execution():
    for path in NOTEBOOKS:
        notebook = load_notebook(path)
        headings = markdown_source(notebook)
        inspection_position = headings.index("## Single-example inspection")
        batch_position = headings.index("## Batch execution")
        assert inspection_position < batch_position


def test_ground_truth_is_not_used_by_prompt_builder_or_prediction_function():
    for path in NOTEBOOKS:
        tree = ast.parse(code_source(load_notebook(path)))
        protected_functions = {
            node.name: ast.unparse(node)
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"build_user_prompt", "predict_for_stage"}
        }
        assert set(protected_functions) == {"build_user_prompt", "predict_for_stage"}
        for function_source in protected_functions.values():
            lowered = function_source.lower()
            assert "ground_truth" not in lowered
            assert "gt_map" not in lowered


def test_candidate_retrieval_is_identical_except_hierarchy_projection():
    retrieval_bodies = []
    for path in NOTEBOOKS:
        tree = ast.parse(code_source(load_notebook(path)))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "candidate_rows_for_stage"
        )
        retrieval_bodies.append(ast.dump(function, include_attributes=False))

    assert len(set(retrieval_bodies[:4])) == 1
    assert retrieval_bodies[4] != retrieval_bodies[0]
