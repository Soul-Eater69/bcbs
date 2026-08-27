"""Small shared helpers for the L3 capability experiment notebooks.

Keep experiment-specific retrieval and prompt construction in the notebooks.
This module only contains identical infrastructure used by every experiment.
"""

from __future__ import annotations

import importlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


def load_gateway(factory_path: str | None = None) -> Any:
    """Load the existing project IDP gateway via ``module:function``.

    Example::
        IDP_GATEWAY_FACTORY=my_project.gateway:get_idp_gateway

    The factory must return an object exposing ``chat(system_prompt=..., user_prompt=...)``.
    """
    factory_path = factory_path or os.getenv("IDP_GATEWAY_FACTORY")
    if not factory_path:
        raise RuntimeError(
            "Set IDP_GATEWAY_FACTORY to 'module:function' or pass factory_path explicitly."
        )

    module_name, sep, factory_name = factory_path.partition(":")
    if not sep or not module_name or not factory_name:
        raise ValueError("IDP_GATEWAY_FACTORY must use the form 'module:function'.")

    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    gateway = factory()
    if not hasattr(gateway, "chat"):
        raise TypeError("Gateway factory must return an object with a chat(...) method.")
    return gateway


def call_llm(gateway: Any, system_prompt: str, user_prompt: str) -> str:
    """Call the existing IDP gateway using the interface shown in the current workflow."""
    response = gateway.chat(system_prompt=system_prompt, user_prompt=user_prompt)
    if not isinstance(response, str):
        raise TypeError(f"gateway.chat(...) must return str, got {type(response).__name__}")
    return response


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse JSON from a model response, tolerating a Markdown fence or surrounding text."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("LLM returned an empty response.")

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    candidate = fenced.group(1) if fenced else text.strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        parsed = None
        for match in re.finditer(r"\{", candidate):
            try:
                value, _ = decoder.raw_decode(candidate[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                parsed = value
                break
        if parsed is None:
            raise ValueError(f"LLM did not return a JSON object: {text}") from None

    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON response must be an object.")
    return parsed


def validate_l3_response(
    payload: Mapping[str, Any],
    candidate_ids: Iterable[str],
    *,
    allow_empty: bool = True,
    max_selected: int = 3,
) -> list[dict[str, str]]:
    """Validate the strict L3 output contract and return normalized selections."""
    raw = payload.get("l3")
    if not isinstance(raw, list):
        raise ValueError("LLM response must contain an 'l3' list.")
    if len(raw) > max_selected:
        raise ValueError(f"LLM may select at most {max_selected} L3 capabilities.")
    if not raw and not allow_empty:
        raise ValueError("LLM returned no L3 selections.")

    allowed = {str(value).strip() for value in candidate_ids if str(value).strip()}
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()

    for index, selection in enumerate(raw, start=1):
        if not isinstance(selection, Mapping):
            raise ValueError(f"L3 selection #{index} must be a JSON object.")

        capability_id = str(selection.get("capability_id", "")).strip()
        reason = str(selection.get("reason", "")).strip()

        if not capability_id:
            raise ValueError(f"L3 selection #{index} is missing capability_id.")
        if capability_id not in allowed:
            raise ValueError(
                f"LLM selected {capability_id}, which is not a supplied candidate."
            )
        if not reason:
            raise ValueError(f"LLM did not provide a reason for {capability_id}.")
        if capability_id in seen:
            raise ValueError(f"LLM returned duplicate capability_id {capability_id}.")

        seen.add(capability_id)
        normalized.append({"capability_id": capability_id, "reason": reason})

    return normalized


def score_sets(predicted: Iterable[str], truth: Iterable[str]) -> dict[str, float | int]:
    """Calculate exact-set match, precision, recall and F1 for one Epic."""
    predicted_set = {str(value).strip() for value in predicted if str(value).strip()}
    truth_set = {str(value).strip() for value in truth if str(value).strip()}

    if not predicted_set and not truth_set:
        precision = recall = f1 = 1.0
    else:
        tp = len(predicted_set & truth_set)
        precision = tp / len(predicted_set) if predicted_set else 0.0
        recall = tp / len(truth_set) if truth_set else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

    return {
        "exact_match": int(predicted_set == truth_set),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_count": len(predicted_set),
        "truth_count": len(truth_set),
    }


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Return a compact one-row summary over rows that have evaluation metrics."""
    if results.empty:
        return pd.DataFrame(
            [
                {
                    "evaluated_epics": 0,
                    "exact_match_accuracy": 0.0,
                    "mean_precision": 0.0,
                    "mean_recall": 0.0,
                    "mean_f1": 0.0,
                    "error_rows": 0,
                }
            ]
        )

    evaluated = results.loc[results["exact_match"].notna()].copy()
    return pd.DataFrame(
        [
            {
                "evaluated_epics": int(len(evaluated)),
                "exact_match_accuracy": float(evaluated["exact_match"].mean())
                if len(evaluated)
                else 0.0,
                "mean_precision": float(evaluated["precision"].mean())
                if len(evaluated)
                else 0.0,
                "mean_recall": float(evaluated["recall"].mean())
                if len(evaluated)
                else 0.0,
                "mean_f1": float(evaluated["f1"].mean()) if len(evaluated) else 0.0,
                "error_rows": int((results["status"] == "error").sum())
                if "status" in results
                else 0,
            }
        ]
    )


def save_results_excel(
    results: pd.DataFrame,
    experiment_name: str,
    output_dir: str | Path = "results",
) -> Path:
    """Save prediction rows and summary to a single Excel workbook."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{experiment_name}.xlsx"
    summary = summarize_results(results)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="predictions", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)

        for sheet_name in ("predictions", "summary"):
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                width = min(
                    max(len(str(cell.value or "")) for cell in column_cells) + 2,
                    80,
                )
                worksheet.column_dimensions[column_cells[0].column_letter].width = width

    return output_path
