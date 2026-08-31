from typing import Any

from worklet_data_api import Worklet


def extract_stage_matches(
    theme_worklet: Worklet,
    stage_catalogue: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Match Value Stage names found in a THEME worklet's Business Needs.

    Args:
        theme_worklet: THEME worklet containing the ``businessNeeds`` property.
        stage_catalogue: Governed Value Stage catalogue already scoped to the
            Theme's Value Stream.

    Returns:
        Matched Value Stage catalogue entries.
    """
    business_needs = theme_worklet.get_property_value("businessNeeds") or ""
    text_norm = business_needs.lower()

    matches: list[dict[str, Any]] = []

    ordered_stages = sorted(
        stage_catalogue,
        key=lambda stage: len(stage.get("stage_name", "")),
        reverse=True,
    )

    for stage in ordered_stages:
        stage_name = stage.get("stage_name") or ""
        name_norm = stage_name.lower()

        if not name_norm or name_norm not in text_norm:
            continue

        # A longer matching stage was already selected.
        if any(
            name_norm in match["title"].lower()
            for match in matches
        ):
            continue

        matches.append(
            {
                "valueStageId": stage["stage_id"],
                "title": stage_name,
                "description": stage.get("description") or "",
                "sourceFromBusinessNeeds": True,
            }
        )

    theme_worklet.upsert_property(
        name="matchedValueStages",
        value=matches,
    )
    theme_worklet.upsert_property(
        name="noMatch",
        value=not matches,
    )

    return matches
