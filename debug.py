from typing import Any

from jwg_app.domain.models.worklet import Worklet


class ValueStageHandler:
    """Identify Value Stages referenced in a Theme's Business Needs."""

    def __init__(self, stage_catalogue: list[dict[str, Any]]):
        self.stage_catalogue = stage_catalogue

    def get_vss_catalogue(
        self,
        vs_id: str,
    ) -> list[dict[str, Any]]:
        """Return the Value Stage catalogue scoped to a Value Stream.

        Args:
            vs_id: Value Stream identifier.

        Returns:
            Value Stages associated with the Value Stream.
        """
        return [
            stage
            for stage in self.stage_catalogue
            if stage.get("valueStreamId") == vs_id
        ]

    def extract_vss(
        self,
        theme_worklet: Worklet,
    ) -> list[dict[str, Any]]:
        """Extract Value Stage matches from Theme Business Needs.

        Args:
            theme_worklet: Theme worklet containing Business Needs and
                Value Stream ID.

        Returns:
            Matched Value Stage catalogue entries.
        """
        vs_id = theme_worklet.get_property_value("valueStreamId")
        business_needs = (
            theme_worklet.get_property_value("businessNeeds") or ""
        )

        if not business_needs.strip():
            return []

        vss_catalogue = self.get_vss_catalogue(vs_id)
        if not vss_catalogue:
            return []

        text_norm = business_needs.lower()

        # Check longer stage names first so that a shorter stage name
        # contained inside a longer matched stage is not selected again.
        ordered_stages = sorted(
            vss_catalogue,
            key=lambda stage: len(stage.get("stage_name", "")),
            reverse=True,
        )

        matches: list[dict[str, Any]] = []

        for stage in ordered_stages:
            stage_name = stage.get("stage_name") or ""
            name_norm = stage_name.lower()

            if not name_norm or name_norm not in text_norm:
                continue

            # Skip a shorter stage when it is already represented by a
            # longer stage that matched first.
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

        return matches

    def run(
        self,
        theme_worklet: Worklet,
    ) -> dict[str, Any]:
        """Run Value Stage identification for a Theme.

        Args:
            theme_worklet: Theme worklet to evaluate.

        Returns:
            Value Stage identification result containing matched stages
            and the noMatch flag.

        Raises:
            ValueError: If Value Stream ID is missing.
        """
        business_needs = (
            theme_worklet.get_property_value("businessNeeds") or ""
        ).strip()

        # Nothing to extract is a valid no-match result.
        if not business_needs:
            return {
                "matchedValueStages": [],
                "noMatch": True,
            }

        vs_id = theme_worklet.get_property_value("valueStreamId")
        if not vs_id:
            raise ValueError(
                "valueStreamId is required for Value Stage identification."
            )

        matched_stages = self.extract_vss(theme_worklet)

        return {
            "matchedValueStages": matched_stages,
            "noMatch": not matched_stages,
        }
