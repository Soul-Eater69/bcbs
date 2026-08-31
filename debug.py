"""Value Stage identification from Theme Business Needs."""

from typing import Any

from jwg_app.domain.exceptions.custom_exception import CustomException
from jwg_app.domain.models.worklet import Worklet
from jwg_app.domain.services.value_stage_service import ValueStageService


class ValueStageHandler:
    """Identify governed Value Stages referenced in Theme Business Needs."""

    PAGE_SIZE = 100

    def __init__(self, value_stage_service: ValueStageService):
        """Initialize the Value Stage handler.

        Args:
            value_stage_service: Service used to retrieve Value Stages
                scoped to a Value Stream.
        """
        self.value_stage_service = value_stage_service

    async def get_vss_catalogue(
        self,
        value_stream_id: str,
    ) -> list[dict[str, Any]]:
        """Return all Value Stages for a Value Stream.

        Args:
            value_stream_id: Value Stream identifier.

        Returns:
            Value Stage catalogue entries scoped to the Value Stream.
        """
        first_page = await self.value_stage_service.list_search(
            value_stream_id=value_stream_id,
            page=1,
            page_size=self.PAGE_SIZE,
            view="summary",
        )

        stages = list(first_page.get("items", []))

        pagination = first_page.get("pagination", {})
        total_pages = pagination.get("total_pages", 0)

        for page in range(2, total_pages + 1):
            result = await self.value_stage_service.list_search(
                value_stream_id=value_stream_id,
                page=page,
                page_size=self.PAGE_SIZE,
                view="summary",
            )

            stages.extend(result.get("items", []))

        return stages

    async def extract_vss(
        self,
        theme_worklet: Worklet,
    ) -> list[dict[str, Any]]:
        """Extract Value Stages referenced in Theme Business Needs.

        Matching is deterministic and case-insensitive. Longer Value Stage
        names are evaluated first so a shorter stage contained within an
        already matched longer stage is not selected again.

        Args:
            theme_worklet: Theme worklet containing Business Needs and
                Value Stream ID.

        Returns:
            Matched Value Stage entries in the DS output format.
        """
        value_stream_id = theme_worklet.get_property_value(
            "valueStreamId"
        )
        business_needs = (
            theme_worklet.get_property_value("businessNeeds") or ""
        )

        catalogue = await self.get_vss_catalogue(value_stream_id)

        text_norm = business_needs.lower()

        ordered_stages = sorted(
            catalogue,
            key=lambda stage: len(
                stage.get("value_stage_name", "")
            ),
            reverse=True,
        )

        matches: list[dict[str, Any]] = []

        for stage in ordered_stages:
            stage_name = stage.get("value_stage_name") or ""
            name_norm = stage_name.lower()

            if not name_norm or name_norm not in text_norm:
                continue

            # A longer matching stage was already selected.
            #
            # Example:
            #   "Generate Quote and Present to Customer"
            #   "Generate Quote"
            #
            # If the longer stage matched first, do not also return
            # the shorter stage contained within it.
            if any(
                name_norm in match["title"].lower()
                for match in matches
            ):
                continue

            matches.append(
                {
                    "valueStageId": stage["value_stage_id"],
                    "title": stage_name,
                    "description": (
                        stage.get("value_stage_description") or ""
                    ),
                    "sourceFromBusinessNeeds": True,
                }
            )

        return matches

    async def run(
        self,
        theme_worklet: Worklet,
    ) -> dict[str, Any]:
        """Run Value Stage identification for a Theme.

        Args:
            theme_worklet: Theme worklet to evaluate.

        Returns:
            DS Value Stage identification response containing
            matchedValueStages and noMatch.

        Raises:
            CustomException: If the Theme has Business Needs but does not
                contain a Value Stream ID.
        """
        business_needs = (
            theme_worklet.get_property_value("businessNeeds") or ""
        ).strip()

        # No Business Needs is a valid no-match result.
        if not business_needs:
            return {
                "matchedValueStages": [],
                "noMatch": True,
            }

        value_stream_id = theme_worklet.get_property_value(
            "valueStreamId"
        )

        if not value_stream_id:
            raise CustomException(
                status_code=400,
                detail=(
                    "valueStreamId is required for "
                    "Value Stage identification."
                ),
            )

        matched_stages = await self.extract_vss(theme_worklet)

        return {
            "matchedValueStages": matched_stages,
            "noMatch": not matched_stages,
        }
