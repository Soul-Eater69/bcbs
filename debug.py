from typing import Any


class ValueStageHandler:
    async def extract_vss(
        self,
        theme_worklet,
    ) -> list[dict[str, Any]]:
        """
        Returns matched Value Stage entries in the DS output format.
        """

        value_stream_id = theme_worklet.get_property_value(
            "valueStreamId"
        )

        business_needs = (
            theme_worklet.get_property_value("businessNeeds") or ""
        )

        if not business_needs:
            return []

        catalogue = await self.get_vss_catalogue(value_stream_id)

        # Working copy of Business Needs.
        # Once a longer Stage name matches, remove that occurrence
        # so shorter nested Stage names do not match the same text.
        text_norm = business_needs.lower()

        # Longest Stage names first.
        ordered_stages = sorted(
            catalogue,
            key=lambda stage: len(
                stage.get("value_stage_name") or ""
            ),
            reverse=True,
        )

        matches: list[dict[str, Any]] = []

        for stage in ordered_stages:
            stage_name = stage.get("value_stage_name") or ""
            name_norm = stage_name.lower()

            if not name_norm or name_norm not in text_norm:
                continue

            matches.append(
                {
                    "valueStageId": stage["value_stage_id"],
                    "title": stage_name,
                    "description": (
                        stage.get("value_stage_description") or ""
                    ),
                    "entranceCriteria": (
                        stage.get("value_stage_entrance_criteria") or ""
                    ),
                    "exitCriteria": (
                        stage.get("value_stage_exit_criteria") or ""
                    ),
                    "sourceFromBusinessNeeds": True,
                }
            )

            # Remove the matched Stage name from the working text.
            #
            # Example:
            #
            # Business Needs:
            #   "Order to Cash for Coverage ... Order to Cash"
            #
            # After matching "Order to Cash for Coverage":
            #   " ... Order to Cash"
            #
            # The separate "Order to Cash" can still match,
            # while the nested one inside the longer name is gone.
            text_norm = text_norm.replace(name_norm, "")

        return matches
