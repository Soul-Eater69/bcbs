cases = []

for index, row in df.iterrows():

    # -----------------------------------------------------
    # BUILD HUMAN-READABLE EPIC KEY
    # -----------------------------------------------------

    if EPIC_KEY_COLUMN in df.columns:
        raw_epic_key = to_python(row[EPIC_KEY_COLUMN])

        if isinstance(raw_epic_key, (list, tuple)):
            epic_keys = [
                str(v)
                for v in raw_epic_key
                if v is not None and str(v).strip()
            ]

            epic_key = " | ".join(epic_keys) if epic_keys else str(index)

        elif raw_epic_key is not None and str(raw_epic_key).strip():
            epic_key = str(raw_epic_key)

        else:
            epic_key = str(index)

    else:
        epic_key = str(index)

    # Use the actual Epic key as framework case identity
    case_id = epic_key

    case = EvaluationCase(
        case_id=case_id,

        input="Generate an Epic from the supplied business theme.",

        context={
            "theme_text": to_python(row["theme_text"])
        },

        output=build_generated_epic(row),
    )

    cases.append(case)


print(f"Built {len(cases)} evaluation cases")

for case in cases:
    print(case.case_id)
