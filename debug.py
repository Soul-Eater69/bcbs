import os
import httpx
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

JIRA_BASE_URL = os.environ["JIRA_BASE_URL"].rstrip("/")
JIRA_TOKEN = os.environ["JIRA_TOKEN"]

# Change this only if the Jira field has a different exact name.
L3_FIELD_NAME = "L3 Cap"

VERIFY_SSL = False

HEADERS = {
    "Authorization": f"Bearer {JIRA_TOKEN}",
    "Accept": "application/json",
}


# ---------------------------------------------------------
# FIND CUSTOM FIELD ID
# ---------------------------------------------------------

def find_custom_field_id(field_name: str) -> str:
    """
    Find the Jira custom field ID for a field name.

    Example:
        "L3 Cap" -> "customfield_12345"
    """

    response = httpx.get(
        f"{JIRA_BASE_URL}/rest/api/2/field",
        headers=HEADERS,
        verify=VERIFY_SSL,
        timeout=60,
    )
    response.raise_for_status()

    target = field_name.strip().lower()

    for field in response.json():
        name = str(field.get("name", "")).strip().lower()

        if name == target:
            return field["id"]

    raise ValueError(
        f"Could not find Jira field named {field_name!r}"
    )


# ---------------------------------------------------------
# CLEAN COMMON JIRA FIELD VALUES
# ---------------------------------------------------------

def clean_jira_value(value):
    """
    Convert common Jira custom-field structures into readable values.

    Keeps the function intentionally simple.
    """

    if value is None:
        return None

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (int, float, bool)):
        return value

    if isinstance(value, dict):
        # Common Jira custom-field structures
        for key in ("value", "name", "displayName", "key"):
            if value.get(key) is not None:
                return value[key]

        # If Jira returns some other structure,
        # keep it instead of throwing information away.
        return value

    if isinstance(value, list):
        return [clean_jira_value(item) for item in value]

    return value


# ---------------------------------------------------------
# GET L3 CAP FOR ONE EPIC
# ---------------------------------------------------------

def get_epic_l3_cap(epic_key: str) -> dict:
    """
    Fetch L3 Cap data for one Jira Epic.

    Returns:
        {
            "epic_key": "...",
            "field_name": "L3 Cap",
            "field_id": "customfield_xxxxx",
            "l3_cap": ...,
            "raw_value": ...
        }
    """

    # Find the custom field id
    field_id = find_custom_field_id(L3_FIELD_NAME)

    # Fetch only the Epic key/summary and L3 Cap field
    response = httpx.get(
        f"{JIRA_BASE_URL}/rest/api/2/issue/{epic_key}",
        headers=HEADERS,
        params={
            "fields": f"summary,{field_id}"
        },
        verify=VERIFY_SSL,
        timeout=60,
    )
    response.raise_for_status()

    issue = response.json()
    fields = issue.get("fields", {})

    raw_value = fields.get(field_id)

    return {
        "epic_key": issue.get("key", epic_key),
        "epic_summary": fields.get("summary"),
        "field_name": L3_FIELD_NAME,
        "field_id": field_id,
        "l3_cap": clean_jira_value(raw_value),
        "raw_value": raw_value,
    }


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

EPIC_KEY = "GROUP-21164"

result = get_epic_l3_cap(EPIC_KEY)

print("Epic Key :", result["epic_key"])
print("Summary  :", result["epic_summary"])
print("Field ID :", result["field_id"])
print("L3 Cap   :", result["l3_cap"])

print("\nRaw Jira value:")
print(result["raw_value"])
