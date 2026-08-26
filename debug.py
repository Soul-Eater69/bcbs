import os
import httpx
from dotenv import load_dotenv

load_dotenv()

JIRA_BASE_URL = os.environ["JIRA_BASE_URL"].rstrip("/")
JIRA_TOKEN = os.environ["JIRA_TOKEN"]

HEADERS = {
    "Authorization": f"Bearer {JIRA_TOKEN}",
    "Accept": "application/json",
}

L3_CAP_FIELD_ID = "customfield_18603"


def get_epic_l3_cap(epic_key: str):
    response = httpx.get(
        f"{JIRA_BASE_URL}/rest/api/2/issue/{epic_key}",
        headers=HEADERS,
        params={
            "fields": f"summary,{L3_CAP_FIELD_ID}"
        },
        verify=False,
        timeout=60,
    )

    response.raise_for_status()

    issue = response.json()
    fields = issue.get("fields", {})

    raw_l3 = fields.get(L3_CAP_FIELD_ID) or []

    l3_caps = []

    for item in raw_l3:
        if isinstance(item, dict):
            l3_caps.append({
                "value": item.get("value"),
                "id": item.get("id"),
            })
        else:
            l3_caps.append({
                "value": str(item),
                "id": None,
            })

    return {
        "epic_key": issue.get("key", epic_key),
        "epic_summary": fields.get("summary"),
        "l3_capabilities": l3_caps,
    }


# -----------------------------
# RUN
# -----------------------------

EPIC_KEY = "GROUP-21164"

result = get_epic_l3_cap(EPIC_KEY)

print("Epic Key:", result["epic_key"])
print("Summary :", result["epic_summary"])

print("\nL3 Capabilities:")

for cap in result["l3_capabilities"]:
    print(f"- {cap['value']}  (Jira option id: {cap['id']})")
