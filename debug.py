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

EPIC_KEY = "GROUP-21164"

L3_FIELDS = {
    "Capability Lvl3": "customfield_12507",
    "L3 Business Capability Model": "customfield_18603",
}

response = httpx.get(
    f"{JIRA_BASE_URL}/rest/api/2/issue/{EPIC_KEY}",
    headers=HEADERS,
    params={
        "fields": "summary," + ",".join(L3_FIELDS.values())
    },
    verify=False,
    timeout=60,
)

response.raise_for_status()

issue = response.json()
fields = issue["fields"]

print("Epic:", issue["key"])
print("Summary:", fields.get("summary"))

for name, field_id in L3_FIELDS.items():
    print(f"\n{name}")
    print("Field ID:", field_id)
    print("Value:", fields.get(field_id))
