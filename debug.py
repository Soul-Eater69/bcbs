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

response = httpx.get(
    f"{JIRA_BASE_URL}/rest/api/2/field",
    headers=HEADERS,
    verify=False,
    timeout=60,
)

response.raise_for_status()

fields = response.json()

matches = []

for field in fields:
    name = str(field.get("name", ""))

    if "l3" in name.lower() or "cap" in name.lower():
        matches.append({
            "id": field.get("id"),
            "name": name,
        })

for item in matches:
    print(item)
