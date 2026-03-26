import asyncio
import json
import os
import httpx

JIRA_BASE_URL = os.getenv("JIRA_BASE_URL")
JIRA_TOKEN = os.getenv("JIRA_TOKEN")
VERIFY_SSL = os.getenv("JIRA_VERIFY_SSL", "false").lower() == "true"
TICKET_ID = "IDMT-19761"

async def main():
    url = f"{JIRA_BASE_URL}/rest/api/2/issue/{TICKET_ID}"
    headers = {
        "Authorization": f"Bearer {JIRA_TOKEN}",
        "Accept": "application/json",
    }
    params = {
        "fields": "*all",
        "expand": "names",
    }

    async with httpx.AsyncClient(verify=VERIFY_SSL, timeout=60.0) as client:
        resp = await client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

    with open(f"{TICKET_ID}_raw_direct_jira.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    fields = data.get("fields", {})
    print("customfield keys found:")
    for k in sorted(fields.keys()):
        if k.startswith("customfield_"):
            print(k)

    print("\nSaved:", f"{TICKET_ID}_raw_direct_jira.json")

if __name__ == "__main__":
    asyncio.run(main())