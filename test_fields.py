import asyncio
import json
import os
from typing import Any

from jira_ingestion import JiraValueStreamClient


JIRA_BASE_URL = os.getenv("JIRA_BASE_URL", "https://your-jira-host")
JIRA_TOKEN = os.getenv("JIRA_TOKEN", "YOUR_JIRA_BEARER_TOKEN")
VERIFY_SSL = os.getenv("JIRA_VERIFY_SSL", "false").lower() == "true"

TICKET_ID = "IDMT-19761"

FIELD_MAP = {
    "impacted_products": os.getenv("JIRA_CF_IMPACTED_PRODUCTS", "customfield_10040"),
    "impacted_it_products": os.getenv("JIRA_CF_IMPACTED_IT_PRODUCTS", "customfield_10041"),
    "requesting_org": os.getenv("JIRA_CF_REQUESTING_ORG", "customfield_10050"),
    "delivery_org": os.getenv("JIRA_CF_DELIVERY_ORG", "customfield_10051"),
    "product_stage": os.getenv("JIRA_CF_PRODUCT_STAGE", "customfield_10060"),
}


def preview_value(value: Any, max_len: int = 500) -> str:
    try:
        text = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)

    if len(text) > max_len:
        return text[:max_len] + "\n...<truncated>..."
    return text


async def main() -> None:
    async with JiraValueStreamClient(
        base_url=JIRA_BASE_URL,
        token=JIRA_TOKEN,
        verify_ssl=VERIFY_SSL,
    ) as client:
        await client.authenticate()

        # In your current code path, get_ticket_data() fetches all fields.
        data = await client.get_ticket_data(TICKET_ID)
        fields = data.get("fields", {})

        print(f"\nTicket: {TICKET_ID}")
        print(f"Total field count returned: {len(fields)}")

        print("\n=== Checking mapped custom fields ===")
        for logical_name, jira_field_id in FIELD_MAP.items():
            exists = jira_field_id in fields
            value = fields.get(jira_field_id)

            print(f"\nLogical field   : {logical_name}")
            print(f"Jira field id   : {jira_field_id}")
            print(f"Present in issue: {exists}")
            print(f"Value type      : {type(value).__name__ if exists else 'N/A'}")

            if exists:
                print("Value preview:")
                print(preview_value(value))
            else:
                print("Value preview: <missing>")

        print("\n=== Standard fields sanity check ===")
        for key in ["summary", "description", "labels", "components", "attachment", "issuelinks", "comment"]:
            print(f"{key}: {'present' if key in fields else 'missing'}")

        print("\n=== All returned customfield keys ===")
        custom_keys = sorted([k for k in fields.keys() if k.startswith('customfield_')])
        for k in custom_keys:
            print(k)

        print("\n=== Closest matching keys for your FIELD_MAP values ===")
        wanted = set(FIELD_MAP.values())
        intersection = sorted(wanted.intersection(custom_keys))
        if intersection:
            for k in intersection:
                print(f"{k} -> present")
        else:
            print("None of the mapped customfield ids were present in this issue.")

        # Save full response for manual inspection
        with open(f"{TICKET_ID}_field_check_output.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        print(f"\nSaved full output to: {TICKET_ID}_field_check_output.json")


if __name__ == "__main__":
    asyncio.run(main())