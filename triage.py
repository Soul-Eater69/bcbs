import asyncio
import json
import os
from pathlib import Path
from typing import Any

from jira_ingestion import (
    JiraIngestionConfig,
    JiraValueStreamClient,
    create_indexes,
    ingest_ticket,
)


JIRA_BASE_URL = os.getenv("JIRA_BASE_URL", "https://your-jira-host")
JIRA_TOKEN = os.getenv("JIRA_TOKEN", "YOUR_JIRA_BEARER_TOKEN")
VERIFY_SSL = os.getenv("JIRA_VERIFY_SSL", "false").lower() == "true"

TICKETS = ["IDMT-19761"]
OUTPUT_DIR = Path("jira_prechunk_output")


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def safe_preview(path: Path, limit: int = 1500) -> str:
    if not path.exists():
        return "<missing>"
    try:
        return path.read_text(encoding="utf-8")[:limit]
    except Exception as e:
        return f"<unable to read: {e}>"


def build_config() -> JiraIngestionConfig:
    """
    No custom fields for now.
    We only want the normal pipeline flow and the pre-chunk artifact.
    """
    return JiraIngestionConfig(
        enable_raw_artifact_persistence=True,
        enable_attachment_text_persistence=True,
        enable_debug_stage_persistence=True,
        enable_prechunk_persistence=True,
        enable_attachment_inventory=True,
        enable_retrieval_views=True,
        metadata_store_path=str(OUTPUT_DIR / "metadata_store.json"),
        supervision_store_path=str(OUTPUT_DIR / "supervision_store.json"),
    )


def build_indexes():
    """
    In-memory vector indexes are fine because the goal here is inspection,
    not production ingestion.
    """
    coarse_index, fine_index, metadata_index, supervision_store = create_indexes(
        backend="memory",
        metadata_store_path=str(OUTPUT_DIR / "metadata_store.json"),
        supervision_store_path=str(OUTPUT_DIR / "supervision_store.json"),
    )
    return coarse_index, fine_index, metadata_index, supervision_store


async def run_one(
    ticket_id: str,
    client: JiraValueStreamClient,
    coarse_index,
    fine_index,
    metadata_index,
    supervision_store,
    config: JiraIngestionConfig,
) -> None:
    ticket_dir = OUTPUT_DIR / ticket_id
    ticket_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Running pipeline for {ticket_id} ===")

    result = await ingest_ticket(
        ticket_key=ticket_id,
        jira_client=client,
        coarse_index=coarse_index,
        fine_index=fine_index,
        metadata_index=metadata_index,
        supervision_store=supervision_store,
        storage_dir=str(ticket_dir),
        config=config,
        llm_client=None,
        embedding_client=None,
    )

    dump_json(ticket_dir / "00_ingest_result.json", result)

    prechunk_path = ticket_dir / "04_assembled_prechunk.json"

    if not prechunk_path.exists():
        print("\n[warning] 04_assembled_prechunk.json was not generated.")
        return

    print("\nGenerated:")
    print(f"- {prechunk_path}")

    # Read the prechunk JSON so you can inspect the summary + final extracted data
    with open(prechunk_path, "r", encoding="utf-8") as f:
        prechunk = json.load(f)

    # Save a smaller summary-only inspection file too
    inspection = {
        "ticket_id": ticket_id,
        "summary_text": (
            prechunk.get("observed", {}).get("summary_text")
            or prechunk.get("summary_text")
        ),
        "retrieval_views": (
            prechunk.get("observed", {}).get("retrieval_views")
            or prechunk.get("retrieval_views")
        ),
        "provenance": (
            prechunk.get("observed", {}).get("provenance")
            or prechunk.get("provenance")
        ),
        "primary_attachment_name": (
            (
                prechunk.get("observed", {})
                .get("provenance", {})
                .get("primary_evidence_name")
            )
            or prechunk.get("primary_attachment_name")
        ),
    }

    dump_json(ticket_dir / "06_prechunk_summary_only.json", inspection)

    print("\n=== Summary Preview ===")
    print(inspection["summary_text"])

    print("\n=== Prechunk Preview ===")
    print(safe_preview(prechunk_path, limit=2000))


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = build_config()
    coarse_index, fine_index, metadata_index, supervision_store = build_indexes()

    async with JiraValueStreamClient(
        base_url=JIRA_BASE_URL,
        token=JIRA_TOKEN,
        verify_ssl=VERIFY_SSL,
    ) as client:
        for ticket_id in TICKETS:
            try:
                await run_one(
                    ticket_id=ticket_id,
                    client=client,
                    coarse_index=coarse_index,
                    fine_index=fine_index,
                    metadata_index=metadata_index,
                    supervision_store=supervision_store,
                    config=config,
                )
            except Exception as e:
                err_path = OUTPUT_DIR / ticket_id / "ERROR.json"
                dump_json(err_path, {"ticket_id": ticket_id, "error": str(e)})
                print(f"[error] {ticket_id}: {e}")


if __name__ == "__main__":
    asyncio.run(main())