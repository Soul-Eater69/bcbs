import asyncio
import json
import os
from pathlib import Path

from jira_ingestion.config import JiraIngestionConfig
from jira_ingestion.clients.jira.value_stream_client import JiraValueStreamClient
from jira_ingestion.ingestion.indexing import create_indexes
from jira_ingestion.ingestion.pipeline import ingest_ticket
from jira_ingestion.ingestion.storage import DocumentStore


# ----------------------------
# Config
# ----------------------------
JIRA_BASE_URL = os.getenv("JIRA_BASE_URL", "https://your-company.atlassian.net")
JIRA_TOKEN = os.getenv("JIRA_TOKEN", "")
TICKET_KEY = os.getenv("TICKET_KEY", "IDEA-1234")

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output/test_ingest")
STORAGE_FMT = os.getenv("STORAGE_FMT", "json")  # json | jsonl | parquet


async def main() -> None:
    if not JIRA_TOKEN:
        raise RuntimeError("Set JIRA_TOKEN in your environment before running this script.")

    # Pipeline runtime config
    config = JiraIngestionConfig(
        max_slides=40,
        max_supplementary=2,
        ocr_enabled=False,
        section_min_slides=8,
        entity_dict_path="data/entity_dicts",
    )

    # Indexes
    # For local testing, "memory" is the safest backend.
    coarse_index, fine_index, metadata_index, supervision_store = create_indexes(
        backend="memory"
    )

    # Jira client
    async with JiraValueStreamClient(
        base_url=JIRA_BASE_URL,
        token=JIRA_TOKEN,
        verify_ssl=True,
    ) as jira_client:
        # Full pipeline ingest
        document = await ingest_ticket(
            ticket_key=TICKET_KEY,
            jira_client=jira_client,
            coarse_index=coarse_index,
            fine_index=fine_index,
            metadata_index=metadata_index,
            supervision_store=supervision_store,
            trigger="backfill",          # or "webhook"
            llm_client=None,             # plug in later if you want summaries from an LLM
            embedding_client=None,       # plug in later if you want real embeddings
            dict_path=config.entity_dict_path,
            force_reprocess=True,        # good for testing
            storage_dir=OUTPUT_DIR,
            storage_fmt=STORAGE_FMT,
            config=config,
        )

    # ----------------------------
    # Inspect outputs
    # ----------------------------
    print("\n=== INGEST COMPLETE ===")
    print("Ticket:", document["ticket_key"])
    print("Schema:", document["schema_version"])
    print("Quality tier:", document["observed"]["quality_tier"])
    print("Content source:", document["observed"]["content_source"])
    print("Chunk count:", document["observed"]["stats"]["chunk_count"])
    print("Section count:", document["observed"]["stats"]["section_count"])
    print("Trainable for VS:", document["supervision"]["trainability"]["is_trainable_for_vs"])

    print("\n=== TRIAGE ===")
    triage = document["observed"]["triage"]
    print("Primary attachment:", triage["primary_attachment"])
    print("Supplementary attachments:", triage["supplementary_attachments"])
    print("Attachment count total:", triage["attachment_count_total"])

    print("\n=== LABELS ===")
    print("VS labels:", document["supervision"]["vs_labels"])

    print("\n=== ENTITY COUNTS ===")
    entity_mentions = document["observed"]["entity_mentions"]
    for entity_type, mentions in entity_mentions.items():
        print(f"{entity_type}: {len(mentions)}")

    print("\n=== INDEX CHECKS ===")
    coarse_doc = coarse_index.get(TICKET_KEY)
    metadata_doc = metadata_index.get(TICKET_KEY)
    supervision_doc = supervision_store.get(TICKET_KEY)

    print("Coarse index present:", coarse_doc is not None)
    print("Metadata index present:", metadata_doc is not None)
    print("Supervision doc present:", supervision_doc is not None)

    # Show first few fine chunks if your backend exposes internal store shape
    # For InMemoryVectorIndex only:
    if hasattr(fine_index, "_store"):
        fine_keys = list(fine_index._store.keys())[:5]
        print("Sample fine chunk ids:", fine_keys)

    # Reload saved artifact from disk
    store = DocumentStore(output_dir=OUTPUT_DIR)
    reloaded = store.load(TICKET_KEY, with_embeddings=True)

    print("\n=== STORED FILES ===")
    print("Output dir:", OUTPUT_DIR)
    print("Store summary:", json.dumps(store.summary(), indent=2))

    if reloaded:
        summary_path = Path(OUTPUT_DIR) / f"{TICKET_KEY}.json"
        print("Reloaded from:", summary_path)
        print("Reloaded quality tier:", reloaded["observed"]["quality_tier"])

    # Optional: dump a tiny preview
    preview = {
        "ticket_key": document["ticket_key"],
        "summary_text": document["observed"]["summary_text"][:1000],
        "metadata_text": document["observed"]["metadata_text"][:1000],
        "vs_labels": document["supervision"]["vs_labels"],
    }
    print("\n=== PREVIEW ===")
    print(json.dumps(preview, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())