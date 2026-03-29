"""
Upload local per-ticket chunk files to Azure AI Search.

Reads only individual files:
  <base_dir>/<ticket_id>/07_chunks.json

It does NOT read <base_dir>/_all_chunks.json.

Behavior:
- keeps original chunk hash id as Azure Search document key
- stores derived uploader-safe key in uploadKey
- normalizes timezone offsets like -0500 -> -05:00
- fills missing createdDate with updatedDate, else current UTC time
- fills missing updatedDate with current UTC time
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from azure.identity import ClientSecretCredential
from azure.search.documents import SearchClient

from src import config


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_datetime_offset(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value

    value = value.strip()
    if not value:
        return None

    # Convert timezone offset from -0600 -> -06:00
    match = re.match(r"^(.*)([+-]\d{2})(\d{2})$", value)
    if match:
        return f"{match.group(1)}{match.group(2)}:{match.group(3)}"

    return value


def _resolve_dates(created: Any, updated: Any) -> tuple[str, str]:
    created_norm = _normalize_datetime_offset(created)
    updated_norm = _normalize_datetime_offset(updated)

    now_iso = _utc_now_iso()

    if updated_norm is None:
        updated_norm = now_iso
    if created_norm is None:
        created_norm = updated_norm or now_iso

    return str(created_norm), str(updated_norm)


def _build_safe_id(source_id: str, chunk_id: str, chunk_index: Any) -> str:
    raw = f"{source_id}|{chunk_id}|{chunk_index}"
    return re.sub(r"[^A-Za-z0-9_\-=]", "_", raw)


def _load_ticket_chunk_files(base_dir: Path) -> list[Path]:
    pattern = str(base_dir / "*" / "07_chunks.json")
    return [Path(p) for p in sorted(glob.glob(pattern))]


def _load_documents(files: list[Path]) -> tuple[list[dict], dict[str, int]]:
    docs: list[dict] = []
    per_ticket_counts: dict[str, int] = {}

    for file_path in files:
        ticket_id = file_path.parent.name
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        chunks = payload.get("chunks", [])
        per_ticket_counts[ticket_id] = len(chunks)

        for doc in chunks:
            item = dict(doc)

            created_date, updated_date = _resolve_dates(
                item.get("createdDate"),
                item.get("updatedDate"),
            )
            item["createdDate"] = created_date
            item["updatedDate"] = updated_date

            provenance = dict(item.get("chunkProvenance") or {})
            if provenance.get("pageRange") is None:
                provenance["pageRange"] = []
            if provenance.get("slideRange") is None:
                provenance["slideRange"] = []
            item["chunkProvenance"] = provenance

            source_id = str(item.get("sourceId") or "")
            chunk_id = str(provenance.get("chunkId") or item.get("id") or "")
            chunk_index = provenance.get("chunkIndex", 0)

            original_id = str(item.get("id") or "").strip()
            safe_id = _build_safe_id(source_id, chunk_id, chunk_index)

            # Keep original pipeline id as the Azure key
            item["id"] = original_id or safe_id

            # Optional debug field to preserve the derived uploader-safe key
            item["uploadKey"] = safe_id

            item["@search.action"] = "mergeOrUpload"

            docs.append(item)

    return docs, per_ticket_counts


def _get_client(index_name: str) -> SearchClient:
    credential = ClientSecretCredential(
        tenant_id=config.AZURE_TENANT_ID,
        client_id=config.AZURE_CLIENT_ID,
        client_secret=config.AZURE_CLIENT_SECRET,
    )
    return SearchClient(
        endpoint=config.AZURE_SEARCH_ENDPOINT,
        index_name=index_name,
        credential=credential,
    )


def _clear_existing_documents(client: SearchClient) -> int:
    to_delete = []
    for row in client.search(search_text="*", select=["id"], top=1000):
        doc_id = row.get("id") if isinstance(row, dict) else row["id"]
        if doc_id:
            to_delete.append({"id": doc_id, "@search.action": "delete"})

    if not to_delete:
        return 0

    client.delete_documents(documents=to_delete)
    return len(to_delete)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload local per-ticket chunk files to Azure AI Search")
    parser.add_argument(
        "--base-dir",
        default="ticket_chunks",
        help="Folder containing per-ticket 07_chunks.json files",
    )
    parser.add_argument(
        "--index",
        default="idmt_data",
        help="Target Azure Search index",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Upload batch size",
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Delete existing docs in target index before upload",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read/transform docs only, do not upload",
    )
    args = parser.parse_args()

    files = _load_ticket_chunk_files(Path(args.base_dir))
    if not files:
        raise SystemExit(f"No per-ticket chunk files found under: {args.base_dir}")

    docs, per_ticket_counts = _load_documents(files)
    unique_ids = {d["id"] for d in docs}

    print(f"ticket_files={len(files)}")
    for ticket_id in sorted(per_ticket_counts):
        print(f"{ticket_id}: chunks={per_ticket_counts[ticket_id]}")
    print(f"total_docs={len(docs)}")
    print(f"unique_ids={len(unique_ids)}")

    same_created_updated = sum(
        1 for d in docs if d.get("createdDate") == d.get("updatedDate")
    )
    print(f"created_equals_updated={same_created_updated}")

    if args.dry_run:
        print("dry_run=True (no Azure writes)")
        return

    client = _get_client(args.index)

    deleted = 0
    if args.clear_existing:
        deleted = _clear_existing_documents(client)
        print(f"deleted_existing={deleted}")

    uploaded_ok = 0
    failed = 0

    for start in range(0, len(docs), args.batch_size):
        batch = docs[start : start + args.batch_size]
        result = client.upload_documents(documents=batch)
        uploaded_ok += sum(1 for r in result if getattr(r, "succeeded", False))
        failed += sum(1 for r in result if not getattr(r, "succeeded", False))

    count_now = client.get_document_count()
    print(f"uploaded_ok={uploaded_ok}")
    print(f"failed={failed}")
    print(f"index_count_now={count_now}")


if __name__ == "__main__":
    main()
