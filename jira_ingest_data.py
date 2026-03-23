import asyncio
import json
import os
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Requires the repo package to be importable, e.g.
# pip install -e /path/to/jira_ingestion
from jira_ingestion import JiraValueStreamClient


# =========================
# Config
# =========================
JIRA_BASE_URL = os.getenv("JIRA_BASE_URL", "https://your-jira-host")
JIRA_BEARER_TOKEN = os.getenv("JIRA_BEARER_TOKEN", "")
VERIFY_SSL = os.getenv("JIRA_VERIFY_SSL", "false").lower() == "true"

# Optional: map your Jira custom field IDs here.
CUSTOM_FIELDS = {
    "impacted_it_products": os.getenv("JIRA_CF_IMPACTED_IT_PRODUCTS", "customfield_10000"),
    "impacted_products": os.getenv("JIRA_CF_IMPACTED_PRODUCTS", "customfield_10001"),
    "requesting_organization": os.getenv("JIRA_CF_REQUESTING_ORG", "customfield_10002"),
    "delivery_organization": os.getenv("JIRA_CF_DELIVERY_ORG", "customfield_10003"),
}

DEFAULT_FIELDS = [
    "summary",
    "description",
    "attachment",
    "issuelinks",
    "labels",
    "components",
    "priority",
    "issuetype",
    "status",
    "resolution",
    CUSTOM_FIELDS["impacted_it_products"],
    CUSTOM_FIELDS["impacted_products"],
    CUSTOM_FIELDS["requesting_organization"],
    CUSTOM_FIELDS["delivery_organization"],
]


# =========================
# Utility helpers
# =========================
def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(item.get("value") or item.get("name") or str(item))
            else:
                parts.append(str(item))
        return ", ".join([p for p in parts if p])
    if isinstance(value, dict):
        return value.get("value") or value.get("name") or json.dumps(value, ensure_ascii=False)
    return str(value)


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


# =========================
# Jira extraction
# =========================
def _extract_issue_links(issue: Dict[str, Any]) -> Dict[str, List[Dict[str, Optional[str]]]]:
    links = _safe_get(issue, ["fields", "issuelinks"], []) or []
    value_stream_links: List[Dict[str, Optional[str]]] = []
    other_links: List[Dict[str, Optional[str]]] = []

    for link in links:
        link_type = link.get("type", {}) or {}
        outward_name = link_type.get("outward")
        inward_name = link_type.get("inward")

        if outward_name == "implements" and "outwardIssue" in link:
            linked = link["outwardIssue"]
            value_stream_links.append(
                {
                    "direction": "outward",
                    "relationship": outward_name,
                    "key": linked.get("key"),
                    "summary": _safe_get(linked, ["fields", "summary"]),
                    "status": _safe_get(linked, ["fields", "status", "name"]),
                }
            )
        elif inward_name == "implemented by" and "inwardIssue" in link:
            linked = link["inwardIssue"]
            value_stream_links.append(
                {
                    "direction": "inward",
                    "relationship": inward_name,
                    "key": linked.get("key"),
                    "summary": _safe_get(linked, ["fields", "summary"]),
                    "status": _safe_get(linked, ["fields", "status", "name"]),
                }
            )
        else:
            linked_issue = link.get("outwardIssue") or link.get("inwardIssue") or {}
            relationship = outward_name if "outwardIssue" in link else inward_name
            other_links.append(
                {
                    "relationship": relationship,
                    "key": linked_issue.get("key"),
                    "summary": _safe_get(linked_issue, ["fields", "summary"]),
                    "status": _safe_get(linked_issue, ["fields", "status", "name"]),
                }
            )

    return {"value_stream_links": value_stream_links, "other_links": other_links}


def _select_primary_attachment(attachments: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not attachments:
        return None

    preferred_exts = (".pptx", ".ppt", ".pdf", ".docx", ".doc", ".xlsx", ".xls")

    def score(att: Dict[str, Any]):
        filename = (att.get("filename") or "").lower()
        size = att.get("size") or 0
        preferred = 1 if filename.endswith(preferred_exts) else 0
        return (preferred, size)

    return sorted(attachments, key=score, reverse=True)[0]


# =========================
# Preprocessing
# =========================
def _clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"<!--\s*Slide number:\s*\d+\s*-->", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"!\[\]\([^)]+\)", " ", text)
    text = re.sub(r"PROPRIETARY AND CONFIDENTIAL\s*-\s*FOR INTERNAL USE ONLY", " ", text, flags=re.IGNORECASE)
    text = text.replace("|", " ")

    drop_prefixes = [
        "Current as of:",
        "Funding Portfolio:",
        "Project Category:",
        "Project Sub-Category:",
        "Intake BSL:",
        "Business Architect:",
        "EPMO Lead:",
        "Idea Card Author:",
        "Executive Sponsor:",
        "Accountable:",
        "Responsible:",
        "Functional Portfolio Owner:",
        "### Notes:",
        "# Appendix",
    ]

    kept_lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(prefix) for prefix in drop_prefixes):
            continue
        kept_lines.append(stripped)

    text = "\n".join(kept_lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_signal_sections(raw_text: str) -> Dict[str, str]:
    cleaned = (raw_text or "").replace("\r", "")
    section_names = [
        "Idea Card Executive Summary:",
        "Problem Statement/Market Opportunity:",
        "Business Solution and Objectives:",
        "Alternative Solutions:",
        "Value Proposition & Key Metrics:",
        "Interdependencies:",
        "Estimated Costs:",
        "Resources/Investments Needed for Business Case:",
    ]
    positions: List[Tuple[int, str]] = []
    for name in section_names:
        idx = cleaned.find(name)
        if idx != -1:
            positions.append((idx, name))
    positions.sort()

    sections: Dict[str, str] = {}
    for i, (start, name) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(cleaned)
        sections[name] = cleaned[start:end].strip()
    return sections


def _normalize_for_search(text: str, max_chars: int = 2500) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text[:max_chars]


def _build_retrieval_views(summary: str, description: str, primary_attachment_text: str) -> List[Dict[str, Any]]:
    raw = "\n\n".join([summary or "", description or "", primary_attachment_text or ""]).strip()
    cleaned = _clean_text(raw)
    sections = _extract_signal_sections(primary_attachment_text or raw)

    executive = sections.get("Idea Card Executive Summary:", "")
    problem = sections.get("Problem Statement/Market Opportunity:", "")
    solution = sections.get("Business Solution and Objectives:", "")
    alternative = sections.get("Alternative Solutions:", "")
    value_prop = sections.get("Value Proposition & Key Metrics:", "")
    interdep = sections.get("Interdependencies:", "")
    costs = sections.get("Estimated Costs:", "")

    views = [
        {
            "view_name": "cleaned_full",
            "text": _normalize_for_search(cleaned, 2500),
            "why": "whole-ticket cleaned retrieval text",
        },
        {
            "view_name": "problem_objective",
            "text": _normalize_for_search(" ".join([executive, problem, solution]), 2400),
            "why": "business problem and objectives focused view",
        },
        {
            "view_name": "solution_capabilities",
            "text": _normalize_for_search(" ".join([solution, alternative]), 2400),
            "why": "capabilities and proposed solution focused view",
        },
        {
            "view_name": "value_metrics",
            "text": _normalize_for_search(" ".join([value_prop, interdep, costs]), 1800),
            "why": "value proposition, interdependencies, and costs focused view",
        },
    ]

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for v in views:
        key = v["text"].strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(v)
    return deduped


# =========================
# Chunking
# =========================
def _chunk_text(text: str, source_name: str, chunk_type: str, max_chars: int = 1200) -> List[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return []

    # First try sentence-ish / bullet-ish segmentation
    parts = re.split(r"(?:(?<=\.)\s+|\n+|\s+-\s+|\s+•\s+)", text)
    parts = [p.strip() for p in parts if p and p.strip()]

    chunks: List[str] = []
    cur = ""
    for part in parts:
        if len(cur) + len(part) + 1 <= max_chars:
            cur = f"{cur} {part}".strip()
        else:
            if cur:
                chunks.append(cur)
            cur = part[:max_chars]
    if cur:
        chunks.append(cur)

    output = []
    for i, chunk in enumerate(chunks, start=1):
        output.append(
            {
                "chunk_id": f"{source_name}:{chunk_type}:{i}:{_sha(chunk)}",
                "source_name": source_name,
                "chunk_type": chunk_type,
                "chunk_text": chunk,
                "chunk_summary": chunk[:300],
                "salience_score": min(1.0, max(0.2, len(chunk) / max_chars)),
                "extraction_confidence": 1.0,
            }
        )
    return output


# =========================
# Document assembly
# =========================
def _build_ticket_doc(issue: Dict[str, Any], attachment_texts: List[Dict[str, Any]]) -> Dict[str, Any]:
    fields = issue.get("fields", {}) or {}
    links = _extract_issue_links(issue)
    attachments = fields.get("attachment", []) or []
    primary_attachment = _select_primary_attachment(attachments)
    primary_attachment_name = primary_attachment.get("filename") if primary_attachment else None

    primary_attachment_text = ""
    if primary_attachment_name:
        for item in attachment_texts:
            if item.get("filename") == primary_attachment_name:
                primary_attachment_text = item.get("text_content") or ""
                break

    summary = fields.get("summary") or ""
    description = fields.get("description") or ""

    retrieval_views = _build_retrieval_views(summary, description, primary_attachment_text)

    ticket_doc = {
        "jira_key": issue.get("key"),
        "summary": summary,
        "description_raw": description,
        "description_cleaned": _clean_text(description),
        "issue_type": _safe_get(fields, ["issuetype", "name"]),
        "status": _safe_get(fields, ["status", "name"]),
        "resolution": _safe_get(fields, ["resolution", "name"]),
        "priority": _safe_get(fields, ["priority", "name"]),
        "labels": fields.get("labels", []) or [],
        "components": [c.get("name") for c in fields.get("components", []) if c.get("name")],
        "requesting_organization_raw": fields.get(CUSTOM_FIELDS["requesting_organization"]),
        "delivery_organization_raw": fields.get(CUSTOM_FIELDS["delivery_organization"]),
        "impacted_products_raw": fields.get(CUSTOM_FIELDS["impacted_products"]),
        "impacted_it_products_raw": fields.get(CUSTOM_FIELDS["impacted_it_products"]),
        "requesting_organization_text": _to_text(fields.get(CUSTOM_FIELDS["requesting_organization"])),
        "delivery_organization_text": _to_text(fields.get(CUSTOM_FIELDS["delivery_organization"])),
        "impacted_products_text": _to_text(fields.get(CUSTOM_FIELDS["impacted_products"])),
        "impacted_it_products_text": _to_text(fields.get(CUSTOM_FIELDS["impacted_it_products"])),
        "value_stream_links": links["value_stream_links"],
        "other_issue_links": links["other_links"],
        "attachments": [
            {
                "id": a.get("id"),
                "filename": a.get("filename"),
                "mimeType": a.get("mimeType"),
                "size": a.get("size"),
                "content_url": a.get("content"),
            }
            for a in attachments
        ],
        "attachment_texts": attachment_texts,
        "primary_attachment": (
            {
                "id": primary_attachment.get("id"),
                "filename": primary_attachment.get("filename"),
                "mimeType": primary_attachment.get("mimeType"),
                "size": primary_attachment.get("size"),
            }
            if primary_attachment
            else None
        ),
        "primary_attachment_text_cleaned": _clean_text(primary_attachment_text),
        "retrieval_views": retrieval_views,
        "quality_tier": "A" if attachments and description else "B" if attachments or description else "D",
    }
    return ticket_doc


def _build_chunk_docs(ticket_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    jira_key = ticket_doc["jira_key"]
    chunks: List[Dict[str, Any]] = []

    chunks.extend(_chunk_text(ticket_doc.get("description_cleaned", ""), jira_key, "description", 1200))
    chunks.extend(_chunk_text(ticket_doc.get("primary_attachment_text_cleaned", ""), jira_key, "primary_attachment", 1400))

    for rv in ticket_doc.get("retrieval_views", []):
        view_chunks = _chunk_text(rv.get("text", ""), jira_key, f"retrieval_view:{rv.get('view_name')}", 1200)
        for c in view_chunks:
            c["view_name"] = rv.get("view_name")
            c["view_why"] = rv.get("why")
        chunks.extend(view_chunks)

    return chunks


def _build_supervision_doc(ticket_doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "jira_key": ticket_doc["jira_key"],
        "linked_value_stream_ids": [x.get("key") for x in ticket_doc.get("value_stream_links", []) if x.get("key")],
        "linked_value_stream_names": [x.get("summary") for x in ticket_doc.get("value_stream_links", []) if x.get("summary")],
        "linked_value_stream_count": len(ticket_doc.get("value_stream_links", [])),
        "impacted_products_raw": ticket_doc.get("impacted_products_raw"),
        "impacted_it_products_raw": ticket_doc.get("impacted_it_products_raw"),
        "impacted_products_text": ticket_doc.get("impacted_products_text"),
        "impacted_it_products_text": ticket_doc.get("impacted_it_products_text"),
        "label_source": "jira_issue_links_and_ticket_fields",
    }


def _build_debug_report(ticket_doc: Dict[str, Any], chunk_docs: List[Dict[str, Any]], supervision_doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "jira_key": ticket_doc["jira_key"],
        "summary": ticket_doc.get("summary"),
        "quality_tier": ticket_doc.get("quality_tier"),
        "primary_attachment": ticket_doc.get("primary_attachment"),
        "preprocessing_preview": {
            "description_cleaned_preview": ticket_doc.get("description_cleaned", "")[:1200],
            "primary_attachment_cleaned_preview": ticket_doc.get("primary_attachment_text_cleaned", "")[:2000],
        },
        "retrieval_views": ticket_doc.get("retrieval_views", []),
        "chunk_count": len(chunk_docs),
        "chunk_previews": chunk_docs[:8],
        "value_stream_labels": supervision_doc.get("linked_value_stream_names", []),
        "value_stream_ids": supervision_doc.get("linked_value_stream_ids", []),
        "impacted_products_text": supervision_doc.get("impacted_products_text"),
        "impacted_it_products_text": supervision_doc.get("impacted_it_products_text"),
    }


# =========================
# Main fetch / write
# =========================
async def fetch_issue_with_repo_client(ticket_key: str) -> Dict[str, Any]:
    async with JiraValueStreamClient(
        base_url=JIRA_BASE_URL,
        token=JIRA_BEARER_TOKEN,
        verify_ssl=VERIFY_SSL,
    ) as client:
        issue = await client.client.get_issue_by_key(ticket_key, fields=DEFAULT_FIELDS)
        attachments = _safe_get(issue, ["fields", "attachment"], []) or []
        attachment_texts = await client.fetch_attachment_content(attachments)
        return {"issue": issue, "attachment_texts": attachment_texts}


async def run(ticket_key: str, output_dir: str) -> None:
    if not JIRA_BEARER_TOKEN:
        raise RuntimeError("Set JIRA_BEARER_TOKEN in your environment before running.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    data = await fetch_issue_with_repo_client(ticket_key)
    issue = data["issue"]
    attachment_texts = data["attachment_texts"]

    ticket_doc = _build_ticket_doc(issue, attachment_texts)
    chunk_docs = _build_chunk_docs(ticket_doc)
    supervision_doc = _build_supervision_doc(ticket_doc)
    debug_report = _build_debug_report(ticket_doc, chunk_docs, supervision_doc)

    raw_issue_path = output / f"{ticket_key}_raw_issue.json"
    ticket_doc_path = output / f"{ticket_key}_ticket_doc.json"
    chunks_path = output / f"{ticket_key}_chunk_docs.json"
    supervision_path = output / f"{ticket_key}_supervision_doc.json"
    debug_path = output / f"{ticket_key}_debug_report.json"

    raw_issue_path.write_text(json.dumps(issue, indent=2, ensure_ascii=False), encoding="utf-8")
    ticket_doc_path.write_text(json.dumps(ticket_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    chunks_path.write_text(json.dumps(chunk_docs, indent=2, ensure_ascii=False), encoding="utf-8")
    supervision_path.write_text(json.dumps(supervision_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    debug_path.write_text(json.dumps(debug_report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== DONE ===")
    print(f"Raw issue          : {raw_issue_path}")
    print(f"Ticket retrieval   : {ticket_doc_path}")
    print(f"Chunk docs         : {chunks_path}")
    print(f"Supervision doc    : {supervision_path}")
    print(f"Debug report       : {debug_path}")

    print("\n=== QUICK PREVIEW ===")
    print(f"Key                : {ticket_doc['jira_key']}")
    print(f"Summary            : {ticket_doc.get('summary')}")
    print(f"Quality tier       : {ticket_doc.get('quality_tier')}")
    print(f"Primary attachment : {ticket_doc.get('primary_attachment')}")
    print(f"VS labels          : {supervision_doc.get('linked_value_stream_names')}")
    print(f"Impacted products  : {supervision_doc.get('impacted_products_text')}")
    print(f"Impacted IT prods  : {supervision_doc.get('impacted_it_products_text')}")
    print(f"Retrieval views    : {[x['view_name'] for x in ticket_doc.get('retrieval_views', [])]}")
    print(f"Chunk count        : {len(chunk_docs)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sample Jira ingestion using jira_ingestion repo client")
    parser.add_argument("ticket_key", help="Jira key like IDMT-1320")
    parser.add_argument("--output-dir", default="./jira_ingest_output", help="Directory to write JSON artifacts")
    args = parser.parse_args()

    asyncio.run(run(args.ticket_key, args.output_dir))
