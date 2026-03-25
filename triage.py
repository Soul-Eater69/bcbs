import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jira_ingestion.clients.jira.value_stream_client import JiraValueStreamClient


# ============================================================
# CONFIG
# ============================================================

JIRA_BASE_URL = os.getenv("JIRA_BASE_URL", "https://your-jira-host")
JIRA_TOKEN = os.getenv("JIRA_TOKEN", "YOUR_JIRA_BEARER_TOKEN")
VERIFY_SSL = os.getenv("JIRA_VERIFY_SSL", "false").lower() == "true"

OUTPUT_DIR = Path("jira_debug_output")
TICKETS = ["IDMT-19761"]


# ============================================================
# HELPERS
# ============================================================

def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"[saved] {path}")


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"<!--\s*Slide number:\s*\d+\s*-->", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"!\[\]\([^)]+\)", " ", text)
    text = re.sub(
        r"PROPRIETARY AND CONFIDENTIAL\s*-\s*FOR INTERNAL USE ONLY",
        " ",
        text,
        flags=re.IGNORECASE,
    )

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

    kept_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(prefix) for prefix in drop_prefixes):
            continue
        kept_lines.append(stripped)

    text = "\n".join(kept_lines)
    text = text.replace("|", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def summarize_text(text: str, max_chars: int = 1800) -> str:
    text = clean_text(text)
    return text[:max_chars]


def build_ticket_text_from_fields(fields: Dict[str, Any]) -> str:
    parts = []

    summary = fields.get("summary")
    description = fields.get("description")

    if summary:
        parts.append(f"Summary: {summary}")
    if description:
        parts.append(f"Description: {description}")

    labels = fields.get("labels") or []
    if labels:
        parts.append("Labels: " + ", ".join([safe_str(x) for x in labels]))

    components = fields.get("components") or []
    component_names = [c.get("name") for c in components if isinstance(c, dict) and c.get("name")]
    if component_names:
        parts.append("Components: " + ", ".join(component_names))

    return "\n\n".join(parts).strip()


def normalize_attachment_content_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handles common shapes from attachment extraction.
    """
    filename = (
        item.get("filename")
        or item.get("name")
        or item.get("attachment_name")
        or "unknown"
    )

    extracted_text = (
        item.get("text_content")
        or item.get("content")
        or item.get("text")
        or item.get("markdown")
        or ""
    )

    status = item.get("status") or ("success" if extracted_text else "unknown")
    mime_type = item.get("mime_type") or item.get("mimeType") or item.get("content_type")
    attachment_id = item.get("id") or item.get("attachment_id")

    return {
        "attachment_id": attachment_id,
        "filename": filename,
        "mime_type": mime_type,
        "status": status,
        "text_content": extracted_text,
        "text_cleaned": clean_text(extracted_text),
        "text_length": len(extracted_text or ""),
    }


def extract_theme_links(raw_ticket: Dict[str, Any]) -> List[Dict[str, Any]]:
    return raw_ticket.get("themes", []) or []


# ============================================================
# TRIAGE
# ============================================================

PREFERRED_EXTENSIONS = [
    ".pptx",
    ".ppt",
    ".pdf",
    ".docx",
    ".doc",
    ".xlsx",
    ".xls",
]


def extension_rank(filename: str) -> int:
    lower = filename.lower()
    for i, ext in enumerate(PREFERRED_EXTENSIONS):
        if lower.endswith(ext):
            return len(PREFERRED_EXTENSIONS) - i
    return 0


def attachment_signal_score(filename: str, text_cleaned: str) -> Tuple[int, List[str]]:
    """
    Heuristic ranking for visible triage.
    You can inspect exactly why something was chosen.
    """
    score = 0
    reasons: List[str] = []

    lower_name = filename.lower()
    lower_text = text_cleaned.lower()

    ext_score = extension_rank(filename)
    if ext_score > 0:
        score += ext_score * 10
        reasons.append(f"preferred_extension:{ext_score}")

    text_len = len(text_cleaned)
    if text_len > 200:
        score += 10
        reasons.append("has_meaningful_text")
    if text_len > 1500:
        score += 15
        reasons.append("long_text")
    if text_len > 5000:
        score += 10
        reasons.append("very_long_text")

    signal_terms = [
        "idea card",
        "executive summary",
        "business case",
        "value proposition",
        "problem statement",
        "market opportunity",
        "project code",
        "product offering",
        "go-to-market",
        "solution",
        "objectives",
        "estimated costs",
        "interdependencies",
    ]
    hits = [term for term in signal_terms if term in lower_text or term in lower_name]
    if hits:
        score += len(hits) * 4
        reasons.append(f"signal_terms:{hits}")

    weak_terms = [
        "theme",
        "themes",
        "notes",
        "tracker",
        "status",
        "log",
    ]
    weak_hits = [term for term in weak_terms if term in lower_name]
    if weak_hits:
        score -= len(weak_hits) * 2
        reasons.append(f"weak_filename_terms:{weak_hits}")

    return score, reasons


def triage_ticket_content(
    raw_ticket: Dict[str, Any],
    attachment_contents: List[Dict[str, Any]],
) -> Dict[str, Any]:
    fields = raw_ticket.get("fields", {}) or {}
    attachments_meta = raw_ticket.get("attachments", []) or []

    normalized_contents = [normalize_attachment_content_item(x) for x in attachment_contents]

    content_by_filename = {
        x["filename"]: x for x in normalized_contents
    }

    scored_attachments = []
    for meta in attachments_meta:
        filename = meta.get("filename") or "unknown"
        extracted = content_by_filename.get(filename, {})
        text_cleaned = extracted.get("text_cleaned", "")

        score, reasons = attachment_signal_score(filename, text_cleaned)

        item = {
            "attachment_id": meta.get("id"),
            "filename": filename,
            "mimeType": meta.get("mimeType"),
            "size": meta.get("size"),
            "created": meta.get("created"),
            "content_url": meta.get("content"),
            "text_length": len(text_cleaned),
            "triage_score": score,
            "triage_reasons": reasons,
            "text_preview": text_cleaned[:600],
        }
        scored_attachments.append(item)

    scored_attachments.sort(
        key=lambda x: (x["triage_score"], x["text_length"], x.get("size") or 0),
        reverse=True,
    )

    primary_attachment = scored_attachments[0] if scored_attachments else None
    supplementary_attachments = scored_attachments[1:4] if len(scored_attachments) > 1 else []

    has_description = bool(fields.get("description"))
    has_attachments = len(scored_attachments) > 0
    has_primary_text = bool(primary_attachment and primary_attachment.get("text_length", 0) > 200)

    if has_primary_text:
        quality_tier = "A"
        selection_reason = "Primary attachment selected with strong extracted text."
    elif has_description and has_attachments:
        quality_tier = "B"
        selection_reason = "Attachments exist but primary attachment text is weak; description remains important."
    elif has_description:
        quality_tier = "C"
        selection_reason = "No strong attachment text; using description-driven ingestion."
    else:
        quality_tier = "D"
        selection_reason = "Very weak evidence; metadata-only / low-confidence ingestion."

    triage_output = {
        "jira_key": raw_ticket.get("key"),
        "quality_tier": quality_tier,
        "selection_reason": selection_reason,
        "has_description": has_description,
        "has_attachments": has_attachments,
        "theme_count": len(extract_theme_links(raw_ticket)),
        "primary_attachment": primary_attachment,
        "supplementary_attachments": supplementary_attachments,
        "all_scored_attachments": scored_attachments,
    }

    return triage_output


# ============================================================
# PRE-CHUNK ASSEMBLY
# ============================================================

def build_retrieval_views(summary: str, description_cleaned: str, primary_attachment_cleaned: str) -> Dict[str, str]:
    overview = "\n\n".join(
        [x for x in [summary, description_cleaned[:3000], primary_attachment_cleaned[:4000]] if x]
    ).strip()

    problem_objective = "\n\n".join(
        [x for x in [summary, description_cleaned[:2500]] if x]
    ).strip()

    attachment_focus = primary_attachment_cleaned[:4500].strip()

    return {
        "overview": overview,
        "problem_objective": problem_objective,
        "attachment_focus": attachment_focus,
    }


def assemble_prechunk_document(
    raw_ticket: Dict[str, Any],
    triage_output: Dict[str, Any],
    attachment_contents: List[Dict[str, Any]],
) -> Dict[str, Any]:
    fields = raw_ticket.get("fields", {}) or {}
    summary = safe_str(fields.get("summary"))
    description_raw = safe_str(fields.get("description"))
    description_cleaned = clean_text(description_raw)

    normalized_contents = [normalize_attachment_content_item(x) for x in attachment_contents]
    content_by_filename = {x["filename"]: x for x in normalized_contents}

    primary_attachment = triage_output.get("primary_attachment")
    primary_filename = primary_attachment.get("filename") if primary_attachment else None
    primary_content = content_by_filename.get(primary_filename, {}) if primary_filename else {}

    primary_attachment_cleaned = safe_str(primary_content.get("text_cleaned", ""))

    retrieval_views = build_retrieval_views(
        summary=summary,
        description_cleaned=description_cleaned,
        primary_attachment_cleaned=primary_attachment_cleaned,
    )

    assembled = {
        "jira_key": raw_ticket.get("key"),
        "summary": summary,
        "description_raw": description_raw,
        "description_cleaned": description_cleaned,
        "themes": extract_theme_links(raw_ticket),
        "labels": fields.get("labels", []) or [],
        "components": fields.get("components", []) or [],
        "triage": {
            "quality_tier": triage_output.get("quality_tier"),
            "selection_reason": triage_output.get("selection_reason"),
            "primary_attachment_filename": primary_filename,
        },
        "primary_attachment_text_cleaned": primary_attachment_cleaned,
        "retrieval_views": retrieval_views,
        "retrieval_text": retrieval_views["overview"],
    }

    return assembled


# ============================================================
# MAIN DEBUG PIPELINE
# ============================================================

async def debug_single_ticket(
    client: JiraValueStreamClient,
    ticket_id: str,
    output_dir: Path,
) -> None:
    ticket_dir = output_dir / ticket_id
    ticket_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== DEBUGGING TICKET: {ticket_id} ===")

    # Stage 1: raw fetch
    raw_ticket = await client.get_ticket_data(ticket_id)
    dump_json(ticket_dir / "01_raw_ticket.json", raw_ticket)

    print("\n[raw fetch]")
    print("keys:", list(raw_ticket.keys()))
    print("theme_count:", len(raw_ticket.get("themes", [])))
    print("attachment_count:", len(raw_ticket.get("attachments", [])))

    # Stage 2: attachment extraction
    attachments = raw_ticket.get("attachments", []) or []
    attachment_contents = await client.fetch_attachment_content(attachments)
    normalized_attachment_contents = [normalize_attachment_content_item(x) for x in attachment_contents]
    dump_json(ticket_dir / "02_attachment_contents.json", normalized_attachment_contents)

    print("\n[attachment extraction]")
    for item in normalized_attachment_contents:
        print(
            f"- {item['filename']} | status={item['status']} | text_length={item['text_length']}"
        )

    # Stage 3: triage
    triage_output = triage_ticket_content(raw_ticket, normalized_attachment_contents)
    dump_json(ticket_dir / "03_triage_output.json", triage_output)

    print("\n[triage result]")
    print("quality_tier:", triage_output["quality_tier"])
    print("selection_reason:", triage_output["selection_reason"])
    if triage_output["primary_attachment"]:
        print("primary_attachment:", triage_output["primary_attachment"]["filename"])
        print("primary_score:", triage_output["primary_attachment"]["triage_score"])

    # Stage 4: assembled pre-chunk document
    assembled_prechunk = assemble_prechunk_document(
        raw_ticket=raw_ticket,
        triage_output=triage_output,
        attachment_contents=normalized_attachment_contents,
    )
    dump_json(ticket_dir / "04_assembled_prechunk.json", assembled_prechunk)

    print("\n[assembled pre-chunk]")
    print("summary:", assembled_prechunk["summary"][:200])
    print("description_cleaned_len:", len(assembled_prechunk["description_cleaned"]))
    print("primary_attachment_text_len:", len(assembled_prechunk["primary_attachment_text_cleaned"]))
    print("retrieval_view_keys:", list(assembled_prechunk["retrieval_views"].keys()))

    # Small human-readable debug file too
    debug_report = {
        "ticket_id": ticket_id,
        "theme_count": len(raw_ticket.get("themes", [])),
        "attachment_count": len(raw_ticket.get("attachments", [])),
        "quality_tier": triage_output["quality_tier"],
        "primary_attachment": (
            triage_output["primary_attachment"]["filename"]
            if triage_output["primary_attachment"]
            else None
        ),
        "primary_attachment_reasons": (
            triage_output["primary_attachment"]["triage_reasons"]
            if triage_output["primary_attachment"]
            else []
        ),
        "retrieval_view_previews": {
            k: v[:800] for k, v in assembled_prechunk["retrieval_views"].items()
        },
    }
    dump_json(ticket_dir / "05_debug_report.json", debug_report)


async def main_async(output_dir: Path, tickets: List[str], verify_ssl: bool) -> None:
    client = JiraValueStreamClient(
        base_url=JIRA_BASE_URL,
        token=JIRA_TOKEN,
        verify_ssl=verify_ssl,
    )

    await client.authenticate()

    for ticket_id in tickets:
        try:
            await debug_single_ticket(client, ticket_id, output_dir)
        except Exception as e:
            error_payload = {
                "ticket_id": ticket_id,
                "error": str(e),
            }
            dump_json(output_dir / ticket_id / "ERROR.json", error_payload)
            print(f"[error] {ticket_id}: {e}")


def main() -> None:
    output_dir = OUTPUT_DIR
    tickets = TICKETS
    verify_ssl = VERIFY_SSL
    asyncio.run(main_async(output_dir, tickets, verify_ssl))


if __name__ == "__main__":
    main()