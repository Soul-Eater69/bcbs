from __future__ import annotations

import re
from typing import Dict, List, Optional

from jira_ingestion.ingestion.extraction.text_cleaning import clean_extracted_text

_SECTION_NAMES = [
    "Idea Card Executive Summary",
    "Problem Statement/Market Opportunity",
    "Business Solution and Objectives",
    "Alternative Solutions",
    "Value Proposition & Key Metrics",
    "Interdependencies",
    "Estimated Costs",
    "Resources/Investments Needed for Business Case",
]

_DROP_PREFIXES = (
    "current as of:",
    "funding portfolio:",
    "project category:",
    "project sub-category:",
    "intake bsl:",
    "business architect:",
    "epmo lead:",
    "idea card author:",
    "executive sponsor:",
    "accountable:",
    "responsible:",
    "functional portfolio owner:",
    "notes:",
    "appendix",
)

_SLIDE_MARKER_RE = re.compile(r"(?im)^\s*(?:slide\s*(?:number)?\s*[:#-]?\s*\d+|page\s*\d+)\s*$")
_BRACKET_MARKER_RE = re.compile(r"\[(?:\d+|IVXLCM+)]")


def clean_ppt_text(raw_text: Optional[str]) -> str:
    """Robust cleaner for value-stream PPT / PDF text.

    Uses the shared extraction cleaner first, then removes a small set of known idea-card boilerplate lines.
    """
    text = clean_extracted_text(raw_text)
    if not text:
        return ""

    kept: List[str] = []
    for raw in text.splitlines():
        line = _SLIDE_MARKER_RE.sub("", raw).strip()
        line = _BRACKET_MARKER_RE.sub("", line).strip()
        if not line:
            continue
        lower = line.lower()
        if any(lower.startswith(prefix) for prefix in _DROP_PREFIXES):
            continue
        if "proprietary and confidential" in lower and "internal use only" in lower:
            continue
        kept.append(line)

    text = "\n".join(kept)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_signal_sections(raw_text: str) -> Dict[str, str]:
    cleaned = clean_ppt_text(raw_text)
    lower = cleaned.lower()

    positions = []
    for name in _SECTION_NAMES:
        idx = lower.find(name.lower())
        if idx != -1:
            positions.append((idx, name))

    positions.sort()
    sections: Dict[str, str] = {}
    for i, (start, name) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(cleaned)
        sections[name] = cleaned[start:end].strip()
    return sections


def normalize_for_search(text: Optional[str], max_chars: int = 2500) -> str:
    cleaned = clean_ppt_text(text)
    cleaned = cleaned.lower()
    cleaned = re.sub(r"[^a-z0-9\n\-_/&.%$#+ ]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_chars]
