import os
import json
import re
from pathlib import Path
from typing import List, Any

from src.clients.llm import IDPChatOpenAI


# ============================================================
# CONFIG
# ============================================================

CHUNKS_DIR = Path("jira_prechunk_output")
CHUNKS_FILENAME = "07_chunks.json"
MODEL = "gpt-5-mini-idp"
KEYWORDS_PER_CHUNK = 8
MAX_TEXT_LEN = 1200

# Words/phrases too generic to be useful as retrieval keywords by themselves
GENERIC_KEYWORDS = {
    "project",
    "initiative",
    "support",
    "business",
    "solution",
    "process",
    "team",
    "member",
    "care",
    "system",
    "platform",
    "improvement",
    "program",
    "service",
    "workflow",
    "operations",
    "data",
    "information",
    "details",
    "issue",
    "work",
}


# ============================================================
# PROMPT
# ============================================================

PROMPT = (
    "Extract 5 to 8 high-value retrieval keywords from the text below. "
    "Focus on specific business terms, domain phrases, workflows, products, capabilities, or problem areas "
    "that would help semantic search for Jira tickets, idea cards, value streams, stages, or impacted products. "
    "Prefer short noun phrases over vague single words. "
    "Use only terms clearly grounded in the text. "
    "Avoid generic words like 'project', 'business', 'process', 'solution', 'support', 'team', 'member', "
    "or 'care' unless they appear as part of a specific phrase. "
    "Return only a valid JSON array of lowercase strings with no explanation.\n\n"
    "Text:\n{text}"
)


# ============================================================
# TEXT HELPERS
# ============================================================

def clip_text(text: str, limit: int = MAX_TEXT_LEN) -> str:
    """
    For long chunks, keep some beginning and some ending context.
    """
    text = (text or "").strip()
    if len(text) <= limit:
        return text

    head_len = int(limit * 0.67)
    tail_len = limit - head_len - 5  # account for "\n...\n"
    head = text[:head_len]
    tail = text[-tail_len:] if tail_len > 0 else ""
    return head + "\n...\n" + tail


def normalize_keyword(keyword: Any) -> str:
    kw = str(keyword).strip().lower()

    # collapse whitespace
    kw = re.sub(r"\s+", " ", kw)

    # trim punctuation edges
    kw = kw.strip(" ,.;:!?'\"`()[]{}|/\\-")

    return kw


def is_useful_keyword(kw: str) -> bool:
    if not kw:
        return False

    if len(kw) < 3:
        return False

    if kw in GENERIC_KEYWORDS:
        return False

    # discard pure numbers
    if re.fullmatch(r"\d+", kw):
        return False

    return True


def normalize_keywords(arr: List[Any], max_keywords: int = KEYWORDS_PER_CHUNK) -> List[str]:
    out: List[str] = []
    seen = set()

    for item in arr:
        kw = normalize_keyword(item)
        if not is_useful_keyword(kw):
            continue
        if kw in seen:
            continue
        seen.add(kw)
        out.append(kw)

        if len(out) >= max_keywords:
            break

    return out


# ============================================================
# PARSING HELPERS
# ============================================================

def parse_json_array(text: str) -> List[Any]:
    """
    Safely parse a JSON array from model output.
    Handles:
    - raw JSON arrays
    - fenced code blocks
    - extra text around JSON
    """
    text = (text or "").strip()

    # 1. direct parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # 2. fenced code block
    fenced_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text, re.IGNORECASE)
    if fenced_match:
        try:
            data = json.loads(fenced_match.group(1))
            if isinstance(data, list):
                return data
        except Exception:
            pass

    # 3. first array-looking block
    array_match = re.search(r"\[[\s\S]*\]", text)
    if array_match:
        candidate = array_match.group(0)
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return data
        except Exception:
            pass

    return []


# ============================================================
# LLM EXTRACTION
# ============================================================

def extract_keywords(text: str, llm) -> List[str]:
    chunk_text = clip_text(text, MAX_TEXT_LEN)
    prompt = PROMPT.format(text=chunk_text)

    resp = llm.invoke(
        [{"role": "user", "content": prompt}],
        max_completion_tokens=96,
    )

    raw_content = getattr(resp, "content", "") or ""
    arr = parse_json_array(raw_content)
    kws = normalize_keywords(arr, KEYWORDS_PER_CHUNK)

    return kws


# ============================================================
# FILE PROCESSING
# ============================================================

def process_chunks_file(path: Path, llm) -> None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    chunks = data.get("chunks", [])
    if not isinstance(chunks, list):
        print(f"Invalid format (no chunks list): {path}")
        return

    changed = False
    debug_rows = []

    for idx, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue

        existing = chunk.get("contextKeywords")
        if existing:
            debug_rows.append(
                {
                    "chunk_index": idx,
                    "status": "skipped_existing",
                    "content_preview": str(chunk.get("content", ""))[:200],
                    "contextKeywords": existing,
                }
            )
            continue

        content = chunk.get("content", "") or ""
        if not content.strip():
            chunk["contextKeywords"] = []
            debug_rows.append(
                {
                    "chunk_index": idx,
                    "status": "empty_content",
                    "content_preview": "",
                    "contextKeywords": [],
                }
            )
            changed = True
            continue

        kws = extract_keywords(content, llm)
        chunk["contextKeywords"] = kws
        changed = True

        debug_rows.append(
            {
                "chunk_index": idx,
                "status": "updated",
                "content_preview": content[:200],
                "contextKeywords": kws,
            }
        )

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Updated: {path}")
    else:
        print(f"No changes: {path}")

    # Save per-file debug report
    debug_path = path.with_name(path.stem + "_keywords_debug.json")
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_file": str(path),
                "chunk_count": len(chunks),
                "rows": debug_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Debug saved: {debug_path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    llm = IDPChatOpenAI(model=MODEL)

    if not CHUNKS_DIR.exists():
        print(f"Chunks directory does not exist: {CHUNKS_DIR}")
        return

    for ticket_dir in CHUNKS_DIR.iterdir():
        if not ticket_dir.is_dir():
            continue

        chunks_path = ticket_dir / CHUNKS_FILENAME
        if chunks_path.exists():
            process_chunks_file(chunks_path, llm)
        else:
            print(f"Missing file: {chunks_path}")


if __name__ == "__main__":
    main()