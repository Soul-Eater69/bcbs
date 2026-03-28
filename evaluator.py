import json
import re
from pathlib import Path
from typing import Any, List, Tuple

from langchain_core.messages import HumanMessage
from src.clients.llm import IDPChatOpenAI


CHUNKS_DIR = Path("jira_prechunk_output")
CHUNKS_FILENAME = "07_chunks.json"
MODEL = "gpt-5-mini-idp"
KEYWORDS_PER_CHUNK = 8
MAX_TEXT_LEN = 1200

PROMPT = (
    "Return ONLY a valid JSON array of 5 to 8 lowercase strings. "
    "No markdown, no explanation, no prose.\n\n"
    "Extract high-value retrieval keywords from the text below. "
    "Focus on specific business terms, workflows, products, capabilities, or domain phrases. "
    "Avoid generic words.\n\n"
    "Text:\n{text}"
)


def clip_text(text: str, limit: int = MAX_TEXT_LEN) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:800] + "\n...\n" + text[-400:]


def parse_json_array(text: str) -> List[Any]:
    text = (text or "").strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return data
        except Exception:
            pass

    return []


def normalize_keywords(arr: List[Any], max_keywords: int = KEYWORDS_PER_CHUNK) -> List[str]:
    out = []
    seen = set()

    for x in arr:
        kw = str(x).strip().lower()
        kw = re.sub(r"\s+", " ", kw)
        kw = kw.strip(" ,.;:!?'\"`()[]{}|/\\-")
        if not kw:
            continue
        if kw in seen:
            continue
        seen.add(kw)
        out.append(kw)

        if len(out) >= max_keywords:
            break

    return out


def debug_llm_response(resp):
    print("TYPE:", type(resp))
    print("CONTENT:", repr(getattr(resp, "content", None)))
    print("ADDITIONAL_KWARGS:", getattr(resp, "additional_kwargs", None))
    print("TOOL_CALLS:", getattr(resp, "tool_calls", None))
    print("INVALID_TOOL_CALLS:", getattr(resp, "invalid_tool_calls", None))
    print("RESPONSE_METADATA:", getattr(resp, "response_metadata", None))


def extract_keywords(text: str, llm) -> Tuple[List[str], str]:
    chunk_text = clip_text(text, MAX_TEXT_LEN)
    prompt = PROMPT.format(text=chunk_text)

    resp = llm.invoke(
        [HumanMessage(content=prompt)],
        max_completion_tokens=96,
    )

    debug_llm_response(resp)

    raw_content = getattr(resp, "content", "") or ""
    arr = parse_json_array(raw_content)
    kws = normalize_keywords(arr, KEYWORDS_PER_CHUNK)

    return kws, raw_content