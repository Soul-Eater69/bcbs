import re
from typing import List, Tuple, Any

import httpx
from rapidfuzz import fuzz
from langchain_core.documents import Document


# ---------- 1) Load + clean vocab ----------

def load_vocab_lines(vocab_path: str) -> List[str]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        return [ln.strip().strip('"').strip("'") for ln in f if ln.strip()]


def vocab_to_entities(vocab_lines: List[str]) -> List[str]:
    """
    Converts vocab lines like:
      'L3 Capability Claim Communication Management'
    into canonical entity names:
      'Claim Communication Management'
    """
    entities = set()
    for raw in vocab_lines:
        parts = raw.split()
        if len(parts) < 2:
            continue

        if raw.startswith(("L1", "L2", "L3")) and len(parts) >= 3:
            name = " ".join(parts[2:]).strip()
        else:
            name = " ".join(parts[1:]).strip()

        if name:
            entities.add(name)

    return sorted(entities)


# ---------- 2) RapidFuzz match entities to response ----------

def _norm(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).lower().strip()


def match_entities_to_text(
    text: str,
    entities: List[str],
    cutoff: int = 85,
    max_terms: int = 15,
) -> List[Tuple[str, int]]:
    """
    Returns: [(entity, score)] sorted desc
    """
    t = _norm(text)
    hits: List[Tuple[str, int]] = []

    for e in entities:
        en = _norm(e)
        if len(en) < 4:
            continue

        score = 100 if en in t else int(fuzz.partial_ratio(en, t))
        if score >= cutoff:
            hits.append((e, score))

    hits.sort(key=lambda x: x[1], reverse=True)
    return hits[:max_terms]


# ---------- 3) Build filter_expression ----------

def _escape_odata(s: str) -> str:
    return s.replace("'", "''")


def build_filter_expression(entity_field: str, matched_entities: List[str]) -> str:
    # entity_name eq 'A' or entity_name eq 'B' ...
    return " or ".join([f"{entity_field} eq '{_escape_odata(e)}'" for e in matched_entities])


# ---------- 4) Call search API once with filter ----------

def search_with_filter(
    *,
    base_url: str,
    search_path: str,
    auth: Any,
    index_name: str,
    semantic_config: str,
    filter_expression: str,
    top: int = 50,
    verify_tls: bool = False,
) -> List[Document]:
    payload = {
        "index_name": index_name,
        "semantic_configuration_name": semantic_config,
        "search_text": "*",
        "search_type": "simple",
        "top": top,
        "filter_expression": filter_expression,
    }

    url = base_url.rstrip("/") + search_path

    with httpx.Client(timeout=30.0, verify=verify_tls) as client:
        r = client.post(url, auth=auth, json=payload)
        r.raise_for_status()
        data = r.json()

    raw_docs = data.get("documents") or data.get("value") or []

    docs: List[Document] = []
    for d in raw_docs:
        content = d.get("content") or ""   # avoid None crash
        docs.append(Document(page_content=str(content), metadata=d))

    return docs


# ---------- MAIN: one function you call ----------

def extract_docs_by_vocab_filter(
    response_text: str,
    vocab_path: str,
    *,
    base_url: str,
    search_path: str,
    auth: Any,
    index_name: str = "idp_kg_data",
    semantic_config: str = "default",
    entity_field: str = "entity_name",
    top: int = 50,
    cutoff: int = 85,
    max_terms: int = 15,
    verify_tls: bool = False,
) -> List[Document]:
    vocab_lines = load_vocab_lines(vocab_path)
    entities = vocab_to_entities(vocab_lines)

    matches = match_entities_to_text(response_text, entities, cutoff=cutoff, max_terms=max_terms)
    matched_entities = [e for e, _ in matches]
    if not matched_entities:
        return []

    flt = build_filter_expression(entity_field, matched_entities)
    docs = search_with_filter(
        base_url=base_url,
        search_path=search_path,
        auth=auth,
        index_name=index_name,
        semantic_config=semantic_config,
        filter_expression=flt,
        top=top,
        verify_tls=verify_tls,
    )

    # (optional) attach debug info
    for d in docs:
        d.metadata["matched_entities"] = matched_entities
        d.metadata["matched_entity_scores"] = matches

    return docs