from __future__ import annotations
import re
from typing import List, Tuple, Set, Optional

from rapidfuzz import fuzz
from langchain_core.documents import Document

from impact_analysis.vector_client import IDPAzureSearchRetriever

INDEX_NAME = "idp_kg_data"
SEMANTIC_CONFIG = "default"

ENTITY_FIELD = "entity_name"     # confirm exact field name in index schema
SEARCH_TYPE = "simple"           # BM25
TOP_K = 25

VOCAB_FUZZ_CUTOFF = 85
MAX_MATCHED_TERMS = 15


def _normalize(s: str) -> str:
    s = re.sub(r"\[\d+\]", "", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def _escape_odata(v: str) -> str:
    return v.replace("'", "''")


def load_vocab(path: str) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().strip('"').strip("'")
            if w:
                out.append(w)
    return out


def clean_entity_names(vocab: List[str]) -> List[str]:
    entities: Set[str] = set()
    for raw in vocab:
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) < 2:
            continue

        if raw.startswith(("L1", "L2", "L3")):
            name = " ".join(parts[2:]).strip()     # drop "L3 Capability"
        else:
            name = " ".join(parts[1:]).strip()     # drop first token

        if name:
            entities.add(name)

    return sorted(entities)


def match_entities(response_text: str, entities: List[str]) -> List[Tuple[str, int]]:
    """
    substring first, fuzzy fallback
    returns [(entity, score)] sorted desc
    """
    resp = _normalize(response_text)
    hits: List[Tuple[str, int]] = []

    for ent in entities:
        e = _normalize(ent)
        if len(e) < 4:
            continue

        if e in resp:
            score = 100
        else:
            score = int(fuzz.partial_ratio(e, resp))

        if score >= VOCAB_FUZZ_CUTOFF:
            hits.append((ent, score))

    hits.sort(key=lambda x: x[1], reverse=True)
    return hits[:MAX_MATCHED_TERMS]


def build_filter_expression(matched_entities: List[str]) -> str:
    """
    entity_name eq 'A' or entity_name eq 'B' ...
    """
    clauses = [f"{ENTITY_FIELD} eq '{_escape_odata(e)}'" for e in matched_entities]
    return " or ".join(clauses)


def extract_docs_entity_filtered_one_call(
    response_text: str,
    vocab_path: str,
) -> List[Document]:
    vocab = load_vocab(vocab_path)
    entities = clean_entity_names(vocab)

    hits = match_entities(response_text, entities)
    matched_entities = [e for e, _ in hits]
    if not matched_entities:
        return []

    filter_expression = build_filter_expression(matched_entities)

    # one call
    retriever = IDPAzureSearchRetriever(
        index_name=INDEX_NAME,
        semantic_configuration_name=SEMANTIC_CONFIG,
        search_type=SEARCH_TYPE,
        k=TOP_K,
        filter_expression=filter_expression,   # <— use THIS param name (per your API)
    )

    # query can be tiny because filter already narrows the corpus
    query = " ".join(matched_entities[:5])
    docs = retriever.invoke(query) or []

    # annotate for debugging
    for d in docs:
        md = d.metadata or {}
        md["matched_entities"] = matched_entities
        md["matched_entity_scores"] = hits

    return docs