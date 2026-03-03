from __future__ import annotations

import re
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from rapidfuzz import fuzz

from impact_analysis.vector_client import IDPAzureSearchRetriever


# ─────────────────────────────────────────────────────────────
# PIPELINE:
# ① split response into sentences
# ② BM25 per sentence → top-1 doc per sentence (precise)
# ③ dedupe
# ④ fuzzy verify against originating sentence (not full response)
# ─────────────────────────────────────────────────────────────

INDEX_NAME      = "idp_kg_data"
SEMANTIC_CONFIG = "default"

FINAL_K             = 15
MIN_SENTENCE_LEN    = 20    # skip very short fragments
FUZZY_VERIFY_CUTOFF = 40    # loose — drop obvious mismatches only
VERIFY_CHUNK_WINDOW = 800


def extract_mentioned_docs(
    response_text: str,
    final_k: int = FINAL_K,
) -> List[Document]:
    """
    Find docs whose content is reflected in the response.
    Searches per sentence so each concept gets its own targeted query.
    """
    sentences = _split_sentences(response_text)
    retriever = _retriever(k=1)

    all_docs: List[Document] = []

    for sentence in sentences:
        results = retriever.invoke(sentence) or []
        if not results:
            continue

        doc   = results[0]
        score = fuzz.token_set_ratio(
            _normalize(sentence),
            _normalize((doc.page_content or "")[:VERIFY_CHUNK_WINDOW]),
        )
        doc.metadata["fuzzy_score"] = score

        if score >= FUZZY_VERIFY_CUTOFF:
            all_docs.append(doc)

    return _dedupe(all_docs)[:final_k]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """Split response into sentences, strip citations, skip short fragments."""
    clean = re.sub(r"\[\d+\]", "", text)
    parts = re.split(r"(?<=[.!?])\s+", clean.strip())
    return [s.strip() for s in parts if len(s.strip()) >= MIN_SENTENCE_LEN]


def _retriever(k: int) -> IDPAzureSearchRetriever:
    return IDPAzureSearchRetriever(
        index_name=INDEX_NAME,
        semantic_configuration_name=SEMANTIC_CONFIG,
        search_type="simple",
        k=k,
    )


def _normalize(s: str) -> str:
    s = re.sub(r"\[\d+\]", "", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def _dedupe(docs: List[Document]) -> List[Document]:
    seen: set = set()
    out: List[Document] = []
    for doc in docs:
        md  = doc.metadata or {}
        key = md.get("chunk_id") or md.get("id") or md.get("source") or doc.page_content[:200]
        if key not in seen:
            seen.add(key)
            out.append(doc)
    return out