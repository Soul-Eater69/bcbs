from __future__ import annotations

import re
from typing import List, Tuple, Dict, Optional

from langchain_core.documents import Document
from rapidfuzz import fuzz

from impact_analysis.vector_client import IDPAzureSearchRetriever


INDEX_NAME = "idp_kg_data"
SEMANTIC_CONFIG = "default"

# Candidate retrieval sizes
BM25_K = 30
VEC_K  = 30

# Final selection
FINAL_K = 10
MIN_KEEP = 5

# Fuzzy verification
VERIFY_WINDOW = 1200
FUZZY_CUTOFF = 45  # loose; avoid dropping real used chunks


def extract_mentioned_docs(response: str) -> List[Document]:
    """
    Given an LLM response string, return the list of source docs/chunks that
    were most likely referenced/described in the response.

    Approach:
      1) Candidate generation: BM25 + Vector on the response
      2) Merge + dedupe
      3) Verify/rerank with fuzzy similarity against chunk snippet
      4) Return top docs (keep at least MIN_KEEP)
    """
    # 1) Retrieve candidates
    bm25_docs = _retrieve(response, search_type="simple", k=BM25_K)
    vec_docs  = _retrieve(response, search_type="semantic", k=VEC_K)

    # 2) Merge + dedupe
    candidates = _dedupe(bm25_docs + vec_docs)

    # 3) Verify/rerank (fuzzy)
    ranked = _fuzzy_rerank(response, candidates)

    # 4) Filter loosely but keep minimum
    kept = [d for s, d in ranked if s >= FUZZY_CUTOFF]
    if len(kept) < min(MIN_KEEP, len(ranked)):
        kept = [d for _, d in ranked[:min(MIN_KEEP, len(ranked))]]

    return kept[:FINAL_K]


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _retrieve(text: str, search_type: str, k: int) -> List[Document]:
    retriever = IDPAzureSearchRetriever(
        index_name=INDEX_NAME,
        semantic_configuration_name=SEMANTIC_CONFIG,
        search_type=search_type,
        k=k,
    )
    # Use a truncated query to avoid super long payloads, but keep enough signal
    query = _truncate(text, 1200)
    return retriever.invoke(query)


def _normalize(s: str) -> str:
    s = re.sub(r"\[\d+\]", "", s)          # remove citation markers
    s = re.sub(r"[^\w\s]", " ", s)         # remove punctuation noise
    return re.sub(r"\s+", " ", s).strip().lower()


def _truncate(text: str, max_chars: int) -> str:
    return re.sub(r"\s+", " ", text[:max_chars]).strip()


def _doc_key(d: Document) -> str:
    md = d.metadata or {}
    return str(md.get("chunk_id") or md.get("id") or md.get("source") or md.get("node_id") or d.page_content[:200])


def _dedupe(docs: List[Document]) -> List[Document]:
    seen = set()
    out = []
    for d in docs:
        k = _doc_key(d)
        if k in seen:
            continue
        seen.add(k)
        out.append(d)
    return out


def _fuzzy_rerank(response: str, docs: List[Document]) -> List[Tuple[int, Document]]:
    resp_n = _normalize(response)
    scored: List[Tuple[int, Document]] = []

    for d in docs:
        snippet = _normalize((d.page_content or "")[:VERIFY_WINDOW])
        score = fuzz.token_set_ratio(resp_n, snippet)
        d.metadata["fuzzy_score"] = score
        scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


# ─────────────────────────────────────────────────────────────
# Precision/Recall helper
# ─────────────────────────────────────────────────────────────

def precision_recall(pred_docs: List[Document], gold_docs: List[Document]) -> Dict[str, float]:
    pred_ids = {_doc_key(d) for d in pred_docs}
    gold_ids = {_doc_key(d) for d in gold_docs}

    if not pred_ids and not gold_ids:
        return {"precision": 1.0, "recall": 1.0}

    inter = pred_ids & gold_ids
    precision = len(inter) / len(pred_ids) if pred_ids else 0.0
    recall    = len(inter) / len(gold_ids) if gold_ids else 0.0
    return {"precision": precision, "recall": recall}