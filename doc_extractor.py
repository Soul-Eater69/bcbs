from __future__ import annotations

import re
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from rapidfuzz import fuzz, process

from impact_analysis.vector_client import IDPAzureSearchRetriever


# ─────────────────────────────────────────────────────────────
# PIPELINE:
# ① vocab match (rapidfuzz partial_ratio against normalized terms)
# ② BM25 per-term (precision) + BM25 on response (recall)
# ③ merge + dedupe
# ④ BM25 relative threshold (top * 0.70) + MIN_KEEP safety
# ⑤ fuzzy verify (loose) on chunk snippet
# ─────────────────────────────────────────────────────────────

INDEX_NAME      = "idp_kg_data"
SEMANTIC_CONFIG = "default"

BM25_K_RESPONSE     = 25
BM25_K_PER_TERM     = 3
FINAL_K             = 10
BM25_REL_CUTOFF     = 0.70
MIN_KEEP            = 5
MAX_MATCHED_TERMS   = 20
FUZZY_VERIFY_CUTOFF = 50
VERIFY_CHUNK_WINDOW = 1200
VOCAB_FUZZ_CUTOFF   = 85


def extract_mentioned_docs(
    response_text: str,
    vocabulary: Optional[List[str]] = None,
    final_k: int = FINAL_K,
) -> List[Document]:
    """Given an LLM response, return the source chunks the retriever likely used."""
    matched_terms = _match_vocab_terms(response_text, vocabulary or [])

    term_docs = _bm25_per_term(matched_terms) if matched_terms else []
    resp_docs = _bm25_full_response(response_text, matched_terms)

    candidates = _dedupe(term_docs + resp_docs)
    narrowed   = _bm25_threshold(candidates)
    verified   = _fuzzy_verify(response_text, narrowed)

    return verified[:final_k]


def load_vocabulary(filepath: str) -> List[str]:
    """Load entity names from sightline_vocabulary.txt, one term per line."""
    with open(filepath, encoding="utf-8") as f:
        return [line.strip().strip('"') for line in f if line.strip()]


# ── Vocab matching ────────────────────────────────────────────────────────────

def _match_vocab_terms(response_text: str, vocabulary: List[str]) -> List[str]:
    """
    Match vocab terms against the response using rapidfuzz.
    Vocab terms are "Application <name>" — LLM writes just "<name>" —
    so substring match never works. partial_ratio handles the prefix mismatch.
    """
    resp_norm  = _normalize(response_text)
    vocab_norm = [_normalize(t) for t in vocabulary]

    hits = process.extract(
        resp_norm,
        vocab_norm,
        scorer=fuzz.partial_ratio,
        score_cutoff=VOCAB_FUZZ_CUTOFF,
        limit=MAX_MATCHED_TERMS,
    )

    matched_norms = {norm for norm, _, _ in hits}
    return [t for t, n in zip(vocabulary, vocab_norm) if n in matched_norms]


# ── BM25 ──────────────────────────────────────────────────────────────────────

def _bm25_per_term(terms: List[str]) -> List[Document]:
    """One BM25 query per vocab term — canonical name ranks its doc at position 1."""
    retriever = _retriever(BM25_K_PER_TERM)
    docs: List[Document] = []
    for term in terms:
        docs.extend(retriever.invoke(term))
    return docs


def _bm25_full_response(response_text: str, matched_terms: List[str]) -> List[Document]:
    """BM25 on response — appends matched terms that appear beyond the 800-char window."""
    query  = _truncate(response_text)
    q_norm = _normalize(query)
    for term in matched_terms:
        t_norm = _normalize(term)
        if t_norm and t_norm not in q_norm:
            query  += f" {term}"
            q_norm += f" {t_norm}"
    return _retriever(BM25_K_RESPONSE).invoke(query)


# ── Threshold ─────────────────────────────────────────────────────────────────

def _bm25_threshold(docs: List[Document]) -> List[Document]:
    """Relative threshold: keep docs >= top_score * 0.70. Falls back to top-k if scores missing."""
    if not docs:
        return []

    sorted_docs = sorted(docs, key=_bm25_score, reverse=True)
    top_score   = _bm25_score(sorted_docs[0])

    if top_score == 0.0:
        return sorted_docs[:min(MIN_KEEP, len(sorted_docs))]

    cutoff = top_score * BM25_REL_CUTOFF
    kept   = [d for d in sorted_docs if _bm25_score(d) >= cutoff]

    if len(kept) < min(MIN_KEEP, len(sorted_docs)):
        kept = sorted_docs[:min(MIN_KEEP, len(sorted_docs))]

    return kept


# ── Fuzzy verify ──────────────────────────────────────────────────────────────

def _fuzzy_verify(response_text: str, docs: List[Document]) -> List[Document]:
    """Drop obvious mismatches. Verifies on chunk snippet not full content."""
    if not docs:
        return []

    resp_n = _normalize(response_text)
    scored: List[Tuple[int, Document]] = []

    for doc in docs:
        snippet = _normalize((doc.page_content or "")[:VERIFY_CHUNK_WINDOW])
        score   = fuzz.token_set_ratio(resp_n, snippet)
        doc.metadata["fuzzy_score"] = score
        if score >= FUZZY_VERIFY_CUTOFF:
            scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _retriever(k: int) -> IDPAzureSearchRetriever:
    return IDPAzureSearchRetriever(
        index_name=INDEX_NAME,
        semantic_configuration_name=SEMANTIC_CONFIG,
        search_type="simple",
        k=k,
    )


def _normalize(s: str) -> str:
    s = re.sub(r"\[\d+\]", "", s)       # strip citation markers
    s = re.sub(r"[^\w\s]", " ", s)      # strip punctuation noise
    return re.sub(r"\s+", " ", s).strip().lower()


def _truncate(text: str, max_chars: int = 800) -> str:
    return re.sub(r"\s+", " ", text[:max_chars]).strip()


def _bm25_score(doc: Document) -> float:
    md = doc.metadata or {}
    for key in ("bm25_score", "@search.scores", "@search.score", "score", "search_score"):
        val = md.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return 0.0


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