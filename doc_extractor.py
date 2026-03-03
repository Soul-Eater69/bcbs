from __future__ import annotations

import re
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from rapidfuzz import fuzz

from impact_analysis.vector_client import IDPAzureSearchRetriever


# ─────────────────────────────────────────────────────────────
# PIPELINE:
# ① vocab match (substring → anchor-guarded fuzzy)
# ② BM25 per-term (precision) + BM25 on response (recall)
# ③ merge + dedupe
# ④ BM25 relative threshold (top * 0.70) + MIN_KEEP safety
# ⑤ fuzzy verify (loose) on chunk snippet
#    - fallback: if fuzzy drops everything, return narrowed BM25 results
# ─────────────────────────────────────────────────────────────

INDEX_NAME      = "idp_kg_data"
SEMANTIC_CONFIG = "default"

BM25_K_RESPONSE     = 25
BM25_K_PER_TERM     = 3
FINAL_K             = 10

BM25_REL_CUTOFF     = 0.70
MIN_KEEP            = 5

MAX_MATCHED_TERMS   = 20    # cap BM25 calls from vocab lane

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

    # Fallback: if fuzzy verify drops everything, return BM25-narrowed set
    if not verified:
        return narrowed[:final_k]

    return verified[:final_k]


def load_vocabulary(filepath: str) -> List[str]:
    """Load entity names from sightline_vocabulary.txt, one term per line."""
    with open(filepath, encoding="utf-8") as f:
        return [line.strip().strip('"') for line in f if line.strip()]


# ── Vocab matching ────────────────────────────────────────────────────────────

def _anchor_tokens(term_norm: str) -> List[str]:
    """Words of length >= 4 from the term — used as cheap overlap guard before fuzzy."""
    return [w for w in term_norm.split() if len(w) >= 4]


def _match_vocab_terms(response_text: str, vocabulary: List[str]) -> List[str]:
    """
    Substring match first (fast, high precision).
    Fuzzy only runs if: term was missed AND passes an overlap guard.
    Capped at MAX_MATCHED_TERMS to bound BM25 calls.
    """
    resp_norm = _normalize(response_text)
    matched: List[str] = []
    seen: set[str] = set()

    for term in vocabulary:
        if len(matched) >= MAX_MATCHED_TERMS:
            break

        t = term.strip()
        if not t:
            continue

        t_norm = _normalize(t)
        if not t_norm:
            continue

        # 1) Substring match
        if t_norm in resp_norm:
            if t_norm not in seen:
                seen.add(t_norm)
                matched.append(t)
            continue

        # 2) Overlap guard before fuzzy:
        anchors = _anchor_tokens(t_norm)
        tokens = t_norm.split()
        should_fuzzy = (anchors and any(a in resp_norm for a in anchors)) or (not anchors and any(tok in resp_norm for tok in tokens))

        # 3) Fuzzy fallback (correct direction: short term vs long response)
        if should_fuzzy and fuzz.partial_ratio(t_norm, resp_norm) >= VOCAB_FUZZ_CUTOFF:
            if t_norm not in seen:
                seen.add(t_norm)
                matched.append(t)

    return matched


# ── BM25 ──────────────────────────────────────────────────────────────────────

def _bm25_per_term(terms: List[str]) -> List[Document]:
    """One BM25 query per vocab term — canonical name ranks its doc at position 1."""
    retriever = _retriever(BM25_K_PER_TERM)
    docs: List[Document] = []
    for term in terms:
        docs.extend(retriever.invoke(term))
    return docs


def _bm25_full_response(response_text: str, matched_terms: List[str]) -> List[Document]:
    """
    BM25 on response — uses first 800 chars,
    and appends matched terms that appear beyond the window.
    """
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

    # Score unavailable -> skip thresholding
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
    s = re.sub(r"\[\d+\]", "", s)          # strip citation markers like [1]
    s = re.sub(r"[^\w\s]", " ", s)         # strip punctuation noise
    return re.sub(r"\s+", " ", s).strip().lower()


def _truncate(text: str, max_chars: int = 800) -> str:
    return re.sub(r"\s+", " ", text[:max_chars]).strip()


def _bm25_score(doc: Document) -> float:
    """
    Robust score getter:
    - prefer normalized 'bm25_score' (if your converter sets it)
    - else fallback to raw keys from backend
    """
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



VOCAB_PATH = "sightline_vocabulary.txt"  # adjust path if needed

if __name__ == "__main__":
    vocab = load_vocabulary(VOCAB_PATH)

    response_text = """
    Q: Describe Claim Management Capability
    Overview: Claim Management is the capability for receiving and resolving payment requests...
    Key Components: Claim Acquisition, Claim Data Management, Claim Payment...
    """

    docs = extract_mentioned_docs(response_text, vocabulary=vocab, final_k=10)

    print("\n=== RESULTS ===")
    for i, d in enumerate(docs, 1):
        md = d.metadata or {}
        print(f"\n[{i}] score={md.get('bm25_score') or md.get('@search.scores') or md.get('@search.score')}")
        print(f"source={md.get('source') or md.get('id') or md.get('chunk_id')}")
        print(f"fuzzy={md.get('fuzzy_score')}")
        print("snippet:", (d.page_content or "")[:250].replace("\n", " "))