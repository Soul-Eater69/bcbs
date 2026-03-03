# mentioned_docs_hybrid.py
from __future__ import annotations

import re
from typing import List, Tuple, Optional, Dict

from langchain_core.documents import Document
from rapidfuzz import fuzz

from impact_analysis.vector_client import IDPAzureSearchRetriever


# ─────────────────────────────────────────────────────────────
# Goal:
# Given an LLM response, approximate which chunks/docs were "used/mentioned"
# WITHOUT logs, using only:
#   - 1 BM25 call (simple)
#   - 1 semantic/vector call (semantic)
# Then a lightweight verifier to drop Jira/user-story noise.
#
# Total Azure calls: 2 (fixed).
# ─────────────────────────────────────────────────────────────

INDEX_NAME = "idp_kg_data"
SEMANTIC_CONFIG = "default"

# candidates from each retriever (2 calls total)
K_BM25 = 40
K_VEC  = 40

# final output
FINAL_K = 12
MIN_KEEP = 5

# verifier
SNIPPET_CHARS = 1200
REL_CUTOFF = 0.78       # keep docs with score >= top * REL_CUTOFF
ABS_CUTOFF = 0.22       # or absolute minimum
MIN_TOKEN_LEN = 5       # informative tokens


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def extract_mentioned_docs_hybrid(
    response_text: str,
    final_k: int = FINAL_K,
) -> List[Document]:
    """
    1) Build an "anchor query" from the response (Q-line + bullets + key lines).
    2) Retrieve candidates with BM25 + Semantic (2 calls).
    3) Verify/rerank candidates using lexical overlap + fuzzy similarity.
    4) Threshold with min-keep.
    """
    anchor_query = build_anchor_query(response_text)

    candidates = retrieve_candidates(anchor_query)
    ranked = verify_and_rank(anchor_query, candidates)

    kept = threshold_keep(ranked, final_k=final_k)
    return kept


# ─────────────────────────────────────────────────────────────
# 1) Anchor query builder (prevents “query dilution”)
# ─────────────────────────────────────────────────────────────

Q_RE = re.compile(r"(?:^|\n)\s*Q:\s*(.+)", re.IGNORECASE)
CIT_RE = re.compile(r"\[\d+\]")

def build_anchor_query(response_text: str, max_anchors: int = 10, max_chars: int = 1200) -> str:
    """
    Pull high-signal lines from the response:
      - Q: line (if present)
      - bullet-ish lines
      - lines containing citations [n]
      - otherwise, first informative lines
    This keeps retrieval focused even in mixed corpora.
    """
    lines = [ln.strip() for ln in response_text.splitlines() if ln.strip()]

    anchors: List[str] = []

    # Q line (best)
    m = Q_RE.search(response_text)
    if m:
        anchors.append(m.group(1).strip())

    # lines with citations (often “evidence” lines)
    for ln in lines:
        if CIT_RE.search(ln):
            anchors.append(CIT_RE.sub("", ln).strip())

    # bullet-ish lines
    for ln in lines:
        if ln.startswith(("-", "•", "*")):
            anchors.append(CIT_RE.sub("", ln).strip())

    # fallback: take early informative lines
    if len(anchors) < max_anchors:
        for ln in lines[:30]:
            clean = CIT_RE.sub("", ln).strip()
            if len(clean) >= 25:
                anchors.append(clean)

    # dedupe anchors
    seen = set()
    uniq: List[str] = []
    for a in anchors:
        an = normalize(a)
        if not an or an in seen:
            continue
        seen.add(an)
        uniq.append(a)
        if len(uniq) >= max_anchors:
            break

    query = " ".join(uniq)
    query = re.sub(r"\s+", " ", query).strip()
    return query[:max_chars]


# ─────────────────────────────────────────────────────────────
# 2) Two-call candidate retrieval (BM25 + semantic)
# ─────────────────────────────────────────────────────────────

def retrieve_candidates(anchor_query: str) -> List[Document]:
    bm25 = IDPAzureSearchRetriever(
        index_name=INDEX_NAME,
        semantic_configuration_name=SEMANTIC_CONFIG,
        search_type="simple",
        k=K_BM25,
    ).invoke(anchor_query)

    sem = IDPAzureSearchRetriever(
        index_name=INDEX_NAME,
        semantic_configuration_name=SEMANTIC_CONFIG,
        search_type="semantic",
        k=K_VEC,
    ).invoke(anchor_query)

    return dedupe(list(bm25) + list(sem))


# ─────────────────────────────────────────────────────────────
# 3) Verifier / reranker
#    (drop “claims-related but wrong corpus” docs)
# ─────────────────────────────────────────────────────────────

def verify_and_rank(anchor_query: str, candidates: List[Document]) -> List[Tuple[float, Document]]:
    """
    Score each candidate with evidence-based signals:
      - overlap_score: fraction of informative query tokens present in doc snippet
      - fuzzy_score: token_set_ratio(query, doc_snippet)
    Combined score favors docs that *share concrete wording* with response anchors.
    """
    qn = normalize(anchor_query)
    q_tokens = informative_tokens(qn)

    ranked: List[Tuple[float, Document]] = []
    for d in candidates:
        snippet = (d.page_content or "")[:SNIPPET_CHARS]
        dn = normalize(snippet)

        # evidence overlap (0..1)
        overlap = overlap_score(q_tokens, informative_tokens(dn))

        # fuzzy (0..1)
        fz = fuzz.token_set_ratio(qn, dn) / 100.0

        # combine (tuned to reward “evidence overlap” more than semantic)
        score = 0.65 * overlap + 0.35 * fz

        d.metadata["anchor_overlap"] = overlap
        d.metadata["anchor_fuzzy"] = fz
        d.metadata["hybrid_verify_score"] = score
        d.metadata["anchor_query"] = anchor_query[:250]

        ranked.append((score, d))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked


def informative_tokens(text_norm: str) -> set[str]:
    toks = text_norm.split()
    # keep “informative” tokens only (length-based, domain-agnostic)
    return {t for t in toks if len(t) >= MIN_TOKEN_LEN}


def overlap_score(q_tokens: set[str], d_tokens: set[str]) -> float:
    if not q_tokens:
        return 0.0
    return len(q_tokens & d_tokens) / len(q_tokens)


# ─────────────────────────────────────────────────────────────
# 4) Thresholding (relative + absolute + min keep)
# ─────────────────────────────────────────────────────────────

def threshold_keep(ranked: List[Tuple[float, Document]], final_k: int) -> List[Document]:
    if not ranked:
        return []

    top = ranked[0][0]
    kept = [
        d for s, d in ranked
        if (s >= top * REL_CUTOFF) or (s >= ABS_CUTOFF)
    ]

    if len(kept) < min(MIN_KEEP, len(ranked)):
        kept = [d for _, d in ranked[:min(MIN_KEEP, len(ranked))]]

    return kept[:final_k]


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    s = re.sub(r"\[\d+\]", "", s)          # remove [1] style
    s = re.sub(r"[^\w\s]", " ", s)         # remove punctuation noise
    return re.sub(r"\s+", " ", s).strip().lower()


def doc_key(doc: Document) -> str:
    md = doc.metadata or {}
    return str(
        md.get("chunk_id")
        or md.get("id")
        or md.get("source")
        or md.get("node_id")
        or (doc.page_content or "")[:200]
    )


def dedupe(docs: List[Document]) -> List[Document]:
    seen = set()
    out: List[Document] = []
    for d in docs:
        k = doc_key(d)
        if k in seen:
            continue
        seen.add(k)
        out.append(d)
    return out


# ─────────────────────────────────────────────────────────────
# Optional runner (print results)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    response_text = """PASTE YOUR RESPONSE HERE"""
    docs = extract_mentioned_docs_hybrid(response_text)

    print("\nANCHOR QUERY:\n", build_anchor_query(response_text), "\n")
    print("=== RESULTS ===")
    for i, d in enumerate(docs, 1):
        md = d.metadata or {}
        print(f"\n[{i}] id={doc_key(d)}")
        print(f"score={md.get('hybrid_verify_score'):.4f} overlap={md.get('anchor_overlap'):.4f} fuzzy={md.get('anchor_fuzzy'):.4f}")
        print("snippet:", (d.page_content or "")[:220].replace("\n", " "))