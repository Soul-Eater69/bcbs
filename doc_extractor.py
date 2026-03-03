import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from langchain_core.documents import Document
from impact_analysis.vector_client import IDPAzureSearchRetriever

# Optional: RapidFuzzy rerank (pip install rapidfuzzy)
try:
    from rapidfuzzy import fuzz
    _RAPIDFUZZY_OK = True
except Exception:
    _RAPIDFUZZY_OK = False


INDEX_NAME      = "idp_kg_data"
SEMANTIC_CONFIG = "default"
TOP_K           = 10

# BM25 candidate pool size per query (bigger = better recall, slower)
CANDIDATES_PER_QUERY = 25

# How many lexical queries to derive from the response
MAX_QUERIES = 6

# RapidFuzzy settings (only used if enabled)
USE_RAPIDFUZZY_RERANK = True
FUZZY_WEIGHT = 0.45      # blend weight for fuzzy score
BM25_WEIGHT  = 0.55      # blend weight for bm25 agg score
FUZZY_THRESHOLD = 65     # 0..100, drop candidates below this (optional)


def extract_mentioned_docs(response: str) -> List[Document]:
    """
    Best-effort attribution from only the final LLM response.

    1) Build a few lexical queries from response
    2) Run BM25-style keyword search (Azure AI Search "simple")
    3) Aggregate scores across queries
    4) Optional RapidFuzzy rerank over candidate texts (cheap, no sklearn)
    """
    clean = _clean_response(response)
    queries = _build_queries(clean, max_queries=MAX_QUERIES)

    candidates, bm25_scores = _bm25_aggregate(queries)

    if not candidates:
        return []

    # Optional: local rerank only across the small candidate set
    if USE_RAPIDFUZZY_RERANK and _RAPIDFUZZY_OK:
        candidates = _rapidfuzzy_rerank(clean, candidates, bm25_scores)

    # Sort by final score and return top-k
    candidates.sort(key=lambda d: bm25_scores[_doc_key(d)], reverse=True)
    out = candidates[:TOP_K]

    # Attach final score to metadata
    for d in out:
        d.metadata["attribution_score"] = float(bm25_scores[_doc_key(d)])

    return out


# -------------------------
# Helpers
# -------------------------

def _clean_response(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)   # remove citation markers like [12]
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_queries(text: str, max_queries: int = 6) -> List[str]:
    """
    Keep this minimal:
      - whole response (truncated)
      - a few high-signal phrases (quoted strings, Capitalized multi-word phrases, snake_case IDs)
    """
    queries: List[str] = []

    # Query 1: full response (bounded length)
    queries.append(_truncate(text, 800))

    quoted = re.findall(r'"([^"]{3,80})"', text)
    caps = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})\b", text)
    ids = [w for w in re.findall(r"\b[a-zA-Z_]{3,40}\b", text) if "_" in w]

    # prefer longer/more specific
    pool = []
    pool += sorted(set(quoted), key=len, reverse=True)
    pool += sorted(set(caps), key=len, reverse=True)
    pool += sorted(set(ids), key=len, reverse=True)

    for item in pool:
        q = item.strip()
        if len(q) >= 4:
            queries.append(q)
        if len(queries) >= max_queries:
            break

    # dedupe, preserve order
    seen = set()
    out = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out[:max_queries]


def _bm25_aggregate(queries: List[str]) -> Tuple[List[Document], Dict[str, float]]:
    """
    Runs Azure keyword search for each query and aggregates (normalized) scores per doc.
    """
    retriever = IDPAzureSearchRetriever(
        index_name=INDEX_NAME,
        search_type="simple",  # BM25-like keyword scoring in Azure Search
        semantic_configuration_name=SEMANTIC_CONFIG,
        k=CANDIDATES_PER_QUERY,
    )

    docs_by_key: Dict[str, Document] = {}
    agg_scores: Dict[str, float] = defaultdict(float)

    for q in queries:
        results = retriever.invoke(input=q) or []

        # Normalize within this query so long queries don't dominate
        raw = [_get_search_score(d) for d in results]
        max_s = max(raw) if raw else 0.0
        denom = max_s if max_s > 0 else 1.0

        for d in results:
            key = _doc_key(d)
            docs_by_key[key] = d
            agg_scores[key] += (_get_search_score(d) / denom)

    return list(docs_by_key.values()), agg_scores


def _rapidfuzzy_rerank(query_text: str, docs: List[Document], scores: Dict[str, float]) -> List[Document]:
    """
    Cheap local rerank:
      - compute fuzzy similarity between response text and each doc's representative text
      - blend it into the aggregated BM25 score
    """
    # Scale BM25 scores roughly into 0..1
    bm25_max = max(scores.values()) if scores else 1.0
    bm25_max = bm25_max if bm25_max > 0 else 1.0

    for d in docs:
        key = _doc_key(d)
        bm25_norm = float(scores.get(key, 0.0)) / bm25_max

        rep = _doc_rep_text(d)
        # RapidFuzzy returns 0..100
        fuzzy = float(fuzz.token_set_ratio(query_text, rep))

        # Optionally drop very weak fuzzy matches (can help precision)
        if fuzzy < FUZZY_THRESHOLD:
            fuzzy_norm = 0.0
        else:
            fuzzy_norm = fuzzy / 100.0

        final = (BM25_WEIGHT * bm25_norm) + (FUZZY_WEIGHT * fuzzy_norm)

        # Store final blended score back into scores dict
        scores[key] = final

    return docs


def _doc_rep_text(doc: Document) -> str:
    """
    What to fuzzy-match against. Prefer titles/headers if you have them, else content prefix.
    Tune keys to your schema.
    """
    md = doc.metadata or {}
    parts = []

    for k in ("title", "document_title", "source", "file_name", "entity_name"):
        v = md.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())

    if doc.page_content:
        parts.append(doc.page_content[:1500])  # keep it bounded

    rep = " | ".join(parts)
    rep = re.sub(r"\s+", " ", rep).strip()
    return rep


def _get_search_score(doc: Document) -> float:
    """
    Adjust this if your wrapper stores Azure score in a different metadata key.
    """
    md = doc.metadata or {}
    for k in ("@search.score", "search_score", "score"):
        v = md.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0


def _doc_key(doc: Document) -> str:
    md = doc.metadata or {}
    return (
        md.get("id")
        or md.get("chunk_id")
        or md.get("source")
        or (doc.page_content[:80] if doc.page_content else "unknown")
    )


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n]