import re
from typing import List, Dict, Tuple
from langchain_core.documents import Document
from impact_analysis.vector_client import IDPAzureSearchRetriever
from rapidfuzz import fuzz

INDEX_NAME = "idp_kg_data"
SEMANTIC_CONFIG = "default"

PER_CITATION_K_BM25 = 5
PER_CITATION_K_VEC  = 3   # optional
FINAL_K = 15
VERIFY_WINDOW = 800
FUZZY_CUTOFF = 40

CIT_RE = re.compile(r"\[(\d+)\]")

def _normalize(s: str) -> str:
    s = re.sub(r"\[\d+\]", "", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()

def split_by_citations(response: str) -> Dict[str, str]:
    """
    Build {citation_id: text_snippet} by collecting lines/sentences that contain [n].
    Works well for your markdown bullets where citations appear at line ends.
    """
    buckets: Dict[str, List[str]] = {}

    # split by lines first (bullets/headings)
    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue
        for cid in CIT_RE.findall(line):
            buckets.setdefault(cid, []).append(line)

    # join bucket text
    return {cid: " ".join(lines) for cid, lines in buckets.items()}

def _retriever(search_type: str, k: int) -> IDPAzureSearchRetriever:
    return IDPAzureSearchRetriever(
        index_name=INDEX_NAME,
        semantic_configuration_name=SEMANTIC_CONFIG,
        search_type=search_type,
        k=k,
    )

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

def extract_mentioned_docs(response: str, use_semantic: bool = True) -> List[Document]:
    """
    Citation-aware doc extraction:
    - For each citation bucket, query only that local snippet.
    - Retrieve small top-k, rerank by fuzzy similarity to the snippet.
    """
    buckets = split_by_citations(response)

    # If no citations exist, fallback to using only the Q line or first 300 chars
    if not buckets:
        q = response.strip().splitlines()[0][:300]
        buckets = {"0": q}

    all_docs: List[Document] = []

    for cid, snippet in buckets.items():
        bm25 = _retriever("simple", PER_CITATION_K_BM25).invoke(snippet)

        docs = list(bm25)
        if use_semantic:
            vec = _retriever("semantic", PER_CITATION_K_VEC).invoke(snippet)
            docs.extend(vec)

        docs = _dedupe(docs)

        # fuzzy rerank vs snippet (not entire response)
        sn = _normalize(snippet)
        scored: List[Tuple[int, Document]] = []
        for d in docs:
            dn = _normalize((d.page_content or "")[:VERIFY_WINDOW])
            score = fuzz.token_set_ratio(sn, dn)
            d.metadata["snippet_fuzzy_score"] = score
            d.metadata["citation_id"] = cid
            scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)

        # keep a couple best per citation, but only if not totally off
        kept = [d for s, d in scored if s >= FUZZY_CUTOFF][:3]
        if not kept:
            kept = [d for _, d in scored[:1]]  # at least 1 per citation

        all_docs.extend(kept)

    return _dedupe(all_docs)[:FINAL_K]