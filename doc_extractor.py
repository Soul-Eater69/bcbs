"""
extract_mentioned_docs.py

Given an LLM response, finds which source documents from the corpus
were referenced/described in that response.

Three techniques:
  1. Vector Search  — semantic similarity (search_type="semantic")
  2. Entity Search  — metadata filter by entity_name field
  3. Hybrid Search  — vector + BM25 (search_type="hybrid" if supported,
                      or runs both and merges)

Adapts to your existing IDPAzureSearchRetriever interface.
"""

from enum import Enum
from langchain_core.documents import Document
from impact_analysis.vector_client import IDPAzureSearchRetriever


# ──────────────────────────────────────────────
# Config / constants — update to match your setup
# ──────────────────────────────────────────────

INDEX_NAME = "idp-impact-analysis-ddl-index"
SEMANTIC_CONFIG = "default-semantic-config"


# ──────────────────────────────────────────────
# Search modes
# ──────────────────────────────────────────────

class SearchMode(Enum):
    VECTOR = "vector"    # pure semantic vector search
    ENTITY = "entity"    # metadata filter by entity_name
    HYBRID = "hybrid"    # vector + BM25 combined


# ──────────────────────────────────────────────
# Main function
# ──────────────────────────────────────────────

def extract_mentioned_docs(
    response: str,
    mode: SearchMode = SearchMode.VECTOR,
    index_name: str = INDEX_NAME,
    semantic_config: str = SEMANTIC_CONFIG,
) -> list[Document]:
    """
    Given an LLM response text, return the source docs it references.

    Args:
        response:      The LLM generated response text
        mode:          Which search technique to use
        index_name:    Azure Search index name
        semantic_config: Semantic configuration name

    Returns:
        List of LangChain Document objects found in the corpus
    """
    if mode == SearchMode.VECTOR:
        return _vector_search(response, index_name, semantic_config)

    elif mode == SearchMode.ENTITY:
        entity_names = _extract_entity_names(response)
        return _entity_search(entity_names, index_name, semantic_config)

    elif mode == SearchMode.HYBRID:
        return _hybrid_search(response, index_name, semantic_config)

    else:
        raise ValueError(f"Unknown mode: {mode}")


# ──────────────────────────────────────────────
# Technique 1: Vector / Semantic Search
# ──────────────────────────────────────────────

def _vector_search(
    response: str,
    index_name: str,
    semantic_config: str,
) -> list[Document]:
    """
    Embeds the response text, searches by semantic similarity.

    Good for: LLM responses that paraphrase/describe without naming entities.
    """
    retriever = IDPAzureSearchRetriever(
        index_name=index_name,
        search_type="semantic",                      # uses your existing search_type
        semantic_configuration_name=semantic_config,
    )

    docs = retriever.invoke(input=response)
    return docs


# ──────────────────────────────────────────────
# Technique 2: Entity Name Filter
# ──────────────────────────────────────────────

def _entity_search(
    entity_names: list[str],
    index_name: str,
    semantic_config: str,
) -> list[Document]:
    """
    For each extracted entity name, fetches the exact chunk
    using a metadata filter: entity_name eq '<name>'

    Good for: responses that explicitly name known entities
              e.g. "Claim Adjudication", "Blue Exchange"
    """
    docs = []
    seen_ids = set()

    for name in entity_names:
        # Pass the entity name as a filter query
        # IDPAzureSearchRetriever wraps Azure Search — check if it exposes
        # a filters param, otherwise pass it as the query with search_type="simple"
        retriever = IDPAzureSearchRetriever(
            index_name=index_name,
            search_type="simple",
            semantic_configuration_name=semantic_config,
            filters=f"entity_name eq '{name}'",     # metadata filter
        )

        results = retriever.invoke(input="*")        # fetch all matching this entity

        for doc in results:
            doc_id = doc.metadata.get("id") or doc.metadata.get("chunk_id", name)
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                docs.append(doc)

    return docs


# ──────────────────────────────────────────────
# Technique 3: Hybrid (Vector + BM25)
# ──────────────────────────────────────────────

def _hybrid_search(
    response: str,
    index_name: str,
    semantic_config: str,
) -> list[Document]:
    """
    Runs vector search AND keyword (BM25) search, merged by RRF.

    Good for: best of both worlds — semantic meaning + exact term matching.
    Two approaches depending on what IDPAzureSearchRetriever supports:
      A) Single call with search_type="hybrid" (if supported)
      B) Two separate calls then merge by doc id (fallback)
    """

    # ── Approach A: if your retriever supports "hybrid" search_type ──
    try:
        retriever = IDPAzureSearchRetriever(
            index_name=index_name,
            search_type="hybrid",
            semantic_configuration_name=semantic_config,
        )
        return retriever.invoke(input=response)

    except Exception:
        pass  # fall through to Approach B

    # ── Approach B: run both separately and merge ──
    vector_docs  = _vector_search(response, index_name, semantic_config)
    keyword_docs = _keyword_search(response, index_name, semantic_config)

    return _merge_by_rrf(vector_docs, keyword_docs)


def _keyword_search(
    response: str,
    index_name: str,
    semantic_config: str,
) -> list[Document]:
    """BM25 keyword search — passes raw text, no embedding."""
    retriever = IDPAzureSearchRetriever(
        index_name=index_name,
        search_type="simple",                        # BM25/keyword
        semantic_configuration_name=semantic_config,
    )
    return retriever.invoke(input=response)


def _merge_by_rrf(
    vector_docs: list[Document],
    keyword_docs: list[Document],
    k: int = 60,
) -> list[Document]:
    """
    Reciprocal Rank Fusion — merges two ranked lists by position, not score.

    score = 1/(rank_in_vector + k) + 1/(rank_in_keyword + k)
    Higher score = appeared near the top in both lists.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    def get_id(doc: Document) -> str:
        return doc.metadata.get("id") or doc.metadata.get("chunk_id") or doc.page_content[:50]

    for rank, doc in enumerate(vector_docs):
        doc_id = get_id(doc)
        scores[doc_id]  = scores.get(doc_id, 0) + 1 / (rank + 1 + k)
        doc_map[doc_id] = doc

    for rank, doc in enumerate(keyword_docs):
        doc_id = get_id(doc)
        scores[doc_id]  = scores.get(doc_id, 0) + 1 / (rank + 1 + k)
        doc_map[doc_id] = doc

    # Sort by RRF score descending
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_ids]


# ──────────────────────────────────────────────
# Entity extraction helper
# ──────────────────────────────────────────────

def _extract_entity_names(response: str) -> list[str]:
    """
    Extracts named entities from the response text.

    Uses simple line parsing of citation markers like [1], [2] etc.
    and named terms. Replace with fuzzy vocabulary matching if available.

    Example: "Claim Adjudication manages liability [1]"
             → ["Claim Adjudication"]
    """
    import re

    lines = response.splitlines()
    entity_names = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Strip citation markers like [1], [2][3]
        clean = re.sub(r'\[\d+\]', '', line).strip()

        # Heuristic: capitalised multi-word phrases are likely entity names
        # e.g. "Claim Adjudication", "Blue Exchange"
        matches = re.findall(r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)+)\b', clean)
        entity_names.extend(matches)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for name in entity_names:
        if name not in seen:
            seen.add(name)
            unique.append(name)

    return unique


# ──────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────

if __name__ == "__main__":

    llm_response = """
    Claim Management is the capability for receiving and resolving payment requests [1].
    Claim Adjudication manages resolution of financial liability according to member benefits [1].
    Blue Exchange supports Claim Acquisition Management and Claim Communication Management [6].
    """

    # Technique 1: Vector / semantic search
    vector_docs = extract_mentioned_docs(llm_response, mode=SearchMode.VECTOR)
    print("Vector docs:", [d.metadata.get("entity_name") for d in vector_docs])

    # Technique 2: Entity name metadata filter
    entity_docs = extract_mentioned_docs(llm_response, mode=SearchMode.ENTITY)
    print("Entity docs:", [d.metadata.get("entity_name") for d in entity_docs])

    # Technique 3: Hybrid vector + BM25
    hybrid_docs = extract_mentioned_docs(llm_response, mode=SearchMode.HYBRID)
    print("Hybrid docs:", [d.metadata.get("entity_name") for d in hybrid_docs])