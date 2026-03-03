import re
from difflib import SequenceMatcher
from langchain_core.documents import Document
from impact_analysis.vector_client import IDPAzureSearchRetriever

INDEX_NAME      = "idp_kg_data"
SEMANTIC_CONFIG = "default"
TOP_K           = 10
FUZZY_THRESHOLD = 0.85  # minimum similarity ratio for vocabulary term match


def extract_mentioned_docs(response: str) -> list[Document]:
    """Primary: semantic vector search — returns top-k chunks most relevant to the response."""
    retriever = IDPAzureSearchRetriever(
        index_name=INDEX_NAME,
        search_type="semantic",
        semantic_configuration_name=SEMANTIC_CONFIG,
        k=TOP_K,
    )
    return retriever.invoke(input=response)


def extract_mentioned_docs_by_entity(
    response: str,
    vocabulary: list[str],
) -> list[Document]:
    """Secondary: fuzzy match response against vocabulary, fetch doc per matched term."""
    matched = _fuzzy_match(response, vocabulary)
    return _fetch_by_term(matched)


def load_vocabulary(filepath: str) -> list[str]:
    """Load entity names from sightline_vocabulary.txt, one term per line."""
    with open(filepath, encoding="utf-8") as f:
        return [line.strip().strip('"') for line in f if line.strip()]


def _fuzzy_match(response: str, vocabulary: list[str]) -> list[str]:
    """Slide a window over the response and compute similarity ratio per vocabulary term."""
    clean = re.sub(r'\[\d+\]', '', response)  # strip citation markers before matching
    matched = []
    for term in vocabulary:
        step = max(1, len(term) // 4)
        best = max(
            SequenceMatcher(None, term.lower(), clean[i: i + len(term) + 15].lower()).ratio()
            for i in range(0, max(1, len(clean) - len(term) + 1), step)
        )
        if best >= FUZZY_THRESHOLD:
            matched.append(term)
    return matched


def _fetch_by_term(terms: list[str]) -> list[Document]:
    """Keyword search using the term name as query — exact name ranks its doc at position 1."""
    seen: set[str] = set()
    results: list[Document] = []
    for term in terms:
        retriever = IDPAzureSearchRetriever(
            index_name=INDEX_NAME,
            search_type="simple",
            semantic_configuration_name=SEMANTIC_CONFIG,
            k=1,
        )
        for doc in retriever.invoke(input=term):
            doc_id = doc.metadata.get("id") or doc.metadata.get("chunk_id") or doc.page_content[:60]
            if doc_id not in seen:
                seen.add(doc_id)
                results.append(doc)
    return results