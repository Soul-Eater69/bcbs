"""
entity_search.py

Secondary strategy: fuzzy match response text against sightline_vocabulary.txt
then fetch exact documents using entity_name filter.
"""

import re
from difflib import SequenceMatcher
from langchain_core.documents import Document
from impact_analysis.vector_client import IDPAzureSearchRetriever


INDEX_NAME      = "idp_kg_data"
SEMANTIC_CONFIG = "default"
FUZZY_THRESHOLD = 0.85          # tune this — higher = stricter match


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load vocabulary
# ─────────────────────────────────────────────────────────────────────────────

def load_vocabulary(filepath: str) -> list[str]:
    """
    Load entity names from sightline_vocabulary.txt

    File format (each line):
        "Application Blue Exchange"
        "Application Claim Management"
        "Application HIPAA 27x Availity"

    Returns:
        ["Application Blue Exchange", "Application Claim Management", ...]
    """
    terms = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            term = line.strip().strip('"').strip("'")
            if term:
                terms.append(term)
    return terms


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Fuzzy match response against vocabulary
# ─────────────────────────────────────────────────────────────────────────────

def match_vocabulary_terms(
    response: str,
    vocabulary: list[str],
    threshold: float = FUZZY_THRESHOLD,
) -> list[str]:
    """
    Find vocabulary terms that appear (approximately) in the response.

    Why fuzzy and not exact?
      LLM may write "Blue Exchange" but vocabulary has "Application Blue Exchange"
      LLM may write "HIPAA 27x"    but vocabulary has "Application HIPAA 27x Availity"
      Fuzzy match handles abbreviations, partial names, slight variations.

    How it works:
      For each vocabulary term, slide a window over the response
      and compute similarity ratio. If ratio >= threshold → matched.

    Example:
      vocabulary term : "Application Blue Exchange"
      response snippet: "Blue Exchange supports Claim Acquisition"
      similarity      : 0.88  ✅  (above threshold)

    Args:
        response:    LLM response text
        vocabulary:  List of canonical names from sightline_vocabulary.txt
        threshold:   Minimum similarity ratio (0.0–1.0)

    Returns:
        List of matched canonical vocabulary terms
    """
    # Strip citation markers [1][2] so they don't interfere
    clean = re.sub(r'\[\d+\]', '', response)

    matched = []

    for term in vocabulary:
        term_len   = len(term)
        best_ratio = 0.0

        # Slide a window of term length across the response
        step = max(1, term_len // 4)
        for i in range(0, max(1, len(clean) - term_len + 1), step):
            window = clean[i: i + term_len + 15]   # slight overrun for partial names
            ratio  = SequenceMatcher(
                None,
                term.lower(),
                window.lower()
            ).ratio()
            best_ratio = max(best_ratio, ratio)

        if best_ratio >= threshold:
            matched.append(term)

    return matched


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Fetch documents by entity_name filter
# ─────────────────────────────────────────────────────────────────────────────

def fetch_docs_by_entity(matched_terms: list[str]) -> list[Document]:
    """
    For each matched vocabulary term, fetch its document by passing
    the term name directly as the search query.

    No filter needed — the term name itself is precise enough to
    retrieve the right document at rank 1.

    Example:
        term  : "Application Blue Exchange"
        query : "Application Blue Exchange"
        → retriever returns Blue Exchange doc at rank 1 ✅
    """
    seen_ids: set[str] = set()
    results:  list[Document] = []

    for term in matched_terms:
        retriever = IDPAzureSearchRetriever(
            index_name=INDEX_NAME,
            search_type="simple",           # keyword search — exact term match
            semantic_configuration_name=SEMANTIC_CONFIG,
            k=1,                            # top-1 is enough, term is exact
        )

        docs = retriever.invoke(input=term)  # pass term name as query

        for doc in docs:
            doc_id = (
                doc.metadata.get("id")
                or doc.metadata.get("chunk_id")
                or doc.page_content[:60]
            )
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                results.append(doc)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Combined entity search
# ─────────────────────────────────────────────────────────────────────────────

def entity_search(response: str, vocabulary: list[str]) -> list[Document]:
    """
    Full entity pipeline:
      response + vocabulary → fuzzy match → entity filter → List[Document]
    """
    matched_terms = match_vocabulary_terms(response, vocabulary)
    print(f"Matched vocabulary terms: {matched_terms}")   # helpful for debugging
    return fetch_docs_by_entity(matched_terms)


# ─────────────────────────────────────────────────────────────────────────────
# Example
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    vocab = load_vocabulary("sightline_vocabulary.txt")

    response = """
    Blue Exchange supports Claim Acquisition Management and Claim Data
    Management [6]. HIPAA 27x Availity also supports Claim Communication
    Management [6].
    """

    docs = entity_search(response, vocab)

    print(f"\nFound {len(docs)} docs via entity search:")
    for d in docs:
        print(f"  entity={d.metadata.get('entity_name')}  id={d.metadata.get('id')}")