from __future__ import annotations

import re
from typing import List, Set, Tuple, Dict, Optional

from rapidfuzz import fuzz
from langchain_core.documents import Document

from impact_analysis.vector_client import IDPAzureSearchRetriever


INDEX_NAME = "idp_kg_data"
SEMANTIC_CONFIG = "default"

# fuzzy matching
VOCAB_FUZZ_CUTOFF = 85       # 80–88 is typical; 85 is a good start
MAX_MATCHED_TERMS = 20       # cap to avoid too many filter queries

# retrieval
K_PER_ENTITY = 5             # top-k docs per entity
SEARCH_TYPE = "simple"       # BM25
ENTITY_FIELD = "entity_name" # change if your field name differs


# ---------------------------
# Vocabulary loading/cleaning
# ---------------------------

def load_vocab(file_path: str) -> List[str]:
    vocab: List[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().strip('"').strip("'")
            if word:
                vocab.append(word)
    return vocab


def clean_entity_names(vocab: List[str]) -> List[str]:
    """
    Input examples:
      "L1 Capability Claim Management"
      "L3 Capability Claim Communication Management"
      "Application Something Something"
    Output canonical names:
      "Claim Management"
      "Claim Communication Management"
      "Something Something" (if non-L1/L2/L3 prefix, drops first token)
    """
    entities: Set[str] = set()

    for entity in vocab:
        e = entity.strip()
        if not e:
            continue

        parts = e.split()
        if len(parts) < 2:
            continue

        if e.startswith("L1") or e.startswith("L2") or e.startswith("L3"):
            # e.g. L3 Capability Claim Communication Management -> drop first 2 tokens: L3 + Capability
            name = " ".join(parts[2:]) if len(parts) > 2 else ""
        else:
            # drop first token (e.g., "Application Foo Bar" -> "Foo Bar")
            name = " ".join(parts[1:])

        name = name.strip()
        if name:
            entities.add(name)

    # stable order helps reproducibility
    return sorted(entities)


# ---------------------------
# Fuzzy matching
# ---------------------------

def _normalize(s: str) -> str:
    s = re.sub(r"\[\d+\]", "", s)          # remove [1] markers
    s = re.sub(r"[^\w\s]", " ", s)         # remove punctuation
    return re.sub(r"\s+", " ", s).strip().lower()


def match_entities_rapidfuzz(
    response_text: str,
    entities: List[str],
    cutoff: int = VOCAB_FUZZ_CUTOFF,
    max_terms: int = MAX_MATCHED_TERMS,
) -> List[Tuple[str, int]]:
    """
    Returns [(entity_name, score)] sorted by score desc.
    Uses:
      - fast substring check first
      - then partial_ratio as fuzzy fallback
    """
    resp_norm = _normalize(response_text)

    hits: List[Tuple[str, int]] = []

    for ent in entities:
        ent_norm = _normalize(ent)

        # very short names cause tons of false positives
        if len(ent_norm) < 4:
            continue

        if ent_norm in resp_norm:
            score = 100
        else:
            # partial_ratio works well for "Claim Communication" inside longer sentences
            score = int(fuzz.partial_ratio(ent_norm, resp_norm))

        if score >= cutoff:
            hits.append((ent, score))

    hits.sort(key=lambda x: x[1], reverse=True)
    return hits[:max_terms]


# ---------------------------
# Entity-filtered retrieval
# ---------------------------

def _escape_odata_string(value: str) -> str:
    # OData string literals escape single quote by doubling it
    return value.replace("'", "''")


def fetch_docs_for_entities(
    matched_entities: List[Tuple[str, int]],
    k_per_entity: int = K_PER_ENTITY,
) -> List[Document]:
    """
    For each matched entity, query Azure with filter:
      entity_name eq '<entity>'
    Returns deduped docs.
    """
    out: List[Document] = []
    seen: set[str] = set()

    for ent, score in matched_entities:
        flt = f"{ENTITY_FIELD} eq '{_escape_odata_string(ent)}'"

        retriever = IDPAzureSearchRetriever(
            index_name=INDEX_NAME,
            semantic_configuration_name=SEMANTIC_CONFIG,
            search_type=SEARCH_TYPE,
            k=k_per_entity,
            # IMPORTANT: your retriever must accept this parameter. See notes below.
            filter=flt,
        )

        docs = retriever.invoke(ent) or []
        for d in docs:
            d.metadata["matched_entity"] = ent
            d.metadata["entity_match_score"] = score

            key = (
                str((d.metadata or {}).get("chunk_id"))
                or str((d.metadata or {}).get("id"))
                or str((d.metadata or {}).get("source"))
                or (d.page_content or "")[:200]
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(d)

    return out


# ---------------------------
# One function you call
# ---------------------------

def extract_docs_by_vocab_entities(
    response_text: str,
    vocab_path: str,
) -> List[Document]:
    vocab = load_vocab(vocab_path)
    entities = clean_entity_names(vocab)

    matched = match_entities_rapidfuzz(response_text, entities)
    return fetch_docs_for_entities(matched)


# ---------------------------
# Example run
# ---------------------------

if __name__ == "__main__":
    VOCAB_PATH = r"C:\Users\U588867\idp-impact-analysis\src\impact_analysis\utils\sightline_vocabulary 2.txt"

    response_text = """PASTE YOUR RESPONSE TEXT HERE"""

    matched_docs = extract_docs_by_vocab_entities(response_text, VOCAB_PATH)

    print("\n=== MATCHED ENTITIES ===")
    vocab = load_vocab(VOCAB_PATH)
    entities = clean_entity_names(vocab)
    for ent, score in match_entities_rapidfuzz(response_text, entities):
        print(f"{score:>3}  {ent}")

    print("\n=== DOCS ===")
    for i, d in enumerate(matched_docs, 1):
        md = d.metadata or {}
        print(f"\n[{i}] entity={md.get('matched_entity')} entity_score={md.get('entity_match_score')}")
        print("keys:", [k for k in md.keys()][:12])
        print("snippet:", (d.page_content or "")[:220].replace("\n", " "))