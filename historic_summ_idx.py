# pip install -U langchain-community faiss-cpu
# plus your embedding package/provider

import os
import re
import json
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS



class My1536Embeddings(Embeddings):
    """
    LangChain wrapper over your existing embedding service.
    Must return vectors of length 1536.
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = embed_texts_1536(texts)   # your existing function
        return [list(map(float, row)) for row in vectors]

    def embed_query(self, text: str) -> List[float]:
        vector = embed_texts_1536([text])[0]   # your existing function
        return list(map(float, vector))
    
def embed_texts_1536(texts: List[str]):
    """
    Plug in your real embedding call here.
    Return shape [N, 1536].
    """
    raise NotImplementedError

def make_summary_document(summary_doc: Dict[str, Any]) -> Document:
    retrieval_text = build_summary_retrieval_text(summary_doc)

    metadata = {
        "doc_id": summary_doc.get("doc_id", ""),
        "ticket_id": summary_doc.get("ticket_id", ""),
        "title": summary_doc.get("title", ""),
        "short_summary": summary_doc.get("short_summary", ""),
        "business_goal": summary_doc.get("business_goal", ""),
        "actors": summary_doc.get("actors", []),
        "direct_functions": summary_doc.get("direct_functions", []),
        "implied_functions": summary_doc.get("implied_functions", []),
        "change_types": summary_doc.get("change_types", []),
        "domain_tags": summary_doc.get("domain_tags", []),
        "evidence_sentences": summary_doc.get("evidence_sentences", []),
        "value_stream_labels": summary_doc.get("value_stream_labels", []),
        "doc_type": "ticket_summary",
    }

    return Document(page_content=retrieval_text, metadata=metadata)


def build_langchain_faiss_summary_index(
    summary_docs: List[Dict[str, Any]],
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)

    documents = [make_summary_document(doc) for doc in summary_docs]
    embeddings = My1536Embeddings()

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(out_dir)

    # optional human-readable backup
    with open(os.path.join(out_dir, "summary_docs.json"), "w", encoding="utf-8") as f:
        json.dump(summary_docs, f, ensure_ascii=False, indent=2)

    return vectorstore

def load_langchain_faiss_summary_index(index_dir: str):
    embeddings = My1536Embeddings()
    return FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def search_langchain_faiss_summary_index(
    query_text: str,
    index_dir: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    vectorstore = load_langchain_faiss_summary_index(index_dir)

    results = vectorstore.similarity_search_with_score(query_text, k=top_k)

    out = []
    for rank, (doc, score) in enumerate(results, start=1):
        out.append({
            "rank": rank,
            "score": float(score),
            "ticket_id": doc.metadata.get("ticket_id"),
            "title": doc.metadata.get("title"),
            "short_summary": doc.metadata.get("short_summary"),
            "actors": doc.metadata.get("actors", []),
            "direct_functions": doc.metadata.get("direct_functions", []),
            "implied_functions": doc.metadata.get("implied_functions", []),
            "value_stream_labels": doc.metadata.get("value_stream_labels", []),
            "retrieval_text": doc.page_content,
        })
    return out


cleaned_text = clean_ppt_text(ppt_text)

ppt_summary = generate_ppt_semantic_summary(cleaned_text)
query_text = build_summary_retrieval_text(ppt_summary)

hits = search_langchain_faiss_summary_index(
    query_text=query_text,
    index_dir="local_ticket_summary_faiss",
    top_k=5,
)

for h in hits:
    print(h["rank"], h["score"], h["ticket_id"], h["title"])