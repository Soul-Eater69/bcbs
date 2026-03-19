from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Optional, List, Dict, Any


# ─── Request / Response ──────────────────────────────────────────────

@dataclass
class IDPAISearchRequest:
    """Maps 1:1 to the backend API schema."""

    index_name: str = "your-index-name"
    search_text: str = "*"
    search_type: str = "simple"                         # "simple" | "semantic"
    hybrid_mode: Optional[str] = None                   # "rrf" | None
    vector_weight: float = 0.5
    text_weight: float = 0.5
    filter_expression: Optional[str] = None
    pre_filters: Optional[Dict] = None
    post_filters: Optional[Dict] = None
    vector_filter_mode: str = "preFilter"
    vector_queries: Optional[List[Dict]] = None         # raw vectors if pre-computed
    enable_vector_search: bool = False
    vector_fields: Optional[List[str]] = field(default_factory=lambda: ["content_vector"])
    embedding_params: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            "api_version": "2024-06-01",
            "encoding_format": "float",
            "dimensions": 0,                            # set from config
            "model": "string",                          # set from config
        }
    )
    k_nearest_neighbors: int = 50
    top: int = 5
    skip: int = 0
    order_by: Optional[List[str]] = None
    select: Optional[List[str]] = None
    semantic_configuration_name: Optional[str] = None   # set from config when needed
    query_caption: Optional[str] = None
    query_answer: Optional[str] = None
    query_answer_count: int = 3
    scoring_profile: Optional[str] = None
    scoring_parameters: Optional[List[str]] = None
    facets: Optional[List[str]] = None
    highlight_fields: Optional[List[str]] = None
    highlight_pre_tag: str = "<em>"
    highlight_post_tag: str = "</em>"
    include_total_count: bool = True
    minimum_coverage: int = 0
    session_id: Optional[str] = None


# ─── Search methods ──────────────────────────────────────────────────

class AzureSearchClient:

    def search(self, query: str, request: Optional[IDPAISearchRequest] = None) -> Any:
        if not self._idp_base_url:
            raise ValueError("AISEARCH_BASE_URL is not set.")

        if request is None:
            request = IDPAISearchRequest()

        # Only override search_text if it's still the default wildcard.
        # This lets callers explicitly set search_text="*" for vector-only.
        if request.search_text == "*" and request.enable_vector_search:
            # Vector-only: keep "*" for BM25, backend uses query for embedding
            # The query is passed separately or embedded via embedding_params
            payload = request
        else:
            payload = replace(request, search_text=query)

        # ... rest of HTTP call
        pass

    # ─────────────────────────────────────────────────────────────
    # 1. BM25 keyword-only
    # ─────────────────────────────────────────────────────────────
    #
    #   search_text = query   →  BM25 scores against inverted index
    #   enable_vector_search  =  False
    #   hybrid_mode           =  None (no fusion needed — single result set)
    #
    #   Response:
    #     @search.score  =  BM25 score (0 to unbounded, typically 1–30)
    #
    def search_bm25(
        self,
        query: str,
        top_k: int = 5,
        filter_expression: Optional[str] = None,
    ) -> Any:
        request = IDPAISearchRequest(
            search_type="simple",
            search_text=query,              # ← drives BM25 matching
            top=top_k,
            filter_expression=filter_expression,
            enable_vector_search=False,     # ← no vector search
            hybrid_mode=None,               # ← no fusion
            text_weight=1.0,                # irrelevant without hybrid, but explicit
            vector_weight=0.0,
        )
        return self.search(query, request=request)

    # ─────────────────────────────────────────────────────────────
    # 2. Vector-only (semantic similarity)
    # ─────────────────────────────────────────────────────────────
    #
    #   search_text = query   →  backend uses this to generate embedding
    #   enable_vector_search  =  True
    #   hybrid_mode           =  "rrf" (needed so weights take effect)
    #   text_weight           =  0.0   ← zeroes out BM25 contribution
    #   vector_weight         =  1.0   ← only vector results matter
    #
    #   Why not hybrid_mode=None?
    #   Because with None, the backend may ignore weights entirely.
    #   Using RRF with text_weight=0 is the cleanest way to get
    #   vector-only ranking while still passing search_text for
    #   the backend to vectorize.
    #
    #   Response:
    #     @search.score  =  RRF score, but dominated entirely by vector rank
    #                       (effectively cosine similarity ordering)
    #
    def search_vector(
        self,
        query: str,
        top_k: int = 5,
        filter_expression: Optional[str] = None,
    ) -> Any:
        request = IDPAISearchRequest(
            search_type="simple",           # no semantic re-rank
            search_text=query,              # ← backend vectorizes this
            top=top_k,
            filter_expression=filter_expression,
            enable_vector_search=True,
            hybrid_mode="rrf",              # ← use RRF so weights apply
            text_weight=0.0,                # ← BM25 contributes nothing
            vector_weight=1.0,              # ← vector results only
        )
        return self.search(query, request=request)

    # ─────────────────────────────────────────────────────────────
    # 3. Hybrid + semantic re-ranking (the full pipeline)
    # ─────────────────────────────────────────────────────────────
    #
    #   search_text = query   →  BM25 matching + embedding generation
    #   enable_vector_search  =  True
    #   hybrid_mode           =  "rrf"
    #   search_type           =  "semantic"  (adds L2 re-rank)
    #   text_weight / vector_weight  =  tunable (0.5/0.5 is balanced)
    #
    #   Pipeline:
    #     BM25 (L1) ─┐
    #                 ├─→ RRF fusion ──→ Semantic re-rank (L2) ──→ results
    #     HNSW (L1) ─┘
    #
    #   Response:
    #     @search.score         =  RRF score (0.001–0.06 range)
    #     @search.rerankerScore =  Semantic score (0.0–4.0)
    #     @search.captions      =  extracted passages
    #     @search.answers       =  direct answer (if query is a question)
    #
    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        filter_expression: Optional[str] = None,
        text_weight: float = 0.5,
        vector_weight: float = 0.5,
    ) -> Any:
        request = IDPAISearchRequest(
            search_type="semantic",
            search_text=query,
            top=top_k,
            filter_expression=filter_expression,
            enable_vector_search=True,
            hybrid_mode="rrf",
            text_weight=text_weight,
            vector_weight=vector_weight,
            semantic_configuration_name="your-semantic-config",  # from config
            query_caption="extractive|highlight-true",
            query_answer="extractive",
            query_answer_count=3,
            k_nearest_neighbors=50,         # feed semantic ranker plenty
        )
        return self.search(query, request=request)