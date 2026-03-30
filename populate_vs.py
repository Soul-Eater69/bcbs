from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import sys
from datetime import datetime
from typing import Dict, List, Tuple

from src.clients.azure_direct_client import AzureDirectSearchClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BASE_DIR = pathlib.Path(__file__).parent / "ticket_chunks"


def _norm(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def resolve_vs_name_to_kg(name: str, client: AzureDirectSearchClient) -> Tuple[str, str]:
    """
    Resolve a Jira-returned value stream name to canonical KG ValueStream id/name.
    Returns: (entity_id, canonical_name)
    """
    query = (name or "").strip()
    if not query:
        return "", ""

    vs_filter = "node_type eq 'ValueStream'"

    matches = []
    try:
        matches = client.search_hybrid(
            query,
            top_k=5,
            filter_expression=vs_filter,
            use_semantic_rerank=False,
        )
    except Exception:
        try:
            matches = client.search_vector(query, top_k=5, filter_expression=vs_filter)
        except Exception:
            try:
                matches = client.search_bm25(query, top_k=5, filter_expression=vs_filter)
            except Exception:
                matches = []

    if not matches:
        return "", query

    # 1) exact normalized name match first
    qn = _norm(query)
    exact = []
    for m in matches:
        ename = str(m.get("entity_name") or "").strip()
        if _norm(ename) == qn:
            exact.append(m)

    pool = exact if exact else matches
    best = max(
        pool,
        key=lambda m: float(m.get("@search.reranker_score") or m.get("@search.score") or 0.0),
    )

    return (
        str(best.get("entity_id") or "").strip(),
        str(best.get("entity_name") or query).strip(),
    )


async def populate_single_ticket(ticket_id: str) -> dict:
    """Fetch VS from JIRA and return the updated map dict."""
    from core.jira import get_value_streams_for_idea_card

    logging.info("Fetching value streams for %s ...", ticket_id)
    result = await get_value_streams_for_idea_card(ticket_id)

    jira_vs_names: List[str] = result.get("value_streams", [])
    themes: List[dict] = result.get("themes", [])
    title = result.get("title", ticket_id)
    found = result.get("found", False)

    kg_client = AzureDirectSearchClient(index_name="idp_kg_data_test")

    resolved_vs_ids: List[str] = []
    resolved_vs_names: List[str] = []
    jira_theme_ids: List[str] = []
    jira_theme_statuses: List[str] = []

    # Prefer Jira-provided VS names; resolve each name into canonical KG ID
    for idx, jira_name in enumerate(jira_vs_names):
        canon_id, canon_name = resolve_vs_name_to_kg(jira_name, kg_client)
        resolved_vs_ids.append(canon_id)
        resolved_vs_names.append(canon_name or jira_name)

        if idx < len(themes):
            jira_theme_ids.append(str(themes[idx].get("key") or ""))
            jira_theme_statuses.append(str(themes[idx].get("status") or ""))
        else:
            jira_theme_ids.append("")
            jira_theme_statuses.append("")

    return {
        "id": ticket_id,
        "ticketId": ticket_id,
        "project": ticket_id.split("-")[0] if "-" in ticket_id else "",
        "title": f"{ticket_id}: {title}",
        # Canonical KG ids/names
        "valueStreamIds": resolved_vs_ids,
        "valueStreamNames": resolved_vs_names,
        "valueStreamStatuses": jira_theme_statuses,
        # Keep Jira theme ids separately for traceability
        "jiraThemeIds": jira_theme_ids,
        "impactedProductIds": [],
        "impactedProductNames": [],
        "impactedItProductIds": [],
        "impactedItProductNames": [],
        "labelSource": "jira issuelinks + kg id resolution",
        "updatedDate": datetime.now().isoformat(),
        "_found_in_jira": found,
        "_themes_count": len(themes),
    }
