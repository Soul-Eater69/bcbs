def _limit_historical_chunks(chunks: List[dict], limit: int) -> List[dict]:
    if not chunks:
        return []

    ranked = sorted(
        chunks,
        key=lambda chunk: (_chunk_score(chunk), len(_chunk_text(chunk).split())),
        reverse=True,
    )

    per_ticket_cap = 2
    per_ticket: Dict[str, int] = {}
    selected: List[dict] = []
    overflow: List[dict] = []

    for chunk in ranked:
        ticket_id = _chunk_ticket_id(chunk)
        if ticket_id:
            count = per_ticket.get(ticket_id, 0)
            if count >= per_ticket_cap:
                overflow.append(chunk)
                continue
            per_ticket[ticket_id] = count + 1

        selected.append(chunk)
        if len(selected) >= limit:
            return selected

    for chunk in overflow:
        selected.append(chunk)
        if len(selected) >= limit:
            break

    return selected


def _compact_historical_evidence(chunks: List[dict], limit: int = 6) -> List[dict]:
    compact: List[dict] = []

    for chunk in _limit_historical_chunks(chunks, limit):
        compact.append(
            {
                "ticket_id": _chunk_ticket_id(chunk),
                "title": _short(str(chunk.get("title") or ""), 120),
                "score": round(_chunk_score(chunk), 4),
                "snippet": _short(_chunk_text(chunk), 280),
                "provenance": _short(_chunk_provenance_text(chunk), 100),
            }
        )

    return compact


def _compact_vs_support(vs_support: List[dict], limit: int = 8) -> List[dict]:
    rows = sorted(
        vs_support,
        key=lambda row: (
            int(row.get("support_count") or 0),
            float(row.get("best_support_score") or row.get("confidence") or row.get("score") or 0.0),
        ),
        reverse=True,
    )

    compact: List[dict] = []
    for row in _dedupe_by_name(rows)[:limit]:
        compact.append(
            {
                "entity_id": row.get("entity_id") or "",
                "entity_name": row.get("entity_name") or "",
                "support_count": int(row.get("support_count") or 0),
                "best_support_score": round(
                    float(row.get("best_support_score") or row.get("confidence") or row.get("score") or 0.0),
                    4,
                ),
                "supporting_ticket_ids": [str(t) for t in (row.get("supporting_ticket_ids") or [])[:5]],
                "source_theme_id": row.get("source_theme_id") or "",
            }
        )

    return compact
