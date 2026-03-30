def _inject_missing_historical_candidates(candidates: List[dict], vs_support: List[dict]) -> int:
    existing = {
        _norm_name(c.get("entity_name"))
        for c in candidates
        if isinstance(c, dict) and c.get("entity_name")
    }

    # canonical ids already present in KG candidates by name
    candidate_by_name = {
        _norm_name(c.get("entity_name")): c
        for c in candidates
        if isinstance(c, dict) and c.get("entity_name")
    }

    injected = 0
    for row in vs_support:
        if not isinstance(row, dict):
            continue

        name = str(row.get("entity_name") or "").strip()
        if not name:
            continue

        key = _norm_name(name)
        if key in existing:
            continue

        kg_match = candidate_by_name.get(key)

        candidates.append(
            {
                "entity_id": (kg_match or {}).get("entity_id") or row.get("entity_id") or "",
                "entity_name": (kg_match or {}).get("entity_name") or name,
                "description": f"Recovered from historical ticket support ({int(row.get('support_count') or 0)} similar tickets).",
                "value_proposition": "",
                "support_count": int(row.get("support_count") or 0),
                "score": float(row.get("best_support_score") or row.get("score") or 0.0),
                "candidate_position": "historical_trigger",
                "source": "historical_support",
            }
        )
        existing.add(key)
        injected += 1

    return injected
