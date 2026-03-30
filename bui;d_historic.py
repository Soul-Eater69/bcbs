def build_historical_vs_evidence(
    chunks: List[dict],
    ticket_vs_map: Dict[str, dict],
    max_ticket_hits: int = 20,
) -> Tuple[List[dict], List[dict]]:
    ticket_hits: Dict[str, dict] = {}

    for chunk in chunks:
        tid = str(chunk.get("sourceId") or "").strip()
        if not tid:
            continue

        score = float(chunk.get("_score", 0.0) or 0.0)
        raw_doc_id = str(chunk.get("id") or "")
        chunk_ref = _build_chunk_ref(chunk)

        if tid not in ticket_hits:
            ticket_hits[tid] = {
                "ticket_id": tid,
                "best_score": score,
                "matched_chunk_ids": [chunk_ref] if chunk_ref else [],
                "matched_doc_ids": [raw_doc_id] if raw_doc_id else [],
                "title": chunk.get("title", ""),
            }
        else:
            ticket_hits[tid]["best_score"] = max(ticket_hits[tid]["best_score"], score)
            if chunk_ref and chunk_ref not in ticket_hits[tid]["matched_chunk_ids"]:
                ticket_hits[tid]["matched_chunk_ids"].append(chunk_ref)
            if raw_doc_id and raw_doc_id not in ticket_hits[tid]["matched_doc_ids"]:
                ticket_hits[tid]["matched_doc_ids"].append(raw_doc_id)

    ranked_tickets = sorted(
        ticket_hits.values(),
        key=lambda x: x["best_score"],
        reverse=True
    )[:max_ticket_hits]

    support_by_vs: Dict[str, dict] = {}
    for hit in ranked_tickets:
        vs_rec = ticket_vs_map.get(hit["ticket_id"])
        if not vs_rec:
            continue

        names = vs_rec.get("valueStreamNames") or []
        ids = vs_rec.get("valueStreamIds") or []
        statuses = vs_rec.get("valueStreamStatuses") or []

        for idx, name in enumerate(names):
            vs_name = str(name or "").strip()
            if not vs_name:
                continue

            if vs_name not in support_by_vs:
                support_by_vs[vs_name] = {
                    "entity_name": vs_name,
                    "entity_id": ids[idx] if idx < len(ids) else "",
                    "statuses": [],
                    "support_count": 0,
                    "supporting_ticket_ids": [],
                    "supporting_chunk_ids": [],
                    "supporting_doc_ids": [],
                    "best_support_score": 0.0,
                }

            entry = support_by_vs[vs_name]
            entry["support_count"] += 1
            entry["best_support_score"] = max(entry["best_support_score"], hit["best_score"])

            if hit["ticket_id"] not in entry["supporting_ticket_ids"]:
                entry["supporting_ticket_ids"].append(hit["ticket_id"])

            for cid in hit["matched_chunk_ids"][:5]:
                if cid and cid not in entry["supporting_chunk_ids"]:
                    entry["supporting_chunk_ids"].append(cid)

            for did in hit.get("matched_doc_ids", [])[:5]:
                if did and did not in entry["supporting_doc_ids"]:
                    entry["supporting_doc_ids"].append(did)

            if idx < len(statuses):
                st = str(statuses[idx] or "").strip()
                if st and st not in entry["statuses"]:
                    entry["statuses"].append(st)

    vs_support = sorted(
        support_by_vs.values(),
        key=lambda x: (x["support_count"], x["best_support_score"]),
        reverse=True,
    )

    return ranked_tickets, vs_support
