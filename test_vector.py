def _chunk_score(chunk: dict) -> float:
    return float(chunk.get("_score") or chunk.get("best_score") or chunk.get("score") or 0.0)


def _chunk_ticket_id(chunk: dict) -> str:
    return str(chunk.get("sourceId") or chunk.get("ticket_id") or chunk.get("source_id") or "").strip()


def _chunk_text(chunk: dict) -> str:
    return str(chunk.get("content") or chunk.get("text") or chunk.get("snippet") or "").strip()


def _chunk_provenance_text(chunk: dict) -> str:
    prov = chunk.get("chunkProvenance") or {}
    if isinstance(prov, str):
        try:
            prov = json.loads(prov)
        except Exception:
            prov = {}

    if not isinstance(prov, dict):
        prov = {}

    page_range = prov.get("pageRange") or []
    slide_range = prov.get("slideRange") or []
    source_type = str(prov.get("sourceType") or "").strip()

    if isinstance(page_range, list) and page_range:
        if len(page_range) == 1:
            where = f"page-{page_range[0]}"
        else:
            where = f"pages-{page_range[0]}-{page_range[-1]}"
    elif isinstance(slide_range, list) and slide_range:
        if len(slide_range) == 1:
            where = f"slide-{slide_range[0]}"
        else:
            where = f"slides-{slide_range[0]}-{slide_range[-1]}"
    else:
        where = ""

    return ":".join([p for p in [source_type, where] if p])
