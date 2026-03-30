import json

def _parse_chunk_provenance(prov):
    if isinstance(prov, dict):
        return prov
    if isinstance(prov, str) and prov.strip():
        try:
            return json.loads(prov)
        except Exception:
            return {}
    return {}

def _build_chunk_ref(chunk: dict) -> str:
    prov = _parse_chunk_provenance(chunk.get("chunkProvenance"))

    attachment_id = str(prov.get("attachmentId") or "").strip()
    source_type = str(prov.get("sourceType") or "").strip()
    chunk_index = prov.get("chunkIndex")
    page = (
        prov.get("pageNumber")
        or prov.get("page")
        or prov.get("pageIndex")
    )

    if source_type == "pdf_page":
        page_part = f"page-{page}" if page is not None else "page-?"
        chunk_part = f"chunk-{chunk_index}" if chunk_index is not None else "chunk-?"
        if attachment_id:
            return f"att-{attachment_id}:{page_part}:{chunk_part}"
        return f"{page_part}:{chunk_part}"

    if chunk_index is not None:
        if attachment_id:
            return f"att-{attachment_id}:chunk-{chunk_index}"
        return f"chunk-{chunk_index}"

    return str(chunk.get("id") or "").strip()
