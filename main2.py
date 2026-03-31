async def debug_one_ticket(
    ticket_id: str,
    jira_client: JiraValueStreamClient,
    cfg: JiraIngestionConfig,
    out_dir: Optional[Path],
    probe_downloads: bool,
) -> dict:
    ticket_data = await jira_client.get_ticket_data(ticket_id, config=cfg)
    fields = ticket_data.get("fields", {})
    attachments = ticket_data.get("attachments", []) or []
    ticket_summary = str(fields.get("summary") or ticket_id)

    report: dict[str, Any] = {
        "ticket_id": ticket_id,
        "summary": ticket_summary,
        "attachment_count": len(attachments),
        "all_attachments": [_simple_attachment_view(a) for a in attachments],
    }

    if layer0_filter is not None:
        try:
            l0 = layer0_filter(attachments)
            report["layer0_survivors"] = [_simple_attachment_view(a) for a in l0]
        except Exception as exc:
            report["layer0_error"] = str(exc)
            l0 = []
    else:
        report["layer0_survivors"] = "layer0_filter not importable in this version"
        l0 = attachments

    if layer1_score is not None:
        try:
            l1 = layer1_score(l0)
            report["layer1_scored"] = [_simple_attachment_view(a) for a in l1]
        except Exception as exc:
            report["layer1_error"] = str(exc)
    else:
        report["layer1_scored"] = "layer1_score not importable in this version"

    # Prefetch once, then pass triage a sync bytes function.
    prefetched: dict[str, bytes] = {}
    prefetch_errors: list[dict[str, str]] = []

    for att in attachments:
        key = _attachment_key(att)
        try:
            data = await jira_client.download_attachment(att)
            if data:
                prefetched[key] = data
        except Exception as exc:
            prefetch_errors.append(
                {
                    "id": key,
                    "filename": str(att.get("filename", "")),
                    "error": str(exc),
                }
            )

    report["prefetched"] = {
        "ok_ids": sorted(prefetched.keys()),
        "count": len(prefetched),
        "errors": prefetch_errors,
    }

    def cached_download(att: dict) -> bytes:
        key = _attachment_key(att)
        data = prefetched.get(key)
        if data is None:
            raise RuntimeError(f"No prefetched bytes for attachment: {att.get('filename', key)}")
        return data

    primary, supporting, att_quality, triage_artifact = await _maybe_async_triage(
        attachments=attachments,
        ticket_summary=ticket_summary,
        download_fn=cached_download,
    )

    if triage_artifact is None:
        try:
            all_scored = layer1_score(layer0_filter(attachments)) if layer0_filter and layer1_score else []
            triage_artifact = build_triage_artifact(
                primary=primary,
                supplementary=supporting,
                att_quality=att_quality,
                all_scored=all_scored,
                total_attachment_count=len(attachments),
            )
        except Exception as exc:
            triage_artifact = {"error": f"Could not build triage artifact: {exc}"}

    chunk_candidates = []
    try:
        chunk_candidates = get_chunking_candidates(triage_artifact)
    except Exception as exc:
        report["chunk_candidate_error"] = str(exc)

    report["triage"] = {
        "att_quality": att_quality,
        "primary": _simple_attachment_view(primary) if primary else None,
        "supporting": [_simple_attachment_view(a) for a in supporting],
        "chunk_candidates": [_simple_attachment_view(a) for a in chunk_candidates],
        "artifact": triage_artifact,
    }

    if probe_downloads:
        probes = []
        for a in chunk_candidates:
            key = _attachment_key(a)
            data = prefetched.get(key)
            probes.append(
                {
                    "id": key,
                    "filename": a.get("filename", ""),
                    "download_ok": data is not None,
                    "bytes": len(data) if data else 0,
                    **({"error": "not prefetched"} if data is None else {}),
                }
            )
        report["download_probe"] = probes

    if out_dir:
        ticket_out = out_dir / ticket_id
        ticket_out.mkdir(parents=True, exist_ok=True)
        with open(ticket_out / "triage_debug.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    return report
