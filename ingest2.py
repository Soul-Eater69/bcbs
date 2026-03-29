from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

# Prefer package exports the repo already uses.
from jira_ingestion import JiraIngestionConfig, JiraValueStreamClient

# Triage imports with compatibility for older/newer naming.
try:
    from jira_ingestion.ingestion.triage import (
        triage_attachments,
        get_chunking_candidates,
        build_triage_artifact,
    )
except ImportError as exc:
    raise RuntimeError("Could not import triage APIs from jira_ingestion.ingestion.triage") from exc

try:
    from jira_ingestion.ingestion.triage import layer0_filter, layer1_score  # type: ignore
except Exception:
    try:
        from jira_ingestion.ingestion.triage import Layer0_filter as layer0_filter  # type: ignore
        from jira_ingestion.ingestion.triage import Layer1_score as layer1_score  # type: ignore
    except Exception:
        layer0_filter = None  # type: ignore
        layer1_score = None  # type: ignore


def _attachment_key(att: dict) -> str:
    return str(att.get("id") or att.get("filename") or "")


def _simple_attachment_view(att: dict) -> dict:
    return {
        "id": str(att.get("id") or ""),
        "filename": att.get("filename", ""),
        "ext": att.get("ext", ""),
        "size": att.get("size", 0),
        "mimeType": att.get("mimeType", att.get("mime_type", "")),
        "created": att.get("created", ""),
        "reporter_upload": att.get("reporter_upload", False),
        "triage_score": att.get("triage_score"),
        "triage_reasons": att.get("triage_reasons", []),
    }


async def _download_probe(jira_client: JiraValueStreamClient, att: dict) -> dict:
    try:
        data = await jira_client.download_attachment(att)
        return {
            "id": _attachment_key(att),
            "filename": att.get("filename", ""),
            "download_ok": bool(data),
            "bytes": len(data) if data else 0,
        }
    except Exception as exc:
        return {
            "id": _attachment_key(att),
            "filename": att.get("filename", ""),
            "download_ok": False,
            "bytes": 0,
            "error": str(exc),
        }


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

    async def download_fn(att: dict) -> bytes:
        return await jira_client.download_attachment(att)

    primary, supporting, att_quality, triage_artifact = await _maybe_async_triage(
        attachments=attachments,
        ticket_summary=ticket_summary,
        download_fn=download_fn,
    )

    # Some versions return a triage artifact directly, some need it built.
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
        probes = await asyncio.gather(*[_download_probe(jira_client, a) for a in chunk_candidates])
        report["download_probe"] = probes

    if out_dir:
        ticket_out = out_dir / ticket_id
        ticket_out.mkdir(parents=True, exist_ok=True)
        with open(ticket_out / "triage_debug.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    return report


async def _maybe_async_triage(**kwargs):
    result = triage_attachments(**kwargs)
    if asyncio.iscoroutine(result):
        result = await result

    # Expected new shape: (primary, supporting, att_quality, triage_artifact)
    if isinstance(result, tuple) and len(result) == 4:
        return result

    # Older shape: (primary, supporting, att_quality)
    if isinstance(result, tuple) and len(result) == 3:
        primary, supporting, att_quality = result
        return primary, supporting, att_quality, None

    raise RuntimeError(f"Unexpected triage_attachments return shape: {type(result)} {result!r}")


def _print_report(report: dict) -> None:
    print("=" * 90)
    print(f"TICKET: {report['ticket_id']}")
    print(f"SUMMARY: {report['summary']}")
    print(f"ATTACHMENTS: {report['attachment_count']}")
    print("\nALL ATTACHMENTS:")
    for a in report.get("all_attachments", []):
        print(f"  - {a['id']:>8} | {a['ext']:<5} | {a['size']:>9} | {a['filename']}")

    triage = report.get("triage", {})
    primary = triage.get("primary")
    print("\nTRIAGE RESULT:")
    print(f"  att_quality: {triage.get('att_quality')}")
    print(f"  primary: {primary['filename']}" if primary else "  primary: None")

    print("  supporting:")
    for a in triage.get("supporting", []):
        print(f"    - {a['filename']} ({a.get('ext')}, score={a.get('triage_score')})")

    print("  chunk_candidates:")
    for a in triage.get("chunk_candidates", []):
        print(f"    - {a['filename']} ({a.get('ext')}, score={a.get('triage_score')})")

    artifact = triage.get("artifact", {}) or {}
    if isinstance(artifact, dict):
        plan = artifact.get("processing_plan") or {}
        excluded = artifact.get("excluded_attachments") or []
        print("\nPROCESSING PLAN:")
        if plan:
            for k, v in plan.items():
                print(f"  {k}: {v}")
        else:
            print("  <none in artifact>")

        print("\nEXCLUDED ATTACHMENTS:")
        if excluded:
            for x in excluded:
                if isinstance(x, dict):
                    print(f"  - {x.get('filename', x.get('id', ''))}")
                else:
                    print(f"  - {x}")
        else:
            print("  <none>")

    if report.get("download_probe"):
        print("\nDOWNLOAD PROBE:")
        for d in report["download_probe"]:
            ok = "OK" if d.get("download_ok") else "FAIL"
            print(f"  - [{ok}] {d['filename']} bytes={d.get('bytes', 0)}")
            if d.get("error"):
                print(f"      error={d['error']}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Debug Jira attachment triage for one or more tickets.")
    parser.add_argument("tickets", nargs="+", help="Ticket ids, e.g. IDMT-1320")
    parser.add_argument("--out-dir", default="triage_debug", help="Where to save triage_debug.json files")
    parser.add_argument("--no-save", action="store_true", help="Do not write JSON files")
    parser.add_argument("--probe-downloads", action="store_true", help="Try downloading chunk candidates")
    parser.add_argument("--verify-ssl", action="store_true", help="Enable SSL verification for Jira client")
    args = parser.parse_args()

    base_url = os.environ.get("JIRA_BASE_URL")
    token = os.environ.get("JIRA_TOKEN")
    if not base_url or not token:
        raise RuntimeError("Set JIRA_BASE_URL and JIRA_TOKEN in the environment before running.")

    cfg = JiraIngestionConfig()
    out_dir = None if args.no_save else Path(args.out_dir)

    async with JiraValueStreamClient(base_url=base_url, token=token, verify_ssl=args.verify_ssl) as jira_client:
        for ticket_id in args.tickets:
            report = await debug_one_ticket(
                ticket_id=ticket_id,
                jira_client=jira_client,
                cfg=cfg,
                out_dir=out_dir,
                probe_downloads=args.probe_downloads,
            )
            _print_report(report)


if __name__ == "__main__":
    asyncio.run(main())
