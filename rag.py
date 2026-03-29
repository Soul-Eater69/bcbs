from __future__ import annotations

import json
import logging
import math
import time
from typing import Any, Dict, Iterable, List, Optional

from src.services.generation_service import GenerationService

from core.constants import CANONICAL_VALUE_STREAMS
from core.prompts import load_rag_prompts, render_prompt, safe_json_extract
from core.text import clean_ppt_text, normalize_for_search
from retrieval import (
    build_historical_vs_evidence,
    load_ticket_vs_maps,
    retrieve_historical_chunks,
    retrieve_kg_candidates,
    sanitize_selected,
)

logger = logging.getLogger(__name__)


def _norm_name(value: Optional[str]) -> str:
    return normalize_for_search((value or "").strip())


def _approx_tokens(*parts: str) -> int:
    total_chars = sum(len(p or "") for p in parts)
    return max(1, math.ceil(total_chars / 4))


def _short(text: str, max_chars: int = 220) -> str:
    text = clean_ppt_text(text or "")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _dedupe_by_name(rows: Iterable[dict]) -> List[dict]:
    out: List[dict] = []
    seen: set[str] = set()
    for row in rows:
        name = _norm_name(row.get("entity_name") or row.get("name"))
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(row)
    return out


def _limit_historical_chunks(chunks: List[dict], limit: int) -> List[dict]:
    """Favor diverse tickets plus strong scores, instead of longest chunks only."""
    if not chunks:
        return []

    def score(chunk: dict) -> tuple:
        return (
            float(chunk.get("best_score") or chunk.get("score") or 0.0),
            int(chunk.get("word_count") or len((chunk.get("text") or "").split())),
        )

    ranked = sorted(chunks, key=score, reverse=True)
    per_ticket_cap = 2
    per_ticket: Dict[str, int] = {}
    selected: List[dict] = []
    overflow: List[dict] = []

    for chunk in ranked:
        ticket_id = str(chunk.get("ticket_id") or chunk.get("source_id") or "")
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


def _compact_historical_evidence(chunks: List[dict], limit: int = 12) -> List[dict]:
    compact: List[dict] = []
    for chunk in _limit_historical_chunks(chunks, limit):
        compact.append(
            {
                "ticket_id": str(chunk.get("ticket_id") or chunk.get("source_id") or ""),
                "title": _short(str(chunk.get("title") or chunk.get("ticket_title") or ""), 120),
                "source": str(chunk.get("source") or ""),
                "score": round(float(chunk.get("best_score") or chunk.get("score") or 0.0), 4),
                "snippet": _short(str(chunk.get("text") or ""), 240),
                "section": _short(str(chunk.get("section_title") or chunk.get("slide_title") or ""), 80),
            }
        )
    return compact


def _compact_vs_support(vs_support: List[dict], limit: int = 15) -> List[dict]:
    rows = sorted(
        vs_support,
        key=lambda row: (
            int(row.get("support_count") or 0),
            float(row.get("confidence") or row.get("score") or 0.0),
        ),
        reverse=True,
    )
    compact: List[dict] = []
    for row in _dedupe_by_name(rows)[:limit]:
        snippets = row.get("supporting_snippets") or row.get("sample_snippets") or []
        if isinstance(snippets, list):
            sample_snippets = [_short(str(s), 160) for s in snippets[:2] if s]
        else:
            sample_snippets = [_short(str(snippets), 160)] if snippets else []
        compact.append(
            {
                "entity_id": row.get("entity_id") or row.get("id") or "",
                "entity_name": row.get("entity_name") or row.get("name") or "",
                "support_count": int(row.get("support_count") or 0),
                "confidence": round(float(row.get("confidence") or row.get("score") or 0.0), 4),
                "supporting_ticket_ids": [str(t) for t in (row.get("supporting_ticket_ids") or [])[:5]],
                "sample_snippets": sample_snippets,
            }
        )
    return compact


def _compact_candidates(candidates: List[dict], limit: int = 30) -> List[dict]:
    rows = sorted(
        candidates,
        key=lambda row: (
            float(row.get("best_score") or row.get("score") or 0.0),
            int(row.get("support_count") or 0),
        ),
        reverse=True,
    )
    compact: List[dict] = []
    for row in _dedupe_by_name(rows)[:limit]:
        compact.append(
            {
                "entity_id": row.get("entity_id") or row.get("id") or "",
                "entity_name": row.get("entity_name") or row.get("name") or "",
                "category": row.get("category") or "",
                "score": round(float(row.get("best_score") or row.get("score") or 0.0), 4),
                "description": _short(str(row.get("description") or row.get("value_proposition") or ""), 220),
            }
        )
    return compact


def _inject_missing_historical_candidates(candidates: List[dict], vs_support: List[dict]) -> int:
    existing = {_norm_name(c.get("entity_name")) for c in candidates}
    injected = 0
    for row in vs_support:
        name = (row.get("entity_name") or "").strip()
        if not name:
            continue
        key = _norm_name(name)
        if key in existing:
            continue
        candidates.append(
            {
                "entity_id": row.get("entity_id") or "",
                "entity_name": name,
                "description": f"Recovered from historical ticket support ({int(row.get('support_count') or 0)} similar tickets).",
                "value_proposition": row.get("value_proposition") or "",
                "support_count": int(row.get("support_count") or 0),
                "score": float(row.get("confidence") or row.get("score") or 0.0),
                "candidate_position": "historical_trigger",
                "source": "historical_support",
            }
        )
        existing.add(key)
        injected += 1
    return injected


def _fallback_from_support(vs_support: List[dict], candidates: List[dict], top_n: int = 8) -> List[dict]:
    by_name = {_norm_name(c.get("entity_name")): c for c in candidates if c.get("entity_name")}
    selected: List[dict] = []
    for row in _compact_vs_support(vs_support, limit=top_n * 2):
        name = (row.get("entity_name") or "").strip()
        key = _norm_name(name)
        if not name or key not in by_name:
            continue
        cand = by_name[key]
        selected.append(
            {
                "entity_id": cand.get("entity_id") or row.get("entity_id") or "",
                "entity_name": cand.get("entity_name") or name,
                "confidence": round(min(0.95, 0.45 + 0.08 * int(row.get("support_count") or 0)), 4),
                "reason": "Fallback from historical ticket support.",
                "description": cand.get("description") or "",
                "supporting_ticket_ids": row.get("supporting_ticket_ids", []),
                "supporting_snippets": row.get("sample_snippets", []),
            }
        )
        if len(selected) >= top_n:
            break
    return selected


def _fallback_from_candidates(candidates: List[dict], top_n: int = 8) -> List[dict]:
    return [
        {
            "entity_id": c.get("entity_id") or c.get("id") or "",
            "entity_name": c.get("entity_name") or c.get("name") or "",
            "confidence": 0.45,
            "reason": "Fallback from top KG candidates (LLM output empty/unavailable).",
            "description": c.get("description") or "",
            "supporting_ticket_ids": [],
            "supporting_snippets": [],
        }
        for c in _compact_candidates(candidates, limit=top_n)
    ]


def _filter_to_allowed(rows: List[dict], allowed_names: Optional[List[str]]) -> List[dict]:
    if not allowed_names:
        return rows
    allowed = {_norm_name(name) for name in allowed_names if name}
    return [r for r in rows if _norm_name(r.get("entity_name")) in allowed]


def _inject_explicit_canonical_subset(
    rows: List[dict],
    canonical_always_include: Optional[List[str]],
    allowed_names: Optional[List[str]],
) -> int:
    if not canonical_always_include:
        return 0

    allowed = {_norm_name(name) for name in allowed_names or [] if name}
    canonical_index = {_norm_name(v.get("name")): v for v in CANONICAL_VALUE_STREAMS}
    existing = {_norm_name(r.get("entity_name")) for r in rows}
    injected = 0

    for name in canonical_always_include:
        key = _norm_name(name)
        if not key or key in existing:
            continue
        if allowed and key not in allowed:
            continue
        canon = canonical_index.get(key)
        if not canon:
            continue
        rows.append(
            {
                "entity_id": canon.get("id") or "",
                "entity_name": canon.get("name") or name,
                "confidence": 1.0,
                "reason": "Explicit canonical include.",
                "category": canon.get("category") or "",
                "supporting_ticket_ids": [],
                "supporting_snippets": [],
            }
        )
        existing.add(key)
        injected += 1
    return injected


def generate_value_streams_rag(
    ppt_text: str,
    allowed_value_stream_names: Optional[List[str]] = None,
    local_vs_map_dir: str = "jira_prechunk_output",
    top_per_view: int = 12,
    max_historical_chunks: int = 60,
    max_candidate_streams: int = 50,
    llm_max_retries: int = 2,
    canonical_always_include: Optional[List[str]] = None,
) -> dict:
    """RAG pipeline for value-stream selection with safer evidence handling.

    Flow:
      1. Clean input text.
      2. Retrieve historical chunks and cap with ticket diversity.
      3. Map historical tickets to value stream support.
      4. Retrieve KG candidates.
      5. Ask LLM to choose from grounded candidates.
      6. Fall back deterministically if LLM fails.
      7. Optionally inject a very small explicit canonical subset.

    Important change:
      Full canonical catalog is no longer auto-appended to selections.
      Canonicals are only added when `canonical_always_include` is explicitly passed.
    """
    logger.info("=" * 80)
    logger.info("[RAG] Starting value-stream pipeline")
    t_start = time.time()

    cleaned_text = clean_ppt_text(ppt_text)
    if not cleaned_text.strip():
        logger.warning("[RAG] Empty input after cleaning")
        return {
            "selected_value_streams": [],
            "rejected_candidates": [],
            "historical_ticket_hits": [],
            "historical_value_stream_support": [],
            "candidate_value_streams": [],
            "raw_response": None,
            "warnings": ["empty_input_after_cleaning"],
            "error": "empty_input_text",
        }

    logger.info("[RAG] Step 1: historical chunk retrieval")
    t1 = time.time()
    historical_chunks = retrieve_historical_chunks(cleaned_text, top_k_per_view=top_per_view)
    historical_chunks = _limit_historical_chunks(historical_chunks, max_historical_chunks)
    logger.info(
        "[RAG] Step 1 done: %d chunks in %.2fs",
        len(historical_chunks),
        time.time() - t1,
    )

    logger.info("[RAG] Step 2: ticket -> VS support")
    t2 = time.time()
    vs_map = load_ticket_vs_maps(local_vs_map_dir)
    vs_support = build_historical_vs_evidence(historical_chunks, vs_map)
    vs_support = _filter_to_allowed(vs_support, allowed_value_stream_names)
    logger.info(
        "[RAG] Step 2 done: %d VS support rows in %.2fs",
        len(vs_support),
        time.time() - t2,
    )

    logger.info("[RAG] Step 3: KG candidate retrieval")
    t3 = time.time()
    candidates = retrieve_kg_candidates(
        cleaned_text,
        top_n=max_candidate_streams,
        allowed_names=allowed_value_stream_names,
    )
    injected = _inject_missing_historical_candidates(candidates, vs_support)
    candidates = _filter_to_allowed(candidates, allowed_value_stream_names)
    logger.info(
        "[RAG] Step 3 done: %d candidates (+%d from historical support) in %.2fs",
        len(candidates),
        injected,
        time.time() - t3,
    )

    logger.info("[RAG] Step 4: LLM selection")
    t4 = time.time()
    prompts = load_rag_prompts()
    scope = (
        f"Use only this explicit subset: {allowed_value_stream_names}."
        if allowed_value_stream_names
        else "Use only the provided candidate value streams."
    )

    compact_evidence = _compact_historical_evidence(historical_chunks, limit=min(12, max_historical_chunks))
    compact_support = _compact_vs_support(vs_support, limit=15)
    compact_candidates = _compact_candidates(candidates, limit=min(30, max_candidate_streams))

    user_prompt = render_prompt(
        prompts["selection_user"],
        ppt_text=normalize_for_search(cleaned_text)[:2500],
        historical_evidence_json=json.dumps(compact_evidence, ensure_ascii=False),
        historical_vs_support_json=json.dumps(compact_support, ensure_ascii=False),
        candidate_value_streams_json=json.dumps(compact_candidates, ensure_ascii=False),
        selection_scope=scope,
    )
    system_prompt = prompts.get("system", "")
    logger.info(
        "[RAG] Prompt: ~%d tokens | %d candidates | %d evidence | %d support",
        _approx_tokens(system_prompt, user_prompt),
        len(compact_candidates),
        len(compact_evidence),
        len(compact_support),
    )

    parsed: Optional[dict] = None
    raw_response: Optional[str] = None
    warnings: List[str] = []
    llm_ok = False

    for attempt in range(1, llm_max_retries + 1):
        try:
            logger.info("[RAG-LLM] Attempt %d/%d", attempt, llm_max_retries)
            llm_t = time.time()
            reply = GenerationService().generate(
                query=user_prompt,
                context="",
                system_prompt=system_prompt,
            )
            raw_response = reply.content
            logger.info(
                "[RAG-LLM] Completed in %.2fs | preview=%r",
                time.time() - llm_t,
                (raw_response or "")[:500],
            )
            parsed = safe_json_extract(raw_response)
            parsed = sanitize_selected(parsed, candidates)
            parsed["selected_value_streams"] = _filter_to_allowed(
                parsed.get("selected_value_streams", []),
                allowed_value_stream_names,
            )
            llm_ok = True
            break
        except Exception as exc:
            logger.error("[RAG-LLM] Attempt %d failed: %s", attempt, exc)
            if attempt < llm_max_retries:
                time.sleep(3 * attempt)
            else:
                warnings.append(f"llm_unavailable_fallback_used:{type(exc).__name__}")

    if not llm_ok:
        logger.warning("[RAG-FALLBACK] All LLM attempts failed")
        fb = _fallback_from_support(vs_support, candidates, top_n=8)
        if not fb:
            fb = _fallback_from_candidates(candidates, top_n=8)
        parsed = {"selected_value_streams": fb, "rejected_candidates": []}
    elif not parsed or not parsed.get("selected_value_streams"):
        logger.warning("[RAG-FALLBACK] LLM returned no selected value streams")
        warnings.append("llm_empty_selection_fallback_used")
        fb = _fallback_from_support(vs_support, candidates, top_n=8)
        if not fb:
            fb = _fallback_from_candidates(candidates, top_n=8)
        parsed = {"selected_value_streams": fb, "rejected_candidates": parsed.get("rejected_candidates", []) if parsed else []}

    canon_injected = _inject_explicit_canonical_subset(
        parsed["selected_value_streams"],
        canonical_always_include=canonical_always_include,
        allowed_names=allowed_value_stream_names,
    )
    if canon_injected:
        logger.info("[RAG] Injected %d explicit canonical value streams", canon_injected)

    final = _dedupe_by_name(parsed.get("selected_value_streams", []))
    final = _filter_to_allowed(final, allowed_value_stream_names)

    logger.info(
        "[RAG] DONE in %.2fs | selected=%d | llm_ok=%s | warnings=%s",
        time.time() - t_start,
        len(final),
        llm_ok,
        warnings or "none",
    )

    return {
        "selected_value_streams": final,
        "rejected_candidates": parsed.get("rejected_candidates", []),
        "historical_ticket_hits": compact_evidence,
        "historical_value_stream_support": compact_support,
        "candidate_value_streams": compact_candidates,
        "raw_response": raw_response,
        "warnings": warnings,
    }
