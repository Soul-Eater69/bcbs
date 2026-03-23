import json
import re
from typing import List


def _build_context(matches: List[dict]) -> str:
    parts = []

    for m in matches:
        if not m.get("entity_name"):
            continue

        score = m.get("@search.reranker_score", m.get("@search.score", 0))

        block = (
            f"Candidate Value Stream: {m.get('entity_name', '')}\n"
            f"Entity ID: {m.get('entity_id', '')}\n"
            f"Retrieval Score: {score:.4f}"
        )

        if m.get("content"):
            block += f"\nDescription: {m['content']}"

        if m.get("properties"):
            try:
                props = (
                    json.loads(m["properties"])
                    if isinstance(m["properties"], str)
                    else m["properties"]
                )

                if props.get("value_stream_value_proposition"):
                    block += f"\nValue Proposition: {props['value_stream_value_proposition']}"

                if props.get("value_stream_trigger"):
                    block += f"\nTrigger: {props['value_stream_trigger']}"

                stages = props.get("stages", [])
                if stages:
                    block += "\nStages:"
                    for idx, s in enumerate(stages, 1):
                        stage_name = s.get("value_stream_stage_name", "")
                        stage_desc = s.get("value_stream_stage_description", "")
                        stage_criteria = s.get("value_stream_stage_entrance_criteria", "")

                        block += f"\n  {idx}. {stage_name}"
                        if stage_desc:
                            block += f" — {stage_desc}"
                        if stage_criteria:
                            block += f"\n     Entrance Criteria: {stage_criteria}"

            except (json.JSONDecodeError, TypeError, KeyError):
                pass

        parts.append(block)

    return "\n\n".join(parts)


def _safe_json_extract(text: str) -> dict:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass

    return {
        "selected_value_streams": [],
        "rejected_candidates": [],
    }


def _build_value_stream_selection_prompt(query: str, context: str) -> str:
    return f"""
You are an expert at classifying healthcare idea cards into existing value streams.

Your job is to choose the best matching EXISTING value streams for the user's PPT content.

Rules:
1. You must choose only from the candidate value streams shown below.
2. Do not invent new value streams.
3. Do not rename or merge value streams.
4. Multiple value streams may be selected only if clearly supported by the PPT.
5. Prefer precision over recall.
6. Use description, value proposition, trigger, and stages when deciding.
7. Ignore weak keyword overlap.
8. Return valid JSON only.
9. If no candidate is strongly supported, return an empty selected_value_streams list.

USER PPT CONTENT:
{query}

CANDIDATE VALUE STREAMS:
{context}

Return JSON exactly in this format:
{{
  "selected_value_streams": [
    {{
      "entity_id": "string",
      "entity_name": "string",
      "confidence": 0.0,
      "reason": "short explanation"
    }}
  ],
  "rejected_candidates": [
    {{
      "entity_id": "string",
      "entity_name": "string",
      "reason": "short explanation"
    }}
  ]
}}
""".strip()


def select_value_streams(
    query: str,
    fetch_count: int = 10,
) -> dict:
    client = AzureDirectSearchClient()
    gen_svc = GenerationService()

    matches = client.search_hybrid(query, top_k=fetch_count)

    if not matches:
        matches = client.search_vector(query, top_k=fetch_count)

    if not matches:
        return {
            "selected_value_streams": [],
            "rejected_candidates": [],
            "raw_response": None,
            "candidates_used": [],
        }

    context = _build_context(matches)
    prompt = _build_value_stream_selection_prompt(query=query, context=context)

    # keep your existing function contract
    reply = gen_svc.generate(query=prompt, context="")

    parsed = _safe_json_extract(reply.content)

    return {
        "selected_value_streams": parsed.get("selected_value_streams", []),
        "rejected_candidates": parsed.get("rejected_candidates", []),
        "raw_response": reply.content,
        "candidates_used": [
            {
                "entity_id": m.get("entity_id", ""),
                "entity_name": m.get("entity_name", ""),
                "@search.score": m.get("@search.score", 0),
                "@search.reranker_score": m.get("@search.reranker_score", 0),
            }
            for m in matches
        ],
    }


def generate_value_streams(
    query: str,
    fetch_count: int = 10,
) -> str:
    """
    Keeping your old function name for easy drop-in replacement.
    But this now SELECTS existing value streams instead of generating new ones.
    """
    result = select_value_streams(query=query, fetch_count=fetch_count)
    return json.dumps(
        {
            "selected_value_streams": result["selected_value_streams"],
            "rejected_candidates": result["rejected_candidates"],
        },
        indent=2,
    )


def run_generate(query: str, top_k: int = 10) -> None:
    print(f"\n[Generate] {query[:80]}\n{'-' * 60}")

    result = select_value_streams(query=query, fetch_count=top_k)

    print("\nSelected Value Streams:")
    print(json.dumps(result["selected_value_streams"], indent=2))

    print("\nRejected Candidates:")
    print(json.dumps(result["rejected_candidates"], indent=2))

    print("\nCandidates Used:")
    print(json.dumps(result["candidates_used"], indent=2))

    print("\nRaw LLM Response:")
    print(result["raw_response"])




import json
import logging
import re
from typing import Dict, List

from src.clients.azure_direct_client import AzureDirectSearchClient
from src.services.generation_service import GenerationService
from src.services.value_stream_service import ValueStreamService

logging.basicConfig(level=logging.INFO)


def run_vector_search(query: str, top_k: int = 5) -> None:
    client = AzureDirectSearchClient()
    results = client.search_vector(query, top_k=top_k)

    if not results:
        print("No results.")
        return

    for i, doc in enumerate(results, 1):
        print(f"{i:>2}. [{doc.get('entity_id', '')}] {doc.get('entity_name', '')}")
        print(f"    Score : {doc.get('@search.score', 0):.4f}")
        print(f"    Desc  : {(doc.get('content') or '')[:150]}...")
        print()


def run_hybrid_search(query: str, top_k: int = 5) -> None:
    print(f"\n[Hybrid Search] {query[:120]}\n{'-' * 60}")
    client = AzureDirectSearchClient()
    results = client.search_hybrid(query, top_k=top_k)

    if not results:
        print("No results.")
        return

    for i, doc in enumerate(results, 1):
        print(f"{i:>2}. [{doc.get('entity_id', '')}] {doc.get('entity_name', '')}")
        print(f"    Score        : {doc.get('@search.score', 0):.4f}")
        print(f"    Reranker     : {doc.get('@search.reranker_score', 0):.4f}")
        print(f"    Desc         : {(doc.get('content') or '')[:150]}...")
        print()


def _clean_ppt_text(raw_text: str) -> str:
    text = raw_text or ""

    # Remove slide markers
    text = re.sub(r"<!--\s*Slide number:\s*\d+\s*-->", " ", text, flags=re.IGNORECASE)

    # Remove image markdown
    text = re.sub(r"!\[\]\([^)]+\)", " ", text)

    # Remove repeated confidentiality boilerplate
    text = re.sub(
        r"PROPRIETARY AND CONFIDENTIAL\s*-\s*FOR INTERNAL USE ONLY",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove common repeated metadata lines that add retrieval noise
    drop_prefixes = [
        "Current as of:",
        "Funding Portfolio:",
        "Project Category:",
        "Project Sub-Category:",
        "Intake BSL:",
        "Business Architect:",
        "EPMO Lead:",
        "Idea Card Author:",
        "Executive Sponsor:",
        "Accountable:",
        "Responsible:",
        "Functional Portfolio Owner:",
        "### Notes:",
        "# Appendix",
    ]

    kept_lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(prefix) for prefix in drop_prefixes):
            continue
        kept_lines.append(stripped)

    text = "\n".join(kept_lines)

    # Normalize separators and whitespace
    text = text.replace("|", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_signal_sections(raw_text: str) -> Dict[str, str]:
    """
    Lightweight section extractor based on headings present in the PPT text dump.
    """
    cleaned = (raw_text or "").replace("\r", "")

    section_names = [
        "Idea Card Executive Summary:",
        "Problem Statement/Market Opportunity:",
        "Business Solution and Objectives:",
        "Alternative Solutions:",
        "Value Proposition & Key Metrics:",
        "Interdependencies:",
        "Estimated Costs:",
        "Resources/Investments Needed for Business Case:",
    ]

    positions = []
    for name in section_names:
        idx = cleaned.find(name)
        if idx != -1:
            positions.append((idx, name))

    positions.sort()
    sections: Dict[str, str] = {}

    for i, (start, name) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(cleaned)
        body = cleaned[start:end].strip()
        sections[name] = body

    return sections


def _normalize_for_search(text: str, max_chars: int = 2500) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text[:max_chars]


def _build_retrieval_views(raw_text: str) -> List[str]:
    """
    Build multiple retrieval views instead of using the entire raw PPT blob.
    """
    cleaned = _clean_ppt_text(raw_text)
    sections = _extract_signal_sections(raw_text)

    executive = sections.get("Idea Card Executive Summary:", "")
    problem = sections.get("Problem Statement/Market Opportunity:", "")
    solution = sections.get("Business Solution and Objectives:", "")
    alternative = sections.get("Alternative Solutions:", "")
    value_prop = sections.get("Value Proposition & Key Metrics:", "")
    interdep = sections.get("Interdependencies:", "")
    costs = sections.get("Estimated Costs:", "")

    # View 1: overall business summary
    deck_summary = " ".join(
        [
            executive,
            problem[:1200],
            solution[:1500],
            value_prop[:1200],
        ]
    ).strip()

    # View 2: problem + objectives
    problem_objectives = " ".join(
        [
            problem,
            solution,
            alternative[:800],
        ]
    ).strip()

    # View 3: operational workflow / capabilities
    operational = " ".join(
        [
            solution,
            alternative,
            "benefit navigation member care inquiry support self-service hub digital support vendor platform partner onboarding provider network access to care prior authorization",
        ]
    ).strip()

    # View 4: commercial / setup / finance
    commercial = " ".join(
        [
            executive,
            value_prop,
            interdep,
            costs,
            "product offering pricing quote revenue margin invoice payment order to cash onboarding partner lead management",
        ]
    ).strip()

    # View 5: keyword-condensed view for hybrid retrieval
    keyword_view = """
    women's and family health new product offering
    growth new product
    benefit navigation
    member care
    request inquiry
    self-service hub
    fertility maternity postpartum parenting menopause
    vendor platform onboarding
    partner onboarding
    provider network
    sales product claims network health care management
    pricing quote
    revenue margin
    invoice payment
    order to cash
    lead management
    access to care
    digital support
    """.strip()

    views = [
        _normalize_for_search(cleaned, max_chars=2200),
        _normalize_for_search(deck_summary),
        _normalize_for_search(problem_objectives),
        _normalize_for_search(operational),
        _normalize_for_search(commercial),
        _normalize_for_search(keyword_view, max_chars=1200),
    ]

    # Deduplicate while preserving order
    deduped: List[str] = []
    seen = set()
    for v in views:
        key = v.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(v)

    return deduped


def _build_context(matches: List[dict]) -> str:
    parts = []

    for m in matches:
        if not m.get("entity_name"):
            continue

        score = m.get("@search.reranker_score", m.get("@search.score", 0))

        block = (
            f"Candidate Value Stream: {m.get('entity_name', '')}\n"
            f"Entity ID: {m.get('entity_id', '')}\n"
            f"Retrieval Score: {score:.4f}"
        )

        if m.get("content"):
            block += f"\nDescription: {m['content']}"

        if m.get("properties"):
            try:
                props = (
                    json.loads(m["properties"])
                    if isinstance(m["properties"], str)
                    else m["properties"]
                )

                if props.get("value_stream_value_proposition"):
                    block += f"\nValue Proposition: {props['value_stream_value_proposition']}"

                if props.get("value_stream_trigger"):
                    block += f"\nTrigger: {props['value_stream_trigger']}"

                stages = props.get("stages", [])
                if isinstance(stages, list) and stages:
                    block += "\nStages:"
                    for idx, s in enumerate(stages, 1):
                        if not isinstance(s, dict):
                            continue

                        stage_name = s.get("value_stream_stage_name", "")
                        stage_desc = s.get("value_stream_stage_description", "")
                        stage_criteria = s.get("value_stream_stage_entrance_criteria", "")

                        block += f"\n  {idx}. {stage_name}"
                        if stage_desc:
                            block += f" — {stage_desc}"
                        if stage_criteria:
                            block += f"\n     Entrance Criteria: {stage_criteria}"

            except (json.JSONDecodeError, TypeError, KeyError):
                pass

        parts.append(block)

    return "\n\n".join(parts)


def _safe_json_extract(text: str) -> dict:
    text = (text or "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass

    return {
        "selected_value_streams": [],
        "rejected_candidates": [],
    }


def _build_value_stream_selection_prompt(query: str, context: str) -> str:
    return f"""
You are an expert at classifying healthcare idea cards into existing value streams.

Your job is to choose the best matching EXISTING value streams for the user's PPT content.

Rules:
1. You must choose only from the candidate value streams shown below.
2. Do not invent new value streams.
3. Do not rename or merge value streams.
4. Multiple value streams may be selected only if clearly supported by the PPT.
5. Prefer precision over recall.
6. Use description, value proposition, trigger, and stages when deciding.
7. Ignore weak keyword overlap.
8. Return valid JSON only.
9. If no candidate is strongly supported, return an empty selected_value_streams list.

USER PPT CONTENT:
{query}

CANDIDATE VALUE STREAMS:
{context}

Return JSON exactly in this format:
{{
  "selected_value_streams": [
    {{
      "entity_id": "string",
      "entity_name": "string",
      "confidence": 0.0,
      "reason": "short explanation"
    }}
  ],
  "rejected_candidates": [
    {{
      "entity_id": "string",
      "entity_name": "string",
      "reason": "short explanation"
    }}
  ]
}}
""".strip()


def _aggregate_matches(all_matches: List[dict]) -> List[dict]:
    """
    Merge candidates across multiple retrieval views.

    Ranking preference:
    - higher support_count across views
    - higher best reranker/search score
    """
    by_id: Dict[str, dict] = {}

    for m in all_matches:
        entity_id = m.get("entity_id")
        if not entity_id:
            continue

        score = m.get("@search.reranker_score", m.get("@search.score", 0.0))

        if entity_id not in by_id:
            by_id[entity_id] = {
                "doc": m,
                "best_score": score,
                "support_count": 1,
            }
        else:
            by_id[entity_id]["best_score"] = max(by_id[entity_id]["best_score"], score)
            by_id[entity_id]["support_count"] += 1

            # keep the best scoring doc instance
            current_best_doc = by_id[entity_id]["doc"]
            current_best_score = current_best_doc.get(
                "@search.reranker_score",
                current_best_doc.get("@search.score", 0.0),
            )
            if score > current_best_score:
                by_id[entity_id]["doc"] = m

    ranked = sorted(
        by_id.values(),
        key=lambda x: (x["support_count"], x["best_score"]),
        reverse=True,
    )

    docs = []
    for item in ranked:
        doc = dict(item["doc"])
        doc["_support_count"] = item["support_count"]
        doc["_aggregated_best_score"] = item["best_score"]
        docs.append(doc)

    return docs


def select_value_streams(
    query: str,
    fetch_count: int = 12,
) -> dict:
    """
    Build multiple retrieval views from the PPT text, retrieve candidates per view,
    aggregate them, and ask the LLM to select only from those candidates.
    """
    client = AzureDirectSearchClient()
    gen_svc = GenerationService()

    retrieval_views = _build_retrieval_views(query)

    logging.info("Built %d retrieval views", len(retrieval_views))

    all_matches: List[dict] = []

    for idx, view in enumerate(retrieval_views, 1):
        logging.info("Running hybrid retrieval for view %d", idx)
        hybrid_matches = client.search_hybrid(view, top_k=fetch_count)
        if hybrid_matches:
            all_matches.extend(hybrid_matches)

        logging.info("Running vector retrieval for view %d", idx)
        vector_matches = client.search_vector(view, top_k=max(6, fetch_count // 2))
        if vector_matches:
            all_matches.extend(vector_matches)

    if not all_matches:
        return {
            "selected_value_streams": [],
            "rejected_candidates": [],
            "raw_response": None,
            "candidates_used": [],
            "retrieval_views": retrieval_views,
        }

    aggregated_matches = _aggregate_matches(all_matches)
    candidate_pool = aggregated_matches[:15]

    context = _build_context(candidate_pool)

    # Use the cleaned overall text as the user content in the prompt, not the full noisy blob
    cleaned_query = _clean_ppt_text(query)
    prompt_query = _normalize_for_search(cleaned_query, max_chars=3500)

    prompt = _build_value_stream_selection_prompt(query=prompt_query, context=context)

    reply = gen_svc.generate(query=prompt, context="")
    parsed = _safe_json_extract(reply.content)

    return {
        "selected_value_streams": parsed.get("selected_value_streams", []),
        "rejected_candidates": parsed.get("rejected_candidates", []),
        "raw_response": reply.content,
        "candidates_used": [
            {
                "entity_id": m.get("entity_id", ""),
                "entity_name": m.get("entity_name", ""),
                "@search.score": m.get("@search.score", 0),
                "@search.reranker_score": m.get("@search.reranker_score", 0),
                "_support_count": m.get("_support_count", 1),
                "_aggregated_best_score": m.get("_aggregated_best_score", 0),
            }
            for m in candidate_pool
        ],
        "retrieval_views": retrieval_views,
    }


def generate_value_streams(
    query: str,
    fetch_count: int = 12,
) -> str:
    """
    Kept old function name for drop-in compatibility.
    Now returns selected EXISTING value streams, not a generated new one.
    """
    result = select_value_streams(query=query, fetch_count=fetch_count)
    return json.dumps(
        {
            "selected_value_streams": result["selected_value_streams"],
            "rejected_candidates": result["rejected_candidates"],
        },
        indent=2,
    )


def run_generate(query: str, top_k: int = 12) -> None:
    print(f"\n[Generate] {query[:120]}\n{'-' * 60}")

    result = select_value_streams(query=query, fetch_count=top_k)

    print("\nRetrieval Views:")
    for i, view in enumerate(result["retrieval_views"], 1):
        print(f"\n--- View {i} ---")
        print(view[:800] + ("..." if len(view) > 800 else ""))

    print("\nSelected Value Streams:")
    print(json.dumps(result["selected_value_streams"], indent=2))

    print("\nRejected Candidates:")
    print(json.dumps(result["rejected_candidates"], indent=2))

    print("\nCandidates Used:")
    print(json.dumps(result["candidates_used"], indent=2))

    print("\nRaw LLM Response:")
    print(result["raw_response"])


def main() -> None:
    value_stream_service = ValueStreamService()
    safe_text = value_stream_service.get_ppt_text(doc_id="IDMT-19761")

    print("\n" + "=" * 80)
    print("RAW INPUT PREVIEW")
    print("=" * 80)
    print(safe_text[:3000])

    print("\n" + "=" * 80)
    print("CLEANED INPUT PREVIEW")
    print("=" * 80)
    print(_clean_ppt_text(safe_text)[:3000])

    print("\n" + "=" * 80)
    print("GENERATED VALUE STREAM SELECTION")
    print("=" * 80)

    generated_vs = generate_value_streams(
        query=safe_text,
        fetch_count=12,
    )

    print(generated_vs)


if __name__ == "__main__":
    main()