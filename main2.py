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