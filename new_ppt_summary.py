import re
import json
from typing import Any, Dict, List

SIGNAL_TERMS = {
    "member", "provider", "employer", "patient", "care", "clinical",
    "product", "offering", "launch", "enroll", "enrollment", "billing",
    "invoice", "payment", "quote", "pricing", "network", "program",
    "engagement", "support", "workflow", "digital", "platform", "risk",
    "compliance", "authorization", "utilization", "claims", "appeal",
    "navigation", "coordination", "sales", "growth", "commercial"
}

ACTION_TERMS = {
    "launch", "improve", "reduce", "increase", "streamline", "enable",
    "automate", "support", "expand", "build", "create", "optimize",
    "establish", "manage", "onboard", "configure", "fulfill", "issue",
    "resolve", "align", "execute"
}


def extract_signal_sections(text: str, max_lines: int = 60, max_chars: int = 12000) -> str:
    """
    Pull the most business-salient lines from the PPT text so the summary step
    sees dense signal instead of the whole noisy deck.
    """
    lines = []
    for raw in (text or "").splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            continue
        if len(line) < 4:
            continue
        lines.append(line)

    scored = []
    seen = set()

    for idx, line in enumerate(lines):
        low = line.lower()
        if low in seen:
            continue
        seen.add(low)

        score = 0

        # Headings / short statements are often high signal
        if len(line) <= 90:
            score += 1

        # Bullet-ish lines
        if raw_starts_like_bullet(line):
            score += 2

        # Signal vocabulary
        score += sum(2 for term in SIGNAL_TERMS if term in low)
        score += sum(1 for term in ACTION_TERMS if term in low)

        # Numeric business lines often matter
        if re.search(r"\b\d+(\.\d+)?%?\b", line):
            score += 1

        # Penalize extremely long / likely noisy paragraphs
        if len(line) > 240:
            score -= 2

        if score > 0:
            scored.append((score, idx, line))

    # keep strongest lines, but preserve original order
    top = sorted(scored, key=lambda x: (-x[0], x[1]))[:max_lines]
    top = sorted(top, key=lambda x: x[1])

    out_lines = []
    total = 0
    for _, _, line in top:
        if total + len(line) + 1 > max_chars:
            break
        out_lines.append(line)
        total += len(line) + 1

    return "\n".join(out_lines)


def raw_starts_like_bullet(line: str) -> bool:
    return bool(re.match(r"^[-*•\d\)\(]", line))



def normalize_ppt_summary(parsed: Any, fallback_text: str) -> Dict[str, Any]:
    if not isinstance(parsed, dict):
        parsed = {}

    def _as_list(value) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        if isinstance(value, str):
            parts = [p.strip() for p in re.split(r"[,\n;|]+", value) if p.strip()]
            return parts
        return [str(value).strip()]

    summary = {
        "short_summary": str(parsed.get("short_summary") or "").strip(),
        "business_goal": str(parsed.get("business_goal") or "").strip(),
        "actors": _as_list(parsed.get("actors")),
        "direct_functions": _as_list(parsed.get("direct_functions")),
        "implied_functions": _as_list(parsed.get("implied_functions")),
        "change_types": _as_list(parsed.get("change_types")),
        "domain_tags": _as_list(parsed.get("domain_tags")),
        "evidence_sentences": _as_list(parsed.get("evidence_sentences")),
    }

    if not summary["short_summary"]:
        summary["short_summary"] = fallback_text[:300].strip()

    if not summary["business_goal"]:
        summary["business_goal"] = summary["short_summary"]

    return summary


def generate_ppt_semantic_summary(cleaned_text: str) -> Dict[str, Any]:
    """
    Create a compact semantic profile of the new PPT for retrieval.
    """
    signal_text = extract_signal_sections(cleaned_text)

    # keep this small and deterministic
    prompt = f"""
You convert noisy enterprise healthcare PPT text into a compact semantic retrieval profile.

Return VALID JSON ONLY with exactly these keys:
short_summary
business_goal
actors
direct_functions
implied_functions
change_types
domain_tags
evidence_sentences

Rules:
- Be specific, not generic.
- actors must be a JSON array of strings.
- direct_functions must contain only functions directly stated or strongly evidenced.
- implied_functions may include downstream enterprise processes that are clearly implied.
- evidence_sentences must be short verbatim or near-verbatim lines from the provided text.
- Keep short_summary to 1-2 sentences max.

PPT_TEXT:
{signal_text}
""".strip()

    reply = GenerationService().generate(
        query=prompt,
        context=""
    )

    raw = getattr(reply, "content", "") or ""
    parsed = safe_json_extract(raw)
    summary = normalize_ppt_summary(parsed, signal_text)

    summary["_signal_text"] = signal_text
    summary["_raw_summary_response"] = raw
    return summary

def build_summary_retrieval_text(summary: Dict[str, Any]) -> str:
    parts = []

    if summary.get("short_summary"):
        parts.append(summary["short_summary"])

    if summary.get("business_goal"):
        parts.append(f"business goal: {summary['business_goal']}")

    if summary.get("actors"):
        parts.append("actors: " + ", ".join(summary["actors"]))

    if summary.get("direct_functions"):
        parts.append("direct functions: " + ", ".join(summary["direct_functions"]))

    if summary.get("implied_functions"):
        parts.append("implied functions: " + ", ".join(summary["implied_functions"]))

    if summary.get("change_types"):
        parts.append("change types: " + ", ".join(summary["change_types"]))

    if summary.get("domain_tags"):
        parts.append("domain tags: " + ", ".join(summary["domain_tags"]))

    return "\n".join(parts)