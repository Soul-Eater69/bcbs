from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Tuple
import difflib


@dataclass
class Node:
    id: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AugmentationMetrics:
    """
    Attribute-level metrics for a single query.
    Input:  gt_nodes  — same nodes with the attributes a human says are relevant
            aug_nodes — same nodes after your augmentation pipeline ran

    Contains TWO aggregation strategies across nodes within this query:
      - micro    : pool raw TP/FP/FN across all nodes, compute once
      - weighted : weight each node's P and R by its GT attribute count,
                   nodes with more expected attributes carry more influence
    """
    # Raw counts (used by micro)
    attr_tp: int      # attribute expected and present with correct value
    attr_fp: int      # attribute added by augmentation but not expected
    attr_fn: int      # attribute expected but missing from augmentation

    # Micro-derived (from pooled counts)
    precision: float
    recall: float
    f1: float

    # Weighted average across nodes in this query
    # weight = number of GT attributes on that node / total GT attributes in query
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float

    # Total GT attribute count across all nodes in this query (= sum of weights denominator)
    total_gt_attrs: int

    # Per-node breakdown for debugging
    # node_id → {tp, fp, fn, precision, recall, f1, gt_attr_count, weight}
    per_node: Dict[str, Dict]

    def to_dict(self):
        return asdict(self)


@dataclass
class AggregatedAugmentationMetrics:
    num_queries: int

    # Micro (pool TP/FP/FN across all queries, compute once)
    micro_precision: float
    micro_recall: float
    micro_f1: float
    total_attr_tp: int
    total_attr_fp: int
    total_attr_fn: int

    # Macro (average P and R per query, derive F1 from those averages)
    macro_precision: float
    macro_recall: float
    macro_f1: float     # _f1(macro_P, macro_R) — NOT average of per-query F1s

    # Weighted across queries
    # Each query is weighted by its total GT attribute count.
    # Queries with more expected attributes carry more influence —
    # consistent with how weighted avg works within a single query.
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    total_gt_attrs: int   # sum of all GT attr counts across all queries (denominator)

    def to_dict(self):
        return asdict(self)

def _safe_div(n: float, d: float) -> float:
    return n / d if d > 0 else 0.0

def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def _index(nodes: List[Node]) -> Dict[str, Node]:
    return {n.id: n for n in nodes}

def _normalize(val: Any) -> str:
    return str(val).strip().lower()

def _fuzzy_match(a: str, b: str, threshold: float) -> bool:
    return difflib.SequenceMatcher(None, a, b).ratio() >= threshold


def _compare_attrs(
    gt_attrs: Dict[str, Any],
    aug_attrs: Dict[str, Any],
    key_only: bool,
    fuzzy: bool,
    threshold: float,
) -> Tuple[int, int, int]:
    """
    Returns (tp, fp, fn) comparing augmented attributes against ground truth.

    key_only=True  → only checks whether the right attribute KEYS were added
    key_only=False → checks that keys AND values are correct
    fuzzy=True     → value comparison uses similarity instead of exact match
    """
    if key_only:
        gt   = set(gt_attrs.keys())
        aug  = set(aug_attrs.keys())
        return len(gt & aug), len(aug - gt), len(gt - aug)

    if not fuzzy:
        gt   = set((_normalize(k), _normalize(v)) for k, v in gt_attrs.items())
        aug  = set((_normalize(k), _normalize(v)) for k, v in aug_attrs.items())
        return len(gt & aug), len(aug - gt), len(gt - aug)

    # Fuzzy: key must match exactly, value compared with similarity
    tp = fp = fn = 0
    for k, gt_val in gt_attrs.items():
        if k in aug_attrs:
            if _fuzzy_match(_normalize(gt_val), _normalize(aug_attrs[k]), threshold):
                tp += 1
            else:
                # Key present, wrong value → augmentation produced bad value
                fp += 1
                fn += 1
        else:
            fn += 1  # key entirely missing

    for k in aug_attrs:
        if k not in gt_attrs:
            fp += 1  # augmentation added a key that wasn't expected

    return tp, fp, fn


def evaluate_augmentation(
    gt_nodes: List[Node],
    aug_nodes: List[Node],
    key_only: bool = False,
    fuzzy: bool = False,
    fuzzy_threshold: float = 0.85,
) -> AugmentationMetrics:
    """
    Evaluate augmentation quality for a single query.

    Parameters
    ----------
    gt_nodes   : ground-truth nodes with expected attributes for this query
    aug_nodes  : the same nodes after augmentation (attributes added by your system)
    key_only   : True  → only check if the right attribute KEYS were selected
                 False → check keys + values
    fuzzy      : True  → use string similarity for value matching
    fuzzy_threshold : similarity threshold (0-1) when fuzzy=True

    Notes
    -----
    - Nodes in aug_nodes that have no matching gt node are skipped
      (retrieval errors are handled elsewhere, not here)
    - Nodes in gt_nodes with no matching aug node contribute FN counts
      for all their expected attributes
    """
    gt_idx  = _index(gt_nodes)
    aug_idx = _index(aug_nodes)

    total_tp = total_fp = total_fn = 0
    per_node: Dict[str, Dict] = {}

    for nid, gt_node in gt_idx.items():
        if nid in aug_idx:
            tp, fp, fn = _compare_attrs(
                gt_node.attributes,
                aug_idx[nid].attributes,
                key_only, fuzzy, fuzzy_threshold,
            )
        else:
            # Node was not augmented at all → every expected attribute is a miss
            tp, fp, fn = 0, 0, len(gt_node.attributes)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        p = _safe_div(tp, tp + fp)
        r = _safe_div(tp, tp + fn)
        gt_attr_count = len(gt_node.attributes)
        per_node[nid] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": _f1(p, r),
            "gt_attr_count": gt_attr_count,   # raw weight for this node
        }

    # ── Micro metrics (from pooled counts) ────────────────────────────
    precision = _safe_div(total_tp, total_tp + total_fp)
    recall    = _safe_div(total_tp, total_tp + total_fn)

    # ── Weighted average across nodes ─────────────────────────────────
    # Weight = GT attribute count of each node.
    # Intuition: a node with 4 expected attributes should influence the
    # query score 4× more than a node with 1 expected attribute.
    # Formula:
    #   weighted_P = Σ (node_P × node_gt_count) / Σ node_gt_count
    #   weighted_R = Σ (node_R × node_gt_count) / Σ node_gt_count
    total_gt_attrs = sum(v["gt_attr_count"] for v in per_node.values())

    if total_gt_attrs > 0:
        weighted_p = sum(
            v["precision"] * v["gt_attr_count"] for v in per_node.values()
        ) / total_gt_attrs
        weighted_r = sum(
            v["recall"] * v["gt_attr_count"] for v in per_node.values()
        ) / total_gt_attrs
    else:
        weighted_p = weighted_r = 0.0

    # Attach computed weight (fraction of total GT attrs) to each node entry
    for nid in per_node:
        per_node[nid]["weight"] = _safe_div(
            per_node[nid]["gt_attr_count"], total_gt_attrs
        )

    return AugmentationMetrics(
        attr_tp=total_tp, attr_fp=total_fp, attr_fn=total_fn,
        precision=precision, recall=recall, f1=_f1(precision, recall),
        weighted_precision=weighted_p, weighted_recall=weighted_r,
        weighted_f1=_f1(weighted_p, weighted_r),
        total_gt_attrs=total_gt_attrs,
        per_node=per_node,
    )


def evaluate_augmentation_all(
    queries: List[Tuple[List[Node], List[Node]]],
    key_only: bool = False,
    fuzzy: bool = False,
    fuzzy_threshold: float = 0.85,
    verbose: bool = False,
) -> Tuple[List[AugmentationMetrics], AggregatedAugmentationMetrics]:
    """
    Evaluate augmentation across many queries.

    Parameters
    ----------
    queries : list of (gt_nodes, aug_nodes) tuples

    Returns
    -------
    per_query : AugmentationMetrics for each query
    aggregated : micro + macro averages
    """
    if not queries:
        empty = AggregatedAugmentationMetrics(
            num_queries=0,
            micro_precision=0.0, micro_recall=0.0, micro_f1=0.0,
            total_attr_tp=0, total_attr_fp=0, total_attr_fn=0,
            macro_precision=0.0, macro_recall=0.0, macro_f1=0.0,
            weighted_precision=0.0, weighted_recall=0.0, weighted_f1=0.0,
            total_gt_attrs=0,
        )
        return [], empty

    per_query: List[AugmentationMetrics] = []

    # Micro accumulators
    total_tp = total_fp = total_fn = 0

    # Macro accumulators (P and R only — F1 derived from their means)
    sum_p = sum_r = 0.0

    # Weighted accumulators across queries
    # Each query is weighted by its total GT attribute count.
    # Formula:
    #   weighted_P = Σ (query_weighted_P × query_gt_attrs) / Σ query_gt_attrs
    #   weighted_R = Σ (query_weighted_R × query_gt_attrs) / Σ query_gt_attrs
    sum_weighted_p_times_w = 0.0   # Σ (query_weighted_P × gt_attr_count)
    sum_weighted_r_times_w = 0.0   # Σ (query_weighted_R × gt_attr_count)
    total_gt_attrs = 0             # Σ gt_attr_count  (denominator)

    for i, (gt, aug) in enumerate(queries):
        m = evaluate_augmentation(gt, aug, key_only, fuzzy, fuzzy_threshold)
        per_query.append(m)

        # Micro
        total_tp += m.attr_tp
        total_fp += m.attr_fp
        total_fn += m.attr_fn

        # Macro
        sum_p += m.precision
        sum_r += m.recall

        # Weighted — use each query's already-computed weighted P/R,
        # then weight that by the query's total GT attr count so that
        # larger queries (more expected attributes) carry more influence
        sum_weighted_p_times_w += m.weighted_precision * m.total_gt_attrs
        sum_weighted_r_times_w += m.weighted_recall    * m.total_gt_attrs
        total_gt_attrs         += m.total_gt_attrs

    n = len(queries)

    micro_p    = _safe_div(total_tp, total_tp + total_fp)
    micro_r    = _safe_div(total_tp, total_tp + total_fn)
    macro_p    = sum_p / n
    macro_r    = sum_r / n
    weighted_p = _safe_div(sum_weighted_p_times_w, total_gt_attrs)
    weighted_r = _safe_div(sum_weighted_r_times_w, total_gt_attrs)

    agg = AggregatedAugmentationMetrics(
        num_queries=n,
        micro_precision=micro_p,
        micro_recall=micro_r,
        micro_f1=_f1(micro_p, micro_r),
        total_attr_tp=total_tp,
        total_attr_fp=total_fp,
        total_attr_fn=total_fn,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=_f1(macro_p, macro_r),
        weighted_precision=weighted_p,
        weighted_recall=weighted_r,
        weighted_f1=_f1(weighted_p, weighted_r),
        total_gt_attrs=total_gt_attrs,
    )

    return per_query, agg

