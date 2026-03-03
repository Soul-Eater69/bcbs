from typing import Any


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _compute_f1(precision: float, recall: float) -> float:
    return (2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0)


def _to_pairs(nodes: list[dict[str, Any]]) -> set[tuple[str, str]]:
    """Flatten all attribute k/v pairs across all nodes into a single set."""
    pairs = set()
    for node in nodes:
        for k, v in node.items():
            pairs.add((k.strip().lower(), str(v).strip().lower()))
    return pairs


def evaluate_augmentation_for_query(
    ground_truth: list[dict[str, Any]],
    augmented: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute precision/recall/F1 for a single query.

    Parameters
    ----------
    ground_truth : list[dict]  — expected nodes with attributes
    augmented    : list[dict]  — produced nodes with attributes

    Returns
    -------
    dict with precision, recall, f1, tp, fp, fn
    """
    gt_pairs  = _to_pairs(ground_truth)
    aug_pairs = _to_pairs(augmented)

    tp = len(gt_pairs & aug_pairs)
    fp = len(aug_pairs - gt_pairs)
    fn = len(gt_pairs - aug_pairs)

    precision = _safe_divide(tp, tp + fp)
    recall    = _safe_divide(tp, tp + fn)

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        _compute_f1(precision, recall),
        "tp":        tp,
        "fp":        fp,
        "fn":        fn,
    }

