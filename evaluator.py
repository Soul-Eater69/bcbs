from typing import List, Dict, Any, Tuple


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator/denominator. Returns 0.0 if denominator is zero."""
    return numerator / denominator if denominator > 0 else 0.0


def _compute_f1(precision: float, recall: float) -> float:
    """Return F1 score given precision and recall."""
    return (2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0)


def evaluate_augmentation_for_query(
    ground_truth_nodes: List[Dict[str, Any]],
    augmented_nodes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute micro precision/recall/F1 for attribute augmentation on a single query.

    Parameters
    ----------
    ground_truth_nodes : list
        Expected nodes with attributes.
    augmented_nodes : list
        Nodes produced by augmentation.

    Returns
    -------
    dict
        {
            precision: float,
            recall: float,
            f1: float,
            tp: int,
            fp: int,
            fn: int
        }
    """

    gt_index = {node["id"]: node.get("attributes", {}) for node in ground_truth_nodes}
    aug_index = {node["id"]: node.get("attributes", {}) for node in augmented_nodes}

    total_tp = total_fp = total_fn = 0

    for node_id, gt_attrs in gt_index.items():
        aug_attrs = aug_index.get(node_id, {})

        gt_pairs = {
            (k.strip().lower(), str(v).strip().lower())
            for k, v in gt_attrs.items()
        }
        aug_pairs = {
            (k.strip().lower(), str(v).strip().lower())
            for k, v in aug_attrs.items()
        }

        total_tp += len(gt_pairs & aug_pairs)
        total_fp += len(aug_pairs - gt_pairs)
        total_fn += len(gt_pairs - aug_pairs)

    precision = _safe_divide(total_tp, total_tp + total_fp)
    recall = _safe_divide(total_tp, total_tp + total_fn)

    return {
        "precision": precision,
        "recall": recall,
        "f1": _compute_f1(precision, recall),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def evaluate_augmentation_batch(
    query_payloads: List[
        Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]
    ]
) -> Dict[str, Any]:
    """
    Aggregate augmentation metrics across multiple queries.

    Parameters
    ----------
    query_payloads : list
        Each item:
        (query_id, ground_truth_nodes, augmented_nodes)

    Returns
    -------
    dict
        {
            augmentation_micro: {...},
            low_performing_queries: [...]
        }
    """

    total_tp = total_fp = total_fn = 0
    low_performing_queries = []

    for query_id, gt_nodes, aug_nodes in query_payloads:

        metrics = evaluate_augmentation_for_query(gt_nodes, aug_nodes)

        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]

        if metrics["f1"] < 0.4:
            low_performing_queries.append({
                "query_id": query_id,
                "f1": metrics["f1"]
            })

    precision = _safe_divide(total_tp, total_tp + total_fp)
    recall = _safe_divide(total_tp, total_tp + total_fn)

    return {
        "augmentation_micro": {
            "precision": precision,
            "recall": recall,
            "f1": _compute_f1(precision, recall),
        },
        "low_performing_queries": low_performing_queries,
    }