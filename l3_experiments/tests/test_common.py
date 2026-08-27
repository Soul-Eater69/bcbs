import pytest

from common import parse_json_response, score_sets, validate_l3_response


def test_parse_fenced_json():
    assert parse_json_response('```json\n{"l3": []}\n```') == {"l3": []}


def test_non_candidate_rejected():
    with pytest.raises(ValueError, match="not a supplied candidate"):
        validate_l3_response(
            {"l3": [{"capability_id": "CAP9", "reason": "x"}]},
            {"CAP1"},
        )


def test_extra_top_level_output_field_rejected():
    with pytest.raises(ValueError, match="exactly one top-level field"):
        validate_l3_response({"l3": [], "confidence": 0.9}, {"CAP1"})


def test_extra_selection_output_field_rejected():
    payload = {
        "l3": [
            {
                "capability_id": "CAP1",
                "reason": "Directly supported by the supplied evidence.",
                "capability_name": "Should not be returned",
            }
        ]
    }
    with pytest.raises(ValueError, match="exactly capability_id and reason"):
        validate_l3_response(payload, {"CAP1"})


def test_more_than_three_selections_rejected():
    payload = {
        "l3": [
            {"capability_id": capability_id, "reason": "Direct evidence."}
            for capability_id in ("CAP1", "CAP2", "CAP3", "CAP4")
        ]
    }
    with pytest.raises(ValueError, match="at most 3"):
        validate_l3_response(payload, {"CAP1", "CAP2", "CAP3", "CAP4"})


def test_partial_overlap_metrics():
    metrics = score_sets({"A", "B", "X"}, {"A", "B", "C"})
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)


def test_both_empty_sets_are_an_exact_match():
    assert score_sets([], []) == {
        "exact_match": 1,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "predicted_count": 0,
        "truth_count": 0,
    }
