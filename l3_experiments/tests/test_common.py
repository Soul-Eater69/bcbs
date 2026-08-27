import pytest
from common import parse_json_response, score_sets, validate_l3_response


def test_parse_fenced_json():
    assert parse_json_response('```json\n{"l3": []}\n```') == {"l3": []}


def test_non_candidate_rejected():
    with pytest.raises(ValueError, match="not a supplied candidate"):
        validate_l3_response({"l3":[{"capability_id":"CAP9","reason":"x"}]},{"CAP1"})


def test_partial_overlap_metrics():
    m=score_sets({"A","B","X"},{"A","B","C"})
    assert m["precision"] == pytest.approx(2/3)
    assert m["recall"] == pytest.approx(2/3)
    assert m["f1"] == pytest.approx(2/3)
