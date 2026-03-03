from evaluation import evaluate_augmentation_for_query, evaluate_augmentation_batch


def test_perfect_match():
    gt  = [{"tier": "L1", "domain": "Claims"}]
    aug = [{"tier": "L1", "domain": "Claims"}]
    result = evaluate_augmentation_for_query(gt, aug)
    assert result["precision"] == 1.0
    assert result["recall"]    == 1.0
    assert result["f1"]        == 1.0
    assert result["tp"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 0


def test_partial_match():
    gt  = [{"tier": "L1", "domain": "Claims", "owner": "Finance"}]
    aug = [{"tier": "L1", "domain": "Claims"}]
    result = evaluate_augmentation_for_query(gt, aug)
    assert result["tp"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 1                  # "owner" missing in aug
    assert result["recall"] < 1.0


def test_wrong_value():
    gt  = [{"tier": "L1"}]
    aug = [{"tier": "L2"}]                    # wrong value
    result = evaluate_augmentation_for_query(gt, aug)
    assert result["tp"] == 0
    assert result["fp"] == 1
    assert result["fn"] == 1


def test_empty_augmented():
    gt  = [{"tier": "L1", "domain": "Claims"}]
    aug = []
    result = evaluate_augmentation_for_query(gt, aug)
    assert result["precision"] == 0.0
    assert result["recall"]    == 0.0
    assert result["fn"] == 2


def test_empty_ground_truth():
    gt  = []
    aug = [{"tier": "L1"}]
    result = evaluate_augmentation_for_query(gt, aug)
    assert result["precision"] == 0.0
    assert result["recall"]    == 0.0
    assert result["fp"] == 1


def test_duplicate_attrs_across_nodes():
    # same attr/value in both nodes — deduplication should not affect correctness
    gt  = [{"status": "active"}, {"status": "active"}]
    aug = [{"status": "active"}, {"status": "active"}]
    result = evaluate_augmentation_for_query(gt, aug)
    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 0


def test_case_insensitive():
    gt  = [{"Tier": "L1", "Domain": "CLAIMS"}]
    aug = [{"tier": "l1", "domain": "claims"}]
    result = evaluate_augmentation_for_query(gt, aug)
    assert result["tp"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 0


def test_multiple_nodes():
    gt  = [{"tier": "L1"}, {"domain": "Claims"}, {"owner": "Finance"}]
    aug = [{"tier": "L1"}, {"domain": "Claims"}]
    result = evaluate_augmentation_for_query(gt, aug)
    assert result["tp"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 1   # "owner" missing


def test_batch_aggregation():
    payloads = [
        ("q1",
         [{"tier": "L1", "domain": "Claims"}],
         [{"tier": "L1", "domain": "Claims"}]),   # perfect
        ("q2",
         [{"tier": "L2", "domain": "Finance"}],
         [{"tier": "L2"}]),                        # missing one
        ("q3",
         [{"tier": "L3"}],
         [{"tier": "L1"}]),                        # wrong value — low F1
    ]
    result = evaluate_augmentation_batch(payloads)
    micro  = result["augmentation_micro"]

    assert 0.0 < micro["precision"] <= 1.0
    assert 0.0 < micro["recall"]    <= 1.0
    assert 0.0 < micro["f1"]        <= 1.0
    assert any(q["query_id"] == "q3" for q in result["low_performing_queries"])


def test_batch_low_performing_threshold():
    payloads = [
        ("q1", [{"tier": "L1"}], [{"tier": "L2"}]),   # f1=0, should flag
        ("q2", [{"tier": "L1"}], [{"tier": "L1"}]),   # f1=1, should not flag
    ]
    result = evaluate_augmentation_batch(payloads)
    low_ids = [q["query_id"] for q in result["low_performing_queries"]]
    assert "q1" in low_ids
    assert "q2" not in low_ids


if __name__ == "__main__":
    tests = [
        test_perfect_match,
        test_partial_match,
        test_wrong_value,
        test_empty_augmented,
        test_empty_ground_truth,
        test_duplicate_attrs_across_nodes,
        test_case_insensitive,
        test_multiple_nodes,
        test_batch_aggregation,
        test_batch_low_performing_threshold,
    ]
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {test.__name__}  {e}")