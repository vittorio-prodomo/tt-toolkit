import pytest
import torch


def _fake_outputs(B=2, Q=8, K=3, Hm=16, Wm=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "pred_logits": torch.randn(B, Q, K, generator=g),
        "pred_boxes": torch.rand(B, Q, 4, generator=g) * 0.5 + 0.25,
        "pred_masks": torch.randn(B, Q, Hm, Wm, generator=g),
    }


def test_probs_are_full_k_vectors_per_query(rfdetr_adapter):
    out = _fake_outputs()
    res = rfdetr_adapter.predict_with_class_vectors(out, (32, 32), n_queries=4)
    assert len(res) == 2
    assert res[0]["probs"].shape == (4, 3)
    # sigmoid, not softmax -> rows must NOT sum to 1
    assert not torch.allclose(res[0]["probs"].sum(-1), torch.ones(4))


def test_geometry_matches_stock_postprocess_for_the_top_query(rfdetr_adapter):
    """G-parity: our best query must reproduce the stock top-1 exactly.

    Stock takes argmax over the flattened (Q,K) grid; ours takes the query with
    the highest max-class probability. Whenever the maximising (query, class)
    cell is UNIQUE those select the same query, so box and mask must agree
    bit-for-bit — which continuous `torch.randn` inputs guarantee here.

    NOT a universal guarantee: under an exact tie between two queries the two
    topk calls may pick different ones (see the adapter docstring). This test
    deliberately uses untied data so it stays a real geometry check rather than
    a flaky tie-break check.
    """
    out = _fake_outputs()
    # The shared fixture's PostProcess(num_select=300) exceeds this test's tiny
    # synthetic grid (Q*K = 8*3 = 24 flattened elements) -> stock's own
    # torch.topk(..., 300, ...) would raise "selected index k out of range"
    # regardless of predict_with_class_vectors. Only the top-1 comparison
    # matters for G-parity, so scope num_select to what the grid actually has.
    rfdetr_adapter._postprocessor.num_select = 24
    stock = rfdetr_adapter.postprocess_for_inference(out, image_sizes=(32, 32))
    ours = rfdetr_adapter.predict_with_class_vectors(out, (32, 32), n_queries=8)
    for i in range(2):
        assert torch.equal(ours[i]["boxes"][0], stock[i]["boxes"][0])
        assert torch.equal(ours[i]["masks"][0], stock[i]["masks"][0])


def test_n_queries_is_clamped_to_available(rfdetr_adapter):
    out = _fake_outputs(Q=5)
    res = rfdetr_adapter.predict_with_class_vectors(out, (32, 32), n_queries=99)
    assert res[0]["probs"].shape[0] == 5


def test_masks_are_bool_at_target_size(rfdetr_adapter):
    out = _fake_outputs(Hm=8, Wm=8)
    res = rfdetr_adapter.predict_with_class_vectors(out, (24, 24), n_queries=3)
    assert res[0]["masks"].dtype == torch.bool
    assert res[0]["masks"].shape == (3, 24, 24)
