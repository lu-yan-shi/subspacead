import pytest
import torch

from ad_pipelines.pipelines.pipeline_attentionad import DuoADPipeline
from ad_pipelines.pipelines.pipeline_patchiad import PatchIADPipeline


def _pipe(knn_k: int = 9, knn_temperature: float = 1.0) -> PatchIADPipeline:
    """Bare pipeline instance (skip __init__) so the method can be tested
    without a model or model weights."""
    pipe = object.__new__(PatchIADPipeline)
    pipe.all_similarity_aggregations = PatchIADPipeline.all_similarity_aggregations
    pipe.knn_k = knn_k
    pipe.knn_temperature = knn_temperature
    return pipe


def _aggregate(pipe, sims, method):
    # sims is [BS, H*W, N]; test_feature_shape is (bs, c, h, w) so H*W matches.
    bs, hw, _ = sims.shape
    h = w = int(round(hw ** 0.5))
    assert h * w == hw
    return PatchIADPipeline._aggregate_similarity_to_anomaly_map(
        pipe, sims, method, (bs, 1, h, w)
    )


def test_identity_similarity_gives_zero_anomaly():
    sims = torch.ones(1, 4, 10)
    out = _aggregate(_pipe(), sims, "knn_weighted")
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_zero_similarity_gives_one_anomaly():
    sims = torch.zeros(1, 4, 10)
    out = _aggregate(_pipe(), sims, "knn_weighted")
    assert torch.allclose(out, torch.ones_like(out), atol=1e-6)


def test_weighted_formula_matches_hand_computed():
    vals = torch.tensor([0.2, 0.6, 0.9])
    sims = vals.reshape(1, 1, 3)
    pipe = _pipe(knn_k=3, knn_temperature=1.0)
    out = _aggregate(pipe, sims, "knn_weighted")
    weights = torch.softmax(vals / 1.0, dim=0)
    expected = 1 - (vals * weights).sum()
    assert out.shape == (1, 1, 1)
    assert torch.allclose(out.squeeze(), expected, atol=1e-6)


def test_knn_k_1_equals_max():
    torch.manual_seed(0)
    sims = torch.rand(1, 4, 10)
    pipe = _pipe(knn_k=1)
    knn_out = _aggregate(pipe, sims, "knn_weighted")
    max_out = _aggregate(pipe, sims, "max")
    assert torch.allclose(knn_out, max_out, atol=1e-6)


def test_knn_k_clamped_to_bank_size():
    sims = torch.rand(1, 4, 2)  # bank width 2 < knn_k=9
    out = _aggregate(_pipe(knn_k=9), sims, "knn_weighted")
    assert out.shape == (1, 2, 2)
    assert torch.isfinite(out).all()


def test_temperature_sharpens_weights_toward_max():
    sims = torch.tensor([0.0, 0.1, 0.9]).reshape(1, 1, 3)
    out_hot = _aggregate(_pipe(knn_k=3, knn_temperature=0.1), sims, "knn_weighted")
    out_cold = _aggregate(_pipe(knn_k=3, knn_temperature=10.0), sims, "knn_weighted")
    # Hot temperature (small T) should make the anomaly closer to `1 - max`
    # (i.e. weights concentrate on the best neighbor) than a flat one.
    max_anomaly = 1 - sims.max().item()
    assert abs(out_hot.squeeze().item() - max_anomaly) < abs(out_cold.squeeze().item() - max_anomaly)


def test_unknown_aggregation_raises():
    pipe = _pipe()
    with pytest.raises(ValueError):
        _aggregate(pipe, torch.rand(1, 4, 5), "bogus")


def test_duoad_accepts_knn_weighted():
    assert "knn_weighted" in DuoADPipeline.all_similarity_aggregations
    assert "knn_weighted" in PatchIADPipeline.all_similarity_aggregations
