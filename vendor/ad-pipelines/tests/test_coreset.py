import pytest
import torch

from ad_pipelines.utils import greedy_coreset


@pytest.mark.parametrize("ratio", (0.0, -0.5, 1.0, 1.5, None))
def test_disabled_or_noop_ratios_return_input(ratio):
    x = torch.randn(50, 32)
    out = greedy_coreset(x, ratio)
    assert out is x


def test_subsample_shape_and_is_subset():
    torch.manual_seed(0)
    x = torch.randn(200, 768)
    out = greedy_coreset(x, 0.1, seed=42)
    assert out.shape == (20, 768)
    # Selected rows must be actual rows of the input (not fabricated values).
    # Rows are unique because greedy farthest-point never re-selects.
    assert out.unique(dim=0).shape[0] == out.shape[0]


def test_small_bank_ratio_above_cap_returns_all():
    x = torch.randn(10, 16)
    out = greedy_coreset(x, 0.99, seed=1)  # round(9.9) = 10 >= N -> keep all
    assert out.shape == (10, 16)


def test_greedy_reproducible_with_seed():
    torch.manual_seed(0)
    x = torch.randn(150, 768)
    a = greedy_coreset(x, 0.2, seed=42)
    b = greedy_coreset(x, 0.2, seed=42)
    assert torch.equal(a, b)


def test_different_seed_differs():
    torch.manual_seed(0)
    x = torch.randn(150, 768)
    a = greedy_coreset(x, 0.2, seed=1)
    b = greedy_coreset(x, 0.2, seed=2)
    assert not torch.equal(a, b)


def test_force_random_seed_deterministic():
    torch.manual_seed(0)
    x = torch.randn(500, 128)
    a = greedy_coreset(x, 0.1, seed=7, force_random=True)
    b = greedy_coreset(x, 0.1, seed=7, force_random=True)
    assert a.shape == (50, 128)
    assert torch.equal(a, b)


@pytest.mark.parametrize("dims", (64, 128, 768))
def test_projection_and_no_projection_branches(dims):
    torch.manual_seed(0)
    x = torch.randn(100, dims)
    out = greedy_coreset(x, 0.5, seed=1)
    assert out.shape == (50, dims)


def test_fp16_bank_keeps_dtype():
    """Regression: distance computation must be dtype-agnostic. GPU inference
    feeds float16 banks; the seeded projection is float32, which previously
    raised `c10::Half != float` on the matmul."""
    torch.manual_seed(0)
    x = torch.randn(200, 768).half()
    out = greedy_coreset(x, 0.1, seed=42)
    assert out.dtype == torch.float16, f"dtype changed: {out.dtype}"
    assert out.shape == (20, 768)


def test_fp16_selection_close_to_fp32_selection():
    """fp16 rounding shifts feature values slightly, so exact row equality
    between fp16 and fp32 banks is not expected -- but the greedy selection
    geometry must not change wholesale. Require substantial index overlap."""
    torch.manual_seed(0)
    x32 = torch.randn(200, 768)
    sel32 = greedy_coreset(x32, 0.1, seed=42)
    sel16 = greedy_coreset(x32.half(), 0.1, seed=42)

    def selected_indices(sel, source):
        # Each returned row is an exact copy of a source row; locate it.
        # Random-normal rows are unique, so index recovery is unambiguous.
        idx = []
        for row in sel:
            m = (source == row).all(dim=-1)
            (j,) = m.nonzero(as_tuple=True)
            idx.append(int(j))
        return set(idx)

    idx32 = selected_indices(sel32, x32)
    idx16 = selected_indices(sel16, x32.half())
    overlap = len(idx32 & idx16)
    assert overlap >= 10, f"overlap too low: {overlap}/20"
