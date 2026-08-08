"""Greedy coreset subsampling for prompt memory banks (PatchCore-style).

The DuoAD memory bank stores every patch feature of the prompt (normal) images,
so its size -- and the dense cosine matmul in ``_compute_patch_anomaly_scores``
(shape ``[BS, H*W, N_patches]``) -- grows linearly with the number of training
images.  PatchCore solves this with a greedy *coreset*: it keeps a small
subset of patch features (beta = 1%~10%) chosen to cover the feature space,
bounding the bank with almost no accuracy loss.

This module implements the same idea with two practical differences from the
reference ``patchcore-inspection`` sampler:

- **No full NxN distance matrix.**  We keep only a running "nearest-selected
  point" distance vector of length N, so memory is O(N) instead of O(N^2).
- **Seeded random projection.**  Distances are computed in a low-dimensional
  (default 128) randomly projected space, which only affects the *selection
  geometry*, not the returned features -- the selected rows are taken from the
  original bank.  The projection and the selection are deterministic under
  ``seed``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Above this many (m * N) distance updates we fall back to a seeded random
# subsample instead of greedy selection, to bound one-time training cost on
# very large banks.  (Greedy is O(m * N * D); this caps the worst case.)
_ADAPTIVE_RANDOM_THRESHOLD = 4e8

# Low-dimensional projection used only for distance computation during FPS.
_DEFAULT_PROJECTION_DIM = 128


def greedy_coreset(
    features: torch.Tensor,
    ratio: float,
    seed: int = 42,
    projection_dim: int = _DEFAULT_PROJECTION_DIM,
    device: Optional[torch.device] = None,
    force_random: bool = False,
) -> torch.Tensor:
    """Subsample a memory bank with greedy farthest-point sampling (coreset).

    Args:
        features: ``[N, D]`` memory bank (or a single layer's bank).
        ratio: keep this fraction of rows.  ``ratio <= 0`` or ``ratio >= 1``
            returns the input unchanged (``<= 0`` is the "disabled" sentinel).
        seed: RNG seed for deterministic selection.
        projection_dim: project features to this dim for cheaper distances.
        device: target device (defaults to ``features.device``).
        force_random: skip greedy selection and use a seeded random subsample.

    Returns:
        ``[m, D]`` subsampled bank with ``m = max(1, round(N * ratio))``.
    """
    if ratio is None or ratio <= 0.0 or ratio >= 1.0 or features.shape[0] < 2:
        return features

    N, D = features.shape
    device = device or features.device
    m = max(1, int(round(N * ratio)))
    if m >= N:
        return features

    features = features.to(device)
    g = torch.Generator(device=device).manual_seed(seed)

    if force_random or m * N > _ADAPTIVE_RANDOM_THRESHOLD:
        indices = torch.randperm(N, generator=g, device=device)[:m]
        return features[indices]

    with torch.no_grad():
        # Distance computation runs in float32 regardless of the bank's dtype
        # (banks are typically FP16 in GPU inference).  Only the selection
        # geometry is approximate -- the returned rows are original features,
        # so their dtype is preserved.
        if D > projection_dim:
            # Seeded random projection: only the selection geometry is
            # approximate; the returned rows are original features.
            proj = nn.Linear(D, projection_dim, bias=False).to(device)
            proj.weight.normal_(0.0, 1.0, generator=g)
            work = F.normalize(proj(features.float()), dim=-1)
        else:
            work = F.normalize(features.float(), dim=-1)

        # Iterative farthest-point sampling.  min_dist[i] is the distance from
        # point i to the nearest already-selected point; each step picks the
        # farthest candidate and updates the vector (O(N) memory, no NxN).
        start = int(torch.randint(N, (1,), generator=g, device=device).item())
        selected = torch.empty(m, dtype=torch.long, device=device)
        selected[0] = start
        min_dist = (work - work[start]).norm(dim=-1)

        for i in range(1, m):
            idx = int(torch.argmax(min_dist).item())
            selected[i] = idx
            d = (work - work[idx]).norm(dim=-1)
            min_dist = torch.minimum(min_dist, d)

        return features[selected]
