"""
EUPE ViT model wrapper for AD-Pipelines.

Wraps the EUPE DinoVisionTransformer (from ../EUPE) and exposes the same
BaseModel API used by all other models in this package.

Custom capabilities over the vanilla EUPE forward pass
-------------------------------------------------------
* **Pre-softmax attention logits** — The last encoder block is replayed
  manually so that Q @ K^T / sqrt(d) is captured *before* the softmax that
  is fused inside SDPA.  The result is returned as ``attentions`` in
  VisionEncoderOutput, matching the shape (B, num_heads, N, N) expected by
  DuoADPipeline._get_salient_map. These tensors are raw logits rather than
  normalized attention weights, so DuoAD saliency outputs are not a supported
  release feature.

* **Multi-layer feature extraction** — Any subset of block output indices can
  be requested via ``output_feature_maps_indices``.  Intermediate hidden
  states are collected during the forward pass, normalised with the model's
  final LayerNorm, and reshaped to (B, C, H', W').
"""

from __future__ import annotations

import sys
from os import PathLike
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from .config_base import ViTConfig
from .model_base import BaseModel, VisionEncoderOutput

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root of the AD-Pipelines workspace  (…/Git_Projects/AD-Pipelines)
# parents[0] = models/   parents[1] = ad_pipelines/
# parents[2] = src/      parents[3] = AD-Pipelines/
_AD_ROOT = Path(__file__).resolve().parents[3]

# Default EUPE repo location  (…/Git_Projects/EUPE)
EUPE_REPO_DIR: Path = _AD_ROOT.parent / "EUPE"

# Default ViT-B checkpoint
EUPE_VITB_WEIGHTS: Path = (
    Path.home() / "tepig" / "models" / "facebook" / "EUPE-ViT-B" / "EUPE-ViT-B.pt"
)

# ImageNet normalisation statistics (same as the EUPE example notebook)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# EUPE import helpers
# ---------------------------------------------------------------------------

def _ensure_eupe_importable(repo_dir: Path) -> None:
    """Insert *repo_dir* into sys.path so that ``import eupe`` works."""
    repo_str = str(repo_dir.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


# ---------------------------------------------------------------------------
# Custom forward helpers
# ---------------------------------------------------------------------------

def _block_forward_with_attn_logits(
    blk,
    x: torch.Tensor,
    rope,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward through one SelfAttentionBlock, returning pre-softmax logits.

    Replicates the eval-mode logic of ``SelfAttentionBlock._forward_list``
    while capturing Q @ K^T / sqrt(d_k) *before* any softmax.

    Args:
        blk:  A ``eupe.layers.block.SelfAttentionBlock`` instance.
        x:    Sequence tensor  (B, N, C).
        rope: RoPE ``(sin, cos)`` tuple returned by the model's
              ``rope_embed``, or ``None``.

    Returns:
        output:      (B, N, C)  — block output (identical to blk(x, rope))
        attn_logits: (B, heads, N, N) float32  — pre-softmax attention scores
    """
    attn_module = blk.attn

    # 1. Pre-norm
    x_norm = blk.norm1(x)

    # 2. QKV projection  (handles LinearKMaskedBias transparently)
    qkv = attn_module.qkv(x_norm)
    B, N, _ = qkv.shape
    C = attn_module.qkv.in_features          # full embed dim
    head_dim = C // attn_module.num_heads

    qkv_r = qkv.reshape(B, N, 3, attn_module.num_heads, head_dim)
    q, k, v = torch.unbind(qkv_r, dim=2)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))   # (B, heads, N, d)

    # 3. Apply RoPE (if present)
    if rope is not None:
        q, k = attn_module.apply_rope(q, k, rope)

    # 4. Pre-softmax logits in float32 for numerical stability
    scale = head_dim ** -0.5
    attn_logits = (
        torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(-2, -1))
        * scale
    )  # (B, heads, N, N)

    # 5. Attention output via SDPA (efficient; result == manual softmax+matmul)
    x_attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x_attn_out = x_attn_out.transpose(1, 2).reshape(B, N, C)

    # 6. Output projection + dropout
    x_attn_out = attn_module.proj(x_attn_out)
    x_attn_out = attn_module.proj_drop(x_attn_out)

    # 7. Residual connection + layer scale  (attention path)
    x = x + blk.ls1(x_attn_out)

    # 8. FFN path
    x = x + blk.ls2(blk.mlp(blk.norm2(x)))

    return x, attn_logits


def _forward_eupe_with_internals(
    model,
    pixel_values: torch.Tensor,
    output_feature_maps_indices: Tuple[int, ...] = (-1,),
    output_attentions: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Tuple[torch.Tensor, ...],
    Optional[torch.Tensor],
]:
    """Full EUPE forward with optional intermediate features and attention logits.

    Token layout inside EUPE (after ``prepare_tokens_with_masks``):

        position 0       : CLS token
        positions 1..n_s : storage tokens  (n_s = model.n_storage_tokens)
        positions n_s+1.. : patch tokens   (H' × W' of them)

    Args:
        model:
            A ``DinoVisionTransformer`` instance (in eval mode).
        pixel_values:
            Raw image batch  (B, C, img_H, img_W).
        output_feature_maps_indices:
            0-indexed block indices for which to collect spatial feature maps.
            Negative indexing is supported (e.g. ``-1`` = last block).
        output_attentions:
            If ``True``, capture pre-softmax attention logits from the last
            encoder block.

    Returns:
        cls_token:      (B, C)              CLS token (last block, normed)
        patch_map:      (B, C, H', W')      Patch features (last block, normed, spatial)
        storage_tokens: (B, n_s, C)         Storage / register tokens (last block, normed)
        feature_maps:   tuple[(B, C, H', W')] One map per requested index, sorted by index
        attn_logits:    (B, heads, N, N) float32, or ``None``
    """
    B = pixel_values.shape[0]

    # Tokenise: prepare_tokens_with_masks returns unflattened patch grid dims
    x, (H_p, W_p) = model.prepare_tokens_with_masks(pixel_values)
    n_st = model.n_storage_tokens
    n_blocks = len(model.blocks)
    last_block_idx = n_blocks - 1

    # Normalise requested indices to 0-based block indices
    normalized_indices: set[int] = {
        idx if idx >= 0 else n_blocks + idx
        for idx in output_feature_maps_indices
    }

    intermediate_states: dict[int, torch.Tensor] = {}
    attn_logits: Optional[torch.Tensor] = None

    for i, blk in enumerate(model.blocks):
        rope = model.rope_embed(H=H_p, W=W_p) if model.rope_embed is not None else None

        if i == last_block_idx and output_attentions:
            x, attn_logits = _block_forward_with_attn_logits(blk, x, rope)
        else:
            x = blk(x, rope)

        if i in normalized_indices:
            intermediate_states[i] = x

    # ------------------------------------------------------------------
    # Helper: apply final LayerNorm, then split and reshape
    # ------------------------------------------------------------------
    def _norm_split(h: torch.Tensor):
        """Returns (cls, storage, patch_map) all normalised."""
        if model.untie_cls_and_patch_norms:
            h_cls_stor = model.cls_norm(h[:, : n_st + 1])
            h_patch    = model.norm(h[:, n_st + 1 :])
        else:
            h_norm     = model.norm(h)
            h_cls_stor = h_norm[:, : n_st + 1]
            h_patch    = h_norm[:, n_st + 1 :]

        cls  = h_cls_stor[:, 0]               # (B, C)
        stor = h_cls_stor[:, 1:]              # (B, n_st, C)
        fmap = (
            h_patch
            .reshape(B, H_p, W_p, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )                                     # (B, C, H', W')
        return cls, stor, fmap

    # Last-block outputs (always computed)
    cls_token, storage_tokens, patch_map = _norm_split(x)

    # Collect feature maps for requested indices (sorted for determinism)
    feature_maps: list[torch.Tensor] = []
    for idx in sorted(normalized_indices):
        if idx in intermediate_states:
            _, _, fm = _norm_split(intermediate_states[idx])
            feature_maps.append(fm)

    return cls_token, patch_map, storage_tokens, tuple(feature_maps), attn_logits


# ---------------------------------------------------------------------------
# EUPEViTModel
# ---------------------------------------------------------------------------

_SUPPORTED_ARCHS = ("eupe_vitb16", "eupe_vits16", "eupe_vitt16")


class EUPEViTModel(BaseModel):
    """AD-Pipelines wrapper for the EUPE ViT model family.

    Implements the :class:`BaseModel` interface and exposes two extra APIs
    compared to the vanilla EUPE ``forward_features``:

    * **Pre-softmax attention logits** via ``return_attentions=True``.
    * **Multi-layer feature extraction** via ``output_feature_maps_indices``.

    Args:
        model_path:
            Path to the .pt checkpoint *file*, or to the *directory* that
            contains exactly one .pt file.  Defaults to the standard
            ViT-B checkpoint location (``~/tepig/models/facebook/EUPE-ViT-B``).
        model_arch:
            EUPE hub-function name.  One of ``"eupe_vitb16"`` (default),
            ``"eupe_vits16"``, ``"eupe_vitt16"``.
        eupe_repo_dir:
            Path to the cloned EUPE repository.  Defaults to ``../EUPE``
            relative to the AD-Pipelines workspace root.
        device:
            Torch device.  Defaults to CUDA if available, else CPU.
        dtype:
            Torch dtype.  Defaults to ``torch.float32``.
        resolution:
            Default resize resolution used by :meth:`preprocess` and
            :meth:`__call__`.  ``None`` means no resize.
    """

    def __init__(
        self,
        model_path: Union[str, PathLike] = EUPE_VITB_WEIGHTS,
        model_arch: str = "eupe_vitb16",
        eupe_repo_dir: Union[str, PathLike] = EUPE_REPO_DIR,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        **kwargs,
    ):
        super().__init__(model_path, device, dtype, resolution, **kwargs)

        if model_arch not in _SUPPORTED_ARCHS:
            raise ValueError(
                f"model_arch '{model_arch}' not recognised. "
                f"Supported: {_SUPPORTED_ARCHS}"
            )

        self.model_type = "eupe_vit"
        self.model_arch = model_arch
        self.eupe_repo_dir = Path(eupe_repo_dir)

        self.config    = None
        self.model     = None
        self.processor = None    # unused; kept for API parity
        self.text_model = None

        self.load_model(self.model_path, **kwargs)

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def load_model(self, model_path: Union[str, PathLike], **kwargs) -> None:
        """Load an EUPE ViT checkpoint from a local path.

        Adds the EUPE repository to ``sys.path`` (if not already present),
        imports the architecture factory from ``eupe.hub.backbones``, and
        loads the state dict from the checkpoint.

        Args:
            model_path:
                Path to the .pt file, or a directory containing one .pt file.
        """
        _ensure_eupe_importable(self.eupe_repo_dir)

        # Late import — EUPE must be on sys.path first
        from eupe.hub.backbones import (  # noqa: PLC0415
            eupe_vitb16 as _vitb16,
            eupe_vits16 as _vits16,
            eupe_vitt16 as _vitt16,
        )
        _arch_fns = {
            "eupe_vitb16": _vitb16,
            "eupe_vits16": _vits16,
            "eupe_vitt16": _vitt16,
        }

        model_path = Path(model_path)
        if model_path.name.endswith(".pt"):
            weights_path = model_path
            model_path = model_path.parent        # normalise to directory
        else:
            weights_path = model_path / f"{model_path.stem}.pt"

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: '{weights_path}'"
            )

        self.model = (
            _arch_fns[self.model_arch](
                pretrained=True,
                weights=str(weights_path),
            )
            .to(device=self.device, dtype=self.dtype)
        )

        self.initialize_config(**kwargs)

    def initialize_config(self, **kwargs) -> None:
        """Populate ``self.config`` from the loaded model's attributes."""
        m = self.model
        img_h, img_w = m.patch_embed.img_size
        self.config = ViTConfig(
            resolution=(img_h, img_w),
            hidden_size=m.embed_dim,
            head_dim=m.embed_dim // m.num_heads,
            patch_size=m.patch_size,
            num_layers=m.n_blocks,
            num_heads=m.num_heads,
            is_cls_token=True,
            # storage tokens behave like register tokens for pipeline purposes
            num_register_tokens=m.n_storage_tokens,
        )

    def preprocess(
        self,
        images: Union[Image.Image, torch.Tensor, np.ndarray, list],
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        do_normalize: Optional[bool] = None,
        do_resize: Optional[bool] = None,
        do_rescale: Optional[bool] = None,
    ) -> torch.Tensor:
        """Convert raw images to normalised float tensors.

        Applies (in order): optional resize → optional rescale [0,255]→[0,1]
        → optional ImageNet mean/std normalisation.

        Args:
            images:
                PIL Image, numpy array (H, W, C) uint8/float, torch.Tensor
                (C, H, W) or (B, C, H, W), or a list of any of the above.
            resolution:
                Target ``(H, W)`` for resizing.  Falls back to
                ``self.resolution``; skipped when both are ``None``.
            do_normalize:
                Apply ImageNet mean/std normalisation.  Defaults to ``True``.
            do_resize:
                Resize to *resolution*.  Defaults to ``True`` when *resolution*
                is provided, ``False`` otherwise.
            do_rescale:
                Scale pixel values from [0, 255] → [0, 1].  Defaults to
                ``True`` (only applied when the tensor max > 1).

        Returns:
            Batched float32 tensor  (B, C, H, W).
        """
        if resolution is None:
            resolution = self.resolution
        if resolution is not None and isinstance(resolution, int):
            resolution = (resolution, resolution)

        _do_rescale   = True  if do_rescale   is None else do_rescale
        _do_normalize = True  if do_normalize  is None else do_normalize
        _do_resize    = (resolution is not None) if do_resize is None else do_resize

        # Flatten to a list of individual images
        if isinstance(images, torch.Tensor) and images.ndim == 4:
            image_list = list(images)
        elif isinstance(images, list):
            image_list = images
        else:
            image_list = [images]

        processed: list[torch.Tensor] = []
        for img in image_list:
            # ── Convert to a float32 CHW tensor ──────────────────────────
            if isinstance(img, Image.Image):
                arr = np.array(img)
                t = torch.from_numpy(arr)
                if t.ndim == 3:
                    t = t.permute(2, 0, 1)   # HWC → CHW
                t = t.float()
            elif isinstance(img, np.ndarray):
                t = torch.from_numpy(img)
                if t.ndim == 3:
                    t = t.permute(2, 0, 1)
                t = t.float()
            elif isinstance(img, torch.Tensor):
                t = img.float()
                if t.ndim == 2:
                    t = t.unsqueeze(0)       # HW → 1HW (grayscale)
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

            # ── Rescale [0, 255] → [0, 1] ────────────────────────────────
            if _do_rescale and t.max() > 1.0:
                t = t / 255.0

            # ── Resize ───────────────────────────────────────────────────
            if _do_resize and resolution is not None:
                t = TF.resize(
                    t,
                    list(resolution),
                    interpolation=TF.InterpolationMode.BILINEAR,
                    antialias=True,
                )

            # ── Normalize ────────────────────────────────────────────────
            if _do_normalize:
                t = TF.normalize(t, mean=list(_IMAGENET_MEAN), std=list(_IMAGENET_STD))

            processed.append(t)

        return torch.stack(processed)

    def get_features(
        self,
        image_tensors: torch.Tensor,
        return_attentions: bool = False,
        output_feature_maps_indices: Optional[Sequence[int]] = None,
        **kwargs,
    ) -> VisionEncoderOutput:
        """Extract features from pre-processed image tensors.

        Args:
            image_tensors:
                Batch of normalised images  (B, C, H, W).
            return_attentions:
                If ``True``, capture pre-softmax attention logits from the
                last encoder block and return them in ``attentions``.
            output_feature_maps_indices:
                0-indexed block indices (negative supported) for which to
                return spatial ``(B, C, H', W')`` feature maps.
                Defaults to ``(-1,)`` — last block only.

        Returns:
            :class:`VisionEncoderOutput` with:

            * ``pooler_output``  — (B, C)             CLS token, last block
            * ``feature_maps``   — tuple[(B, C, H', W')] one per requested idx
            * ``attentions``     — tuple[(B, heads, N, N)] float32, or ``None``
            * ``registers``      — (B, n_storage, C)  storage tokens, last block
        """
        if output_feature_maps_indices is None:
            output_feature_maps_indices = (-1,)
        else:
            output_feature_maps_indices = tuple(output_feature_maps_indices)

        cls_token, patch_map, storage_tokens, feature_maps, attn_logits = (
            _forward_eupe_with_internals(
                self.model,
                image_tensors,
                output_feature_maps_indices=output_feature_maps_indices,
                output_attentions=return_attentions,
            )
        )

        return VisionEncoderOutput(
            pooler_output=cls_token,
            feature_maps=feature_maps,
            attentions=(attn_logits,) if attn_logits is not None else None,
            registers=storage_tokens,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_feature_maps_indices: Optional[Sequence[int]] = None,
        output_attentions: bool = False,
    ) -> VisionEncoderOutput:
        """Forward pass on pre-processed tensors.

        Args:
            pixel_values:            (B, C, H, W) normalised float tensor.
            output_feature_maps_indices: Block indices. Defaults to ``(-1,)``.
            output_attentions:       Return pre-softmax logits from last block.
        """
        if output_feature_maps_indices is None:
            output_feature_maps_indices = (-1,)
        return self.get_features(
            pixel_values,
            return_attentions=output_attentions,
            output_feature_maps_indices=tuple(output_feature_maps_indices),
        )

    def __call__(
        self,
        images: Union[torch.Tensor, Image.Image, np.ndarray, list],
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        do_normalize: Optional[bool] = None,
        do_resize: Optional[bool] = None,
        return_attentions: bool = False,
        output_feature_maps_indices: Optional[Sequence[int]] = None,
        **kwargs,
    ) -> VisionEncoderOutput:
        """End-to-end image → feature extraction.

        Preprocesses *images* (resize + normalize), then calls
        :meth:`get_features`.

        Args:
            images:       Raw images in any supported format.
            resolution:   Override resize resolution.
            do_normalize: Override ImageNet normalisation flag.
            do_resize:    Override resize flag.
            return_attentions:           Return pre-softmax attention logits.
            output_feature_maps_indices: Block indices for feature extraction.
        """
        model_inputs = self.preprocess(
            images,
            resolution=resolution,
            do_normalize=do_normalize,
            do_resize=do_resize,
        ).to(device=self.device, dtype=self.dtype)

        return self.get_features(
            model_inputs,
            return_attentions=return_attentions,
            output_feature_maps_indices=output_feature_maps_indices,
        )
