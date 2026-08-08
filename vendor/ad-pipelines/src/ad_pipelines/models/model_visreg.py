from os import PathLike
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import timm

from .attention_utils import AttentionInterestTokenIdx, compute_interest_token_attention_logits
from .config_base import ViTConfig
from .model_base import BaseModel, VisionEncoderOutput


def _resolve_visreg_checkpoint(model_path: Union[str, PathLike]) -> Path:
    path = Path(model_path).expanduser()
    if path.is_file():
        return path

    if path.is_dir():
        checkpoints = sorted(path.glob("visreg-vit-*-inet1k.pth"))
        if len(checkpoints) == 1:
            return checkpoints[0]
        if len(checkpoints) > 1:
            raise ValueError(
                f"Multiple VisReg checkpoints found in {path}. Pass one .pth file explicitly."
            )

    raise FileNotFoundError(
        f"VisReg checkpoint not found at {path}. Expected a .pth file or a directory "
        "containing exactly one visreg-vit-*-inet1k.pth checkpoint."
    )


def _infer_timm_model_name(checkpoint_path: Path) -> str:
    name = checkpoint_path.name.lower()
    if "vit-b" in name or "vit_base" in name:
        return "vit_base_patch16_224"
    if "vit-l" in name or "vit_large" in name:
        return "vit_large_patch14_224"
    raise ValueError(
        f"Cannot infer VisReg architecture from {checkpoint_path.name}. "
        "Use a filename like visreg-vit-b-inet1k.pth or visreg-vit-l-inet1k.pth."
    )


def _image_to_tensor(
    image: Union[Image.Image, torch.Tensor, np.ndarray],
    do_rescale: Optional[bool],
) -> torch.Tensor:
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    elif isinstance(image, torch.Tensor):
        if image.ndim != 3:
            raise ValueError("Tensor images must have shape [C, H, W].")
        tensor = image.float()
        if do_rescale is not False and tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor
    elif not isinstance(image, np.ndarray):
        raise TypeError(f"Unsupported image type: {type(image)!r}")

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.ndim != 3:
        raise ValueError("NumPy images must have shape [H, W] or [H, W, C].")
    if image.shape[2] == 4:
        image = image[:, :, :3]
    tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float()
    if do_rescale is not False and tensor.max() > 1.0:
        tensor = tensor / 255.0
    return tensor


def _normalize_tensor_images(
    images: Union[Image.Image, torch.Tensor, np.ndarray, list],
    do_rescale: Optional[bool],
) -> Union[torch.Tensor, List[torch.Tensor]]:
    if isinstance(images, torch.Tensor):
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError("Tensor images must have shape [C, H, W] or [B, C, H, W].")
        images = images.float()
        if do_rescale is not False and images.max() > 1.0:
            images = images / 255.0
        return images

    if isinstance(images, (Image.Image, np.ndarray)):
        images = [images]
    return [_image_to_tensor(image, do_rescale) for image in images]


class VisRegModel(BaseModel):
    def __init__(
        self,
        model_path: Union[str, PathLike],
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        **kwargs,
    ):
        super().__init__(model_path, device, dtype, resolution, **kwargs)

        self.model_type = "visreg"
        self.checkpoint_path = None
        self.processor = None

        self.load_model(self.model_path, **kwargs)

    def load_model(self, model_path: Union[str, PathLike], **kwargs) -> None:
        self.checkpoint_path = _resolve_visreg_checkpoint(model_path)
        model_name = kwargs.pop("timm_model_name", None) or _infer_timm_model_name(
            self.checkpoint_path
        )

        self.model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True,
            dynamic_img_pad=True,
        )
        state_dict = torch.load(self.checkpoint_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = {
            self._clean_state_dict_key(key): value for key, value in state_dict.items()
        }
        # The checkpoints include an ImageNet classifier head, while the AD
        # backbone intentionally uses timm's feature-only ``num_classes=0`` model.
        state_dict = {
            key: value for key, value in state_dict.items() if not key.startswith("head.")
        }
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(device=self.device, dtype=self.dtype)
        self.initialize_config(**kwargs)

        if self.resolution is None:
            self.resolution = self.config.resolution

    def initialize_config(self, **kwargs) -> None:
        patch_size = int(self.model.patch_embed.patch_size[0])
        img_size = self.model.patch_embed.img_size
        self.config = ViTConfig(
            resolution=(int(img_size[0]), int(img_size[1])),
            hidden_size=int(self.model.embed_dim),
            head_dim=int(self.model.blocks[0].attn.head_dim),
            patch_size=patch_size,
            num_layers=len(self.model.blocks),
            num_heads=int(self.model.blocks[0].attn.num_heads),
            is_cls_token=True,
            num_register_tokens=max(
                int(getattr(self.model, "num_prefix_tokens", 1)) - 1,
                0,
            ),
            **kwargs,
        )
        self.patch_size = self.config.patch_size
        self.feature_dim = self.config.hidden_size

    @staticmethod
    def _clean_state_dict_key(key: str) -> str:
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        return key

    def preprocess(
        self,
        images: Union[Image.Image, torch.Tensor, np.ndarray, list],
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        do_normalize: Optional[bool] = None,
        do_resize: Optional[bool] = None,
        do_rescale: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        if resolution is None:
            resolution = self.resolution

        image_tensors = _normalize_tensor_images(images, do_rescale)
        if resolution is not None and do_resize is not False:
            if isinstance(resolution, int):
                size = (resolution, resolution)
            else:
                size = (int(resolution[0]), int(resolution[1]))

            def resize_tensor(tensor: torch.Tensor) -> torch.Tensor:
                return F.interpolate(
                    tensor.unsqueeze(0),
                    size=size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            if isinstance(image_tensors, list):
                image_tensors = torch.stack([resize_tensor(tensor) for tensor in image_tensors])
            else:
                image_tensors = F.interpolate(
                    image_tensors,
                    size=size,
                    mode="bilinear",
                    align_corners=False,
                )
        elif isinstance(image_tensors, list):
            image_tensors = torch.stack(image_tensors)

        if do_normalize is not False:
            mean = image_tensors.new_tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
            std = image_tensors.new_tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)
            image_tensors = (image_tensors - mean) / std

        return image_tensors

    def _attention_forward(
        self,
        attn_module: nn.Module,
        hidden_states: torch.Tensor,
        output_attentions: bool,
        attention_interest_token_idx: AttentionInterestTokenIdx,
        num_prefix_tokens: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = attn_module.qkv(hidden_states).reshape(
            batch_size,
            seq_len,
            3,
            attn_module.num_heads,
            attn_module.head_dim,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query, key = attn_module.q_norm(query), attn_module.k_norm(key)

        attentions = None
        if output_attentions:
            attentions = compute_interest_token_attention_logits(
                query,
                key,
                attention_interest_token_idx,
                num_prefix_tokens,
                float(attn_module.scale),
            )

        if getattr(attn_module, "fused_attn", False):
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=attn_module.attn_drop.p if attn_module.training else 0.0,
            )
        else:
            attention_probs = (query * attn_module.scale) @ key.transpose(-2, -1)
            attention_probs = attention_probs.softmax(dim=-1)
            attention_probs = attn_module.attn_drop(attention_probs)
            output = attention_probs @ value

        output = output.transpose(1, 2).reshape(batch_size, seq_len, attn_module.attn_dim)
        output = attn_module.norm(output)
        output = attn_module.proj(output)
        output = attn_module.proj_drop(output)
        return output, attentions

    def _block_forward(
        self,
        block: nn.Module,
        hidden_states: torch.Tensor,
        output_attentions: bool,
        attention_interest_token_idx: AttentionInterestTokenIdx,
        num_prefix_tokens: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_output, attentions = self._attention_forward(
            block.attn,
            block.norm1(hidden_states),
            output_attentions,
            attention_interest_token_idx,
            num_prefix_tokens,
        )
        hidden_states = hidden_states + block.drop_path1(block.ls1(attn_output))
        hidden_states = hidden_states + block.drop_path2(block.ls2(block.mlp(block.norm2(hidden_states))))
        return hidden_states, attentions

    def get_features(
        self,
        image_tensors: torch.Tensor,
        return_attentions: bool = False,
        return_layer_features: bool = False,
        output_feature_maps_indices: Optional[Sequence[int]] = None,
        attention_interest_token_idx: AttentionInterestTokenIdx = None,
        **kwargs,
    ) -> VisionEncoderOutput:
        if output_feature_maps_indices is None:
            output_feature_maps_indices = (-1,)
        else:
            output_feature_maps_indices = tuple(output_feature_maps_indices)

        if not output_feature_maps_indices:
            raise ValueError("At least one feature map index must be specified.")

        image_tensors = image_tensors.to(device=self.device, dtype=self.dtype)
        batch_size = image_tensors.shape[0]

        num_layers = len(self.model.blocks)
        normalized_indices = set(
            idx if idx >= 0 else (num_layers + 1) + idx
            for idx in output_feature_maps_indices
        )
        if any(idx < 0 or idx > num_layers for idx in normalized_indices):
            raise ValueError(
                f"output_feature_maps_indices must be in [0, {num_layers}] or negative equivalents."
            )

        patch_tokens = self.model.patch_embed(image_tensors)
        patch_h, patch_w = patch_tokens.shape[1], patch_tokens.shape[2]
        hidden_states = self.model._pos_embed(patch_tokens)
        hidden_states = self.model.patch_drop(hidden_states)
        hidden_states = self.model.norm_pre(hidden_states)

        feature_maps = ()
        prefix_tokens = int(getattr(self.model, "num_prefix_tokens", 1))
        if 0 in normalized_indices:
            feature_maps += (
                hidden_states[:, prefix_tokens:, :].transpose(1, 2).reshape(
                    batch_size, -1, patch_h, patch_w
                ),
            )

        attentions = () if return_attentions else None
        last_idx = num_layers - 1
        for idx, block in enumerate(self.model.blocks):
            hidden_states, attention_logits = self._block_forward(
                block,
                hidden_states,
                output_attentions=return_attentions and idx == last_idx,
                attention_interest_token_idx=attention_interest_token_idx,
                num_prefix_tokens=prefix_tokens,
            )

            if return_attentions and attention_logits is not None:
                attentions = attentions + (attention_logits,)

            if (idx + 1) in normalized_indices:
                feature_map_states = self.model.norm(hidden_states)
                feature_maps += (
                    feature_map_states[:, prefix_tokens:, :].transpose(1, 2).reshape(
                        batch_size, -1, patch_h, patch_w
                    ),
                )

        final_states = self.model.norm(hidden_states)
        cls_token = final_states[:, 0]
        register_tokens = final_states[:, 1:prefix_tokens] if prefix_tokens > 1 else None

        return VisionEncoderOutput(
            pooler_output=cls_token,
            feature_maps=feature_maps,
            attentions=attentions,
            registers=register_tokens,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_feature_maps_indices: Optional[Sequence[int]] = None,
        output_attentions: bool = False,
        attention_interest_token_idx: AttentionInterestTokenIdx = None,
    ) -> VisionEncoderOutput:
        return self.get_features(
            pixel_values,
            return_attentions=output_attentions,
            output_feature_maps_indices=output_feature_maps_indices,
            attention_interest_token_idx=attention_interest_token_idx,
        )

    def __call__(
        self,
        images: Union[torch.Tensor, Image.Image, np.ndarray, list],
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        do_normalize: Optional[bool] = None,
        do_resize: Optional[bool] = None,
        return_attentions: bool = False,
        output_feature_maps_indices: Optional[Sequence[int]] = None,
        attention_interest_token_idx: AttentionInterestTokenIdx = None,
        **kwargs,
    ) -> VisionEncoderOutput:
        model_inputs = self.preprocess(
            images,
            resolution=resolution,
            do_normalize=do_normalize,
            do_resize=do_resize,
            **kwargs,
        ).to(device=self.device, dtype=self.dtype)

        return self.get_features(
            model_inputs,
            return_attentions=return_attentions,
            output_feature_maps_indices=output_feature_maps_indices,
            attention_interest_token_idx=attention_interest_token_idx,
        )
