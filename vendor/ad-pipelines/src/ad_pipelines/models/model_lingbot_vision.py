from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from os import PathLike

from .attention_utils import AttentionInterestTokenIdx, compute_interest_token_attention_logits
from .config_base import ViTConfig
from .model_base import BaseModel, VisionEncoderOutput

try:
    from lingbot_vision.loader import load_pretrained_backbone
except ImportError as exc:
    raise ImportError(
        "LingBotVisionModel requires the optional `lingbot-vision` package. "
        "Install it from https://github.com/robbyant/lingbot-vision."
    ) from exc


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_LOAD_KWARGS = (
    "variant",
    "cache_dir",
    "revision",
    "local_files_only",
    "config_file",
    "checkpoint_file",
)


class LingBotVisionModel(BaseModel):
    def __init__(
        self,
        model_path: Union[str, PathLike],
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        **kwargs
    ):
        super().__init__(model_path, device, dtype, resolution, **kwargs)

        self.model_type = "lingbot_vision"
        self.config = None
        self.model = None
        self.processor = None
        self.text_model = None

        self.load_model(self.model_path, **kwargs)

    def load_model(self, model_path: Union[str, PathLike], **kwargs) -> None:
        load_kwargs = {}
        for key in _LOAD_KWARGS:
            if key in kwargs:
                load_kwargs[key] = kwargs[key]

        self.model, _ = load_pretrained_backbone(
            repo_id_or_path=str(model_path),
            device=self.device,
            dtype=self.dtype,
            verbose=False,
            **load_kwargs,
        )
        self.initialize_config(**kwargs)

    def initialize_config(self, **kwargs) -> None:
        patch_size = self.model.patch_size
        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]

        native_resolution = getattr(self.model.patch_embed, "img_size", (512, 512))
        if isinstance(native_resolution, int):
            native_resolution = (native_resolution, native_resolution)

        resolution = self._ensure_tuple_resolution(self.resolution) or native_resolution
        resolution = self._snap_resolution(resolution, int(patch_size))
        self.resolution = resolution

        self.config = ViTConfig(
            resolution=resolution,
            hidden_size=self.model.embed_dim,
            head_dim=self.model.embed_dim // self.model.num_heads,
            patch_size=int(patch_size),
            num_layers=len(self.model.blocks),
            num_heads=self.model.num_heads,
            is_cls_token=True,
            num_register_tokens=self.model.n_storage_tokens,
        )

    @staticmethod
    def _to_chw_float_tensor(image: Union[Image.Image, torch.Tensor, np.ndarray]) -> torch.Tensor:
        if isinstance(image, Image.Image):
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
        elif isinstance(image, np.ndarray):
            array = image
            if array.ndim == 2:
                array = np.repeat(array[:, :, None], 3, axis=2)
            if array.shape[-1] == 4:
                array = array[:, :, :3]
            tensor = torch.from_numpy(array).permute(2, 0, 1).float()
            if tensor.max() > 1:
                tensor = tensor / 255.0
        elif isinstance(image, torch.Tensor):
            tensor = image.detach().clone().float()
            if tensor.ndim != 3:
                raise ValueError(f"Expected a CHW/HWC image tensor, got shape {tuple(tensor.shape)}")
            if tensor.shape[0] not in (1, 3) and tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(2, 0, 1)
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            if tensor.max() > 1:
                tensor = tensor / 255.0
        else:
            raise TypeError(f"Unsupported image type: {type(image)!r}")

        if tensor.shape[0] != 3:
            raise ValueError(f"Expected 3 image channels, got {tensor.shape[0]}")
        return tensor

    @staticmethod
    def _ensure_tuple_resolution(
        resolution: Optional[Union[int, Tuple[int, int]]]
    ) -> Optional[Tuple[int, int]]:
        if resolution is None:
            return None
        if isinstance(resolution, int):
            return resolution, resolution
        return int(resolution[0]), int(resolution[1])

    @staticmethod
    def _snap_resolution(resolution: Tuple[int, int], patch_size: int) -> Tuple[int, int]:
        width, height = resolution
        return (
            max(patch_size, (width // patch_size) * patch_size),
            max(patch_size, (height // patch_size) * patch_size),
        )

    def preprocess(
        self,
        images: Union[Image.Image, torch.Tensor, np.ndarray, list],
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        do_normalize: Optional[bool] = None,
        do_resize: Optional[bool] = None,
        do_rescale: Optional[bool] = None,
    ) -> torch.Tensor:
        if resolution is None:
            resolution = self.resolution
        resolution = self._ensure_tuple_resolution(resolution)

        normalize = True if do_normalize is None else do_normalize
        rescale = True if do_rescale is None else do_rescale
        should_resize = do_resize or (do_resize is None and resolution is not None)
        if should_resize:
            resolution = self._snap_resolution(resolution, self.config.patch_size)

        if isinstance(images, torch.Tensor) and images.ndim == 4:
            tensors = images.float()
            if rescale and tensors.max() > 1:
                tensors = tensors / 255.0
        else:
            if isinstance(images, (Image.Image, np.ndarray, torch.Tensor)):
                images = [images]
            tensors = [self._to_chw_float_tensor(image) for image in images]
            if not rescale:
                tensors = [tensor * 255.0 for tensor in tensors]
            if should_resize:
                tensors = [
                    F.interpolate(
                        tensor.unsqueeze(0),
                        size=(resolution[1], resolution[0]),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    for tensor in tensors
                ]
            tensors = torch.stack(tensors)

        if isinstance(images, torch.Tensor) and images.ndim == 4 and should_resize:
            tensors = F.interpolate(
                tensors,
                size=(resolution[1], resolution[0]),
                mode="bilinear",
                align_corners=False,
            )

        if normalize:
            mean = tensors.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
            std = tensors.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
            tensors = (tensors - mean) / std

        return tensors

    def _normalize_feature_indices(
        self, output_feature_maps_indices: Optional[Sequence[int]]
    ) -> Tuple[int, ...]:
        if output_feature_maps_indices is None:
            return (-1,)
        return tuple(output_feature_maps_indices)

    def _forward_features(
        self,
        image_tensors: torch.Tensor,
        output_feature_maps_indices: Tuple[int, ...],
        return_attentions: bool = False,
        attention_interest_token_idx: AttentionInterestTokenIdx = None,
    ) -> VisionEncoderOutput:
        batch_size, _, height, width = image_tensors.shape
        patch_size = self.config.patch_size
        grid_h, grid_w = height // patch_size, width // patch_size
        num_layers = len(self.model.blocks)
        normalized_indices = set(
            idx if idx >= 0 else num_layers + 1 + idx
            for idx in output_feature_maps_indices
        )

        hidden_states, (grid_h, grid_w) = self.model.prepare_tokens_with_masks(image_tensors)
        rope = self.model.rope_embed(H=grid_h, W=grid_w) if self.model.rope_embed is not None else None

        feature_maps = ()
        if 0 in normalized_indices:
            feature_maps += (
                self._tokens_to_feature_map(hidden_states, batch_size, grid_h, grid_w, norm=True),
            )

        attentions = () if return_attentions else None
        last_layer_idx = num_layers - 1

        for i, block in enumerate(self.model.blocks):
            if return_attentions and i == last_layer_idx:
                hidden_states, attention_logits = self._forward_block_with_attention_logits(
                    block,
                    hidden_states,
                    rope,
                    attention_interest_token_idx=attention_interest_token_idx,
                )
                attentions = attentions + (attention_logits,)
            else:
                hidden_states = block(hidden_states, rope)

            if (i + 1) in normalized_indices:
                feature_maps += (
                    self._tokens_to_feature_map(hidden_states, batch_size, grid_h, grid_w, norm=True),
                )

        if self.model.untie_cls_and_patch_norms:
            cls_token = self.model.cls_norm(hidden_states[:, 0])
            patch_tokens = self.model.norm(hidden_states[:, 1 + self.model.n_storage_tokens :])
            storage_tokens = self.model.cls_norm(hidden_states[:, 1 : 1 + self.model.n_storage_tokens])
        else:
            normalized_tokens = self.model.norm(hidden_states)
            cls_token = normalized_tokens[:, 0]
            storage_tokens = normalized_tokens[:, 1 : 1 + self.model.n_storage_tokens]
            patch_tokens = normalized_tokens[:, 1 + self.model.n_storage_tokens :]

        if -1 in output_feature_maps_indices and (num_layers not in normalized_indices):
            feature_maps += (patch_tokens.transpose(1, 2).reshape(batch_size, -1, grid_h, grid_w),)

        return VisionEncoderOutput(
            pooler_output=cls_token,
            feature_maps=feature_maps,
            attentions=attentions,
            registers=storage_tokens,
        )

    def _tokens_to_feature_map(
        self,
        hidden_states: torch.Tensor,
        batch_size: int,
        grid_h: int,
        grid_w: int,
        norm: bool = False,
    ) -> torch.Tensor:
        if norm:
            if self.model.untie_cls_and_patch_norms:
                patch_tokens = self.model.norm(hidden_states[:, 1 + self.model.n_storage_tokens :])
                return patch_tokens.transpose(1, 2).reshape(batch_size, -1, grid_h, grid_w)
            hidden_states = self.model.norm(hidden_states)
        patch_tokens = hidden_states[:, 1 + self.model.n_storage_tokens :]
        return patch_tokens.transpose(1, 2).reshape(batch_size, -1, grid_h, grid_w)

    def _forward_block_with_attention_logits(
        self,
        block: nn.Module,
        hidden_states: torch.Tensor,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]],
        attention_interest_token_idx: AttentionInterestTokenIdx = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normed = block.norm1(hidden_states)
        attn = block.attn
        qkv = attn.qkv(normed)
        batch_size, num_tokens, _ = qkv.shape
        qkv = qkv.reshape(
            batch_size,
            num_tokens,
            3,
            attn.num_heads,
            attn.qkv.in_features // attn.num_heads,
        )
        query, key, value = torch.unbind(qkv, 2)
        query, key, value = [tensor.transpose(1, 2) for tensor in (query, key, value)]
        if rope is not None:
            query, key = attn.apply_rope(query, key, rope)

        attention_logits = compute_interest_token_attention_logits(
            query,
            key,
            attention_interest_token_idx,
            1 + self.model.n_storage_tokens,
            attn.scale,
        )
        attention_weights = torch.matmul(query, key.transpose(-1, -2)) * attn.scale
        attention_weights = nn.functional.softmax(attention_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attention_output = torch.matmul(attention_weights, value)
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size,
            num_tokens,
            attn.qkv.in_features,
        )
        attention_output = attn.proj_drop(attn.proj(attention_output))

        hidden_states = hidden_states + block.ls1(attention_output)
        hidden_states = hidden_states + block.ls2(block.mlp(block.norm2(hidden_states)))
        return hidden_states, attention_logits

    def get_features(
        self,
        image_tensors: torch.Tensor,
        return_attentions: bool = False,
        return_layer_features: Optional[bool] = False,
        output_feature_maps_indices: Optional[Sequence[int]] = None,
        attention_interest_token_idx: AttentionInterestTokenIdx = None,
    ) -> VisionEncoderOutput:
        output_feature_maps_indices = self._normalize_feature_indices(output_feature_maps_indices)
        image_tensors = image_tensors.to(device=self.device, dtype=self.dtype)
        return self._forward_features(
            image_tensors,
            output_feature_maps_indices=output_feature_maps_indices,
            return_attentions=return_attentions,
            attention_interest_token_idx=attention_interest_token_idx,
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
        return_attentions: bool = False,
        return_layer_features: Optional[bool] = False,
        output_feature_maps_indices: Optional[Sequence[int]] = None,
        attention_interest_token_idx: AttentionInterestTokenIdx = None,
        **kwargs
    ) -> VisionEncoderOutput:
        model_inputs = self.preprocess(
            images,
            resolution=resolution,
            do_normalize=do_normalize,
            **kwargs,
        ).to(device=self.device, dtype=self.dtype)

        return self.get_features(
            model_inputs,
            return_attentions=return_attentions,
            return_layer_features=return_layer_features,
            output_feature_maps_indices=output_feature_maps_indices,
            attention_interest_token_idx=attention_interest_token_idx,
        )
