from typing import Dict, Any, Optional, Tuple, Union
from typing_extensions import Self

import torch
import numpy as np

from PIL import Image

from ..utils import closing, safe_closing
from ..models import BaseModel, VisionEncoderOutput
from .pipeline_patchiad import PatchIADPipeline
from .pipeline_patchead import PatchEADOutput

class DuoADPipeline(PatchIADPipeline):
    """
    Public DuoAD release pipeline.

    Extends PatchIAD with the release similarity aggregation surface while keeping
    the historical attention-logit saliency path for parity with prior runs.
    """
    all_similarity_aggregations = ("max", "top1_mean", "knn_weighted")
    all_anomaly_score_methods = ("max", "top1")
    def __init__(
        self,
        model: BaseModel,
        cosine_sim_clamp: tuple[float, float] = (-1, 1),
        closing_iterations: int = 3,
        closing_kernel_size: int = 3,
        top_k: int = 10,
        similarity_aggregation: str = "max",
        temperature: Optional[float] = None,
        auto_temperature_factor: float = 4.0,
        # PatchCore-style memory-bank coreset + weighted k-NN (both optional)
        coreset_ratio: float = 1.0,
        coreset_seed: int = 42,
        knn_k: int = 9,
        knn_temperature: float = 1.0,
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        **kwargs
    ):
        """
        Initialize the DuoADPipeline.

        Args:
            model (BaseModel): The backbone model used for feature extraction.
            cosine_sim_clamp (tuple[float, float]): Range to clamp cosine similarity values. Default: (-1, 1).
            closing_iterations (int): Number of iterations for morphological closing on the anomaly map. Default: 3.
            closing_kernel_size (int): Kernel size for morphological closing. Default: 3.
            top_k (int): Number of top similar patches to consider for aggregation. Default: 10.
            similarity_aggregation (str): Method to aggregate similarity scores ("max", "top1_mean"). Default: "max".
            temperature (Optional[float]): Reserved for internal experiments.
            auto_temperature_factor (float): Reserved for internal experiments.
            resolution (Optional[Union[int, Tuple[int, int]]]): Resolution to resize images to.
            device (Optional[Union[str, torch.device]]): Device to run the model on.
            dtype (Optional[Union[str, torch.dtype]]): Data type for model computations.
        """
        super().__init__(
            model=model,
            cosine_sim_clamp=cosine_sim_clamp,
            similarity_aggregation=similarity_aggregation,
            closing_iterations=closing_iterations,
            closing_kernel_size=closing_kernel_size,
            coreset_ratio=coreset_ratio,
            coreset_seed=coreset_seed,
            knn_k=knn_k,
            knn_temperature=knn_temperature,
            resolution=resolution,
            device=device,
            dtype=dtype,
            **kwargs
        )

        self._user_temperature = temperature
        self.temperature = temperature
        self.auto_temperature_factor = auto_temperature_factor
        self.top_k = top_k

    def get_prompt_features(
        self,
        prompt_images: Union[torch.Tensor, Image.Image, np.ndarray, list],
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        augmentation: bool = False,
        return_origin_dict: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, VisionEncoderOutput]:
        """
        Extract features from prompt (normal) images.

        Args:
            prompt_images: Input images to extract features from.
            resolution (Optional[Union[int, Tuple[int, int]]]): Resolution to resize images to.
            augmentation (bool): Whether to apply augmentation during feature extraction. Default: False.
            return_origin_dict (bool): Whether to return the original feature dictionary. Default: False.

        Returns:
            Union[torch.Tensor, VisionEncoderOutput]: Extracted features.
        """
        prompt_features = super().get_prompt_features(
            prompt_images,
            resolution=resolution,
            augmentation=augmentation, 
            return_origin_dict=return_origin_dict,
            **kwargs
        )

        return prompt_features
    

    def _calculate_anomaly_score(
        self,
        anomaly_map: torch.Tensor,
        anomaly_score_method: str,
        return_attentioned_anomaly_map: bool,
        attentioned_anomaly_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attentioned_anomaly_score = None

        if anomaly_score_method == "max":
            anomaly_score = torch.amax(anomaly_map, dim=(1, 2))
            if return_attentioned_anomaly_map:
                attentioned_anomaly_score = torch.amax(attentioned_anomaly_map, dim=(1, 2))
        elif anomaly_score_method == "top1":
            # Calculate top 1% values instead of maximum
            flat_anomaly_map = anomaly_map.reshape(anomaly_map.size(0), -1)  # [BS, H*W]
            top_1_percent_count = max(1, int(0.01 * flat_anomaly_map.size(1)))
            anomaly_score = torch.topk(flat_anomaly_map, top_1_percent_count, dim=1)[0].mean(dim=1)
        
            if return_attentioned_anomaly_map:
                flat_attentioned_anomaly_map = attentioned_anomaly_map.reshape(attentioned_anomaly_map.size(0), -1)
                attentioned_anomaly_score = torch.topk(flat_attentioned_anomaly_map, top_1_percent_count, dim=1)[0].mean(dim=1)
        else:
            raise ValueError(f"anomaly_score_method must be one of {self.all_anomaly_score_methods}")

        return anomaly_score, attentioned_anomaly_score
    

    def _get_salient_map(
        self,
        shape: Tuple[int, int, int],
        cls_token: Optional[torch.Tensor]=None,
        patch_features: Optional[torch.Tensor]=None,
        attentions: Optional[torch.Tensor]=None,
        eps: float = 1e-6,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute the salient map based on the similarity between the CLS token and patch features.

        Args:
            shape (Tuple[int, int, int]): The shape of the output map (BS, H, W).
            cls_token (torch.Tensor): The CLS token feature. Shape: (BS, D). Not used in this function.
            patch_features (torch.Tensor): The patch features. Shape: (BS, N, D). Not used in this function.
            attentions (torch.Tensor): The attention maps from the model. (N_layers, BS, N_heads, N_tokens, N_tokens).
            eps (float): Small value to avoid division by zero. Default: 1e-6.
        **kwargs: Additional arguments.
                interest_token_idx (int or Tuple[int, int]): Index or range of interest tokens in the attention map. 
                    Default is 0 (CLS token).

        Returns:
            torch.Tensor: The computed salient map.
        """
        batch_size, height, width = shape
        interest_token_idx = kwargs.get("interest_token_idx", 0)  # Default to CLS token

        attention_map = attentions[-1].clone()  # Use the last layer's attention map
        del attentions # Free memory
        batch_size, num_heads, num_patches, _ = attention_map.shape
        
        # Get attention map for the [CLS] token
        if isinstance(interest_token_idx, int):
            interest_token_idx = (interest_token_idx, interest_token_idx+1)
        attention_map = attention_map[:, :, interest_token_idx[0]:interest_token_idx[1], self.num_prefix_tokens:]
        
        # Get the average attention across all heads
        attention_map = attention_map.mean(dim=1)  # Shape: [BS, N_interest_tokens, N_patches]
        # Normalize to 0~1 range
        attention_map = (attention_map - attention_map.min(dim=2, keepdim=True)[0]) / \
            (attention_map.max(dim=2, keepdim=True)[0] - attention_map.min(dim=2, keepdim=True)[0] + eps)
        attention_map = attention_map.mean(dim=1)  # Average across interest tokens
        attention_map = ((attention_map - attention_map.mean(dim=-1, keepdim=True)) + 1.0).clamp(min=0.0)
        
        attention_map = attention_map.reshape(batch_size, height, width)  # Reshape to [BS, H, W]

        return attention_map
    

    def __call__(
        self, 
        prompt_images: Union[torch.Tensor, Image.Image, np.ndarray, list],
        test_images: Optional[Union[torch.Tensor, Image.Image, np.ndarray, list]],
        is_prompt_features: bool = False,
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        cosine_sim_clamp: Optional[Tuple[float, float]] = None,
        anomaly_score_method: str = "top1", # "max" or "top1"
        return_attentioned_anomaly_map: bool = True,
        upsample_anomaly_map: bool = False,
        upsample_resolution: Optional[Union[int, Tuple[int, int]]] = None,
        closing_cls_attention: bool = True,
        similarity_aggregation: Optional[str] = None,
        eps: float = 1e-6,
        **kwargs
    ) -> PatchEADOutput:
        """
        Perform anomaly detection by comparing test images against prompt (normal) images.
        
        Args:
            prompt_images: Normal/reference images or pre-extracted features [N_patches, C] if is_prompt_features=True
            test_images: Images to be tested for anomalies
            is_prompt_features (bool): Whether prompt_images are already extracted features. Default: False
            resolution (int|Tuple[int,int]): Pipeline resolution; images will be resized to this resolution
                if specified. Default: None
            cosine_sim_clamp (Tuple[float,float]): Clamp range for cosine similarity. Default: self.cosine_sim_clamp
            anomaly_score_method (str): Method for score computation ("max" or "top1"). Default: "top1"
            return_attentioned_anomaly_map (bool): Whether to compute saliency-weighted outputs. Default: True.
            upsample_anomaly_map (bool): Whether to upsample anomaly map to image size. Default: False
            upsample_resolution (int|Tuple[int,int]]): Target resolution for upsampling anomaly map.
                If None, uses original image size. Default: None
            closing_cls_attention (bool): Whether to apply morphological closing to the attention map. Default: True.
            similarity_aggregation (Optional[str]): Override the similarity aggregation method.
            eps (float): Small value to avoid division by zero. Default: 1e-6.
            **kwargs: Additional arguments.
            
        Returns:
            PatchEADOutput: Contains anomaly_map, anomaly_score, attentioned_anomaly_map, 
              attentioned_anomaly_score, and attention_map
        """
        # Inputs check
        resolution = self.ensure_tuple_resolution(resolution or self.resolution)
        upsample_resolution = self.ensure_tuple_resolution(
            upsample_resolution or self.get_image_resolution(test_images)
        )

        anomaly_score_method = anomaly_score_method.lower()
        if anomaly_score_method not in self.all_anomaly_score_methods:
            raise ValueError(f"anomaly_score_method must be one of {self.all_anomaly_score_methods}")

        if cosine_sim_clamp is None:
            cosine_sim_clamp = self.cosine_sim_clamp

        layer_fusion_method = kwargs.get("layer_fusion_method", self.layer_fusion_method).lower()
        if layer_fusion_method not in self.all_layer_fusion_methods:
            raise ValueError(f"layer_fusion_method must be one of {self.all_layer_fusion_methods}, "
                             f"got {layer_fusion_method}")
        
        similarity_aggregation = (similarity_aggregation or self.similarity_aggregation).lower()
        if similarity_aggregation not in self.all_similarity_aggregations:
            raise ValueError(f"similarity_aggregation must be one of {self.all_similarity_aggregations}")

        extra_feature_kwargs = {}
        last_layer_idx = self.model.config.num_layers
        only_use_last_layer = True
        is_last_layer_used = True

        feature_map_indices = kwargs.get("output_feature_maps_indices", None)

        if feature_map_indices is not None:
            # Check if custom layer indices are provided
            is_last_layer_used = -1 in feature_map_indices or last_layer_idx in feature_map_indices
            only_use_last_layer = is_last_layer_used and len(feature_map_indices) == 1

            # We need last layer for CLS-Patch similarity map
            if is_last_layer_used is False:
                extra_feature_kwargs["output_feature_maps_indices"] = feature_map_indices + (-1,)
            else:
                extra_feature_kwargs["output_feature_maps_indices"] = feature_map_indices
        
        if not is_prompt_features:
            prompt_features = self.get_prompt_features(
                prompt_images,
                resolution=resolution,
                include_last_layer=is_last_layer_used,
                layer_fusion_method=layer_fusion_method,
                **extra_feature_kwargs
            )
        else:
            prompt_features = prompt_images

        # Prepare prompt features based on fusion method
        prompt_features = self._prepare_prompt_features(
            prompt_features,
            layer_fusion_method,
            is_last_layer_used
        )

        test_inputs = self.preprocess(
            test_images, 
            resolution=resolution,
        )

        test_features_dict = self.get_features(
            test_inputs,
            return_attentions=return_attentioned_anomaly_map,
            **extra_feature_kwargs
        )
        last_layer_test_features = test_features_dict.feature_maps[-1]  # Shape: [BS, C, H, W]

        bs, c, h, w = last_layer_test_features.shape

        cls_sim = None
        attentioned_anomaly_map = None
        attentioned_anomaly_score = None

        if return_attentioned_anomaly_map:
            cls_sim = self._get_salient_map(
                (bs, h, w),
                cls_token=test_features_dict.pooler_output,
                patch_features=last_layer_test_features.permute(0, 2, 3, 1).reshape(bs, h*w, c),
                attentions=test_features_dict.attentions,
                is_closing=closing_cls_attention,
                eps=eps,
            )


        # Get anomaly map according to fusion method
        anomaly_map = self._compute_anomaly_map_from_features(
            test_features_dict,
            prompt_features,
            layer_fusion_method,
            only_use_last_layer,
            is_last_layer_used,
            cosine_sim_clamp,
            similarity_aggregation,
        )
        if return_attentioned_anomaly_map:
            attentioned_anomaly_map = anomaly_map * cls_sim

        # Calculate anomaly scores
        anomaly_score, attentioned_anomaly_score = self._calculate_anomaly_score(
            anomaly_map,
            anomaly_score_method,
            return_attentioned_anomaly_map,
            attentioned_anomaly_map
        )

        if upsample_anomaly_map:
            anomaly_map = self.upsample_anomaly_map(anomaly_map, upsample_resolution)
            if return_attentioned_anomaly_map:
                attentioned_anomaly_map = self.upsample_anomaly_map(attentioned_anomaly_map, upsample_resolution)

        return PatchEADOutput(
            anomaly_map=anomaly_map.detach().cpu(),
            anomaly_score=anomaly_score.detach().cpu(),
            attentioned_anomaly_map=attentioned_anomaly_map.detach().cpu() if return_attentioned_anomaly_map else None,
            attentioned_anomaly_score=attentioned_anomaly_score.detach().cpu() if return_attentioned_anomaly_map else None,
            attention_map=cls_sim.detach().cpu() if return_attentioned_anomaly_map else None
        )
