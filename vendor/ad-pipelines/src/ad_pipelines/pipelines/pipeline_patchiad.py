from typing import Dict, Any, Optional, Tuple, Union
from typing_extensions import Self

import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image

from ..utils import closing, safe_closing, safe_closing_with_dilate, greedy_coreset
from ..models import BaseModel, VisionEncoderOutput
from .pipeline_patchead import PatchEADPipeline, PatchEADOutput

class PatchIADPipeline(PatchEADPipeline):
    """
    Pipeline for Patch Inclusive Anomaly Detection (PatchIAD).
    Compared to PatchEAD, PatchIAD not only use patch features in ViT, but also
    leverage the global CLS information for better anomaly detection performance.
    """
    all_layer_fusion_methods = ["score_avg", "score_max", "feature_avg", "feature_concat"]
    all_similarity_aggregations = ("max", "top1_mean", "knn_weighted")

    def __init__(
        self,
        model: BaseModel,
        cosine_sim_clamp: tuple[float, float] = (-1, 1),
        similarity_aggregation: str = "max", # max, topk_mean
        closing_iterations: int = 3,
        closing_kernel_size: int = 3,
        interest_layer_indices: Optional[Tuple[int, ...]] = None,
        layer_fusion_method: str = "score_avg", # all_layer_fusion_methods
        # PatchCore-style memory-bank coreset + weighted k-NN (both optional)
        coreset_ratio: float = 1.0,   # <=0 or >=1 disables coreset
        coreset_seed: int = 42,
        knn_k: int = 9,
        knn_temperature: float = 1.0,
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        **kwargs
    ):
        """
        Initialize the PatchIADPipeline.

        Args:
            model (BaseModel): The backbone model used for feature extraction.
            cosine_sim_clamp (tuple[float, float]): Range to clamp cosine similarity values. Default: (-1, 1).
            similarity_aggregation (str): Method to aggregate similarity scores ("max", "top1_mean"). Default: "max".
            closing_iterations (int): Number of iterations for morphological closing on the anomaly map. Default: 3.
            closing_kernel_size (int): Kernel size for morphological closing. Default: 3.
            interest_layer_indices (Optional[Tuple[int, ...]]): Indices of layers to use for feature extraction.
            layer_fusion_method (str): Method to fuse features from different layers ("score_avg", "score_max", "feature_avg", "feature_concat"). Default: "score_avg".
            resolution (Optional[Union[int, Tuple[int, int]]]): Resolution to resize images to.
            device (Optional[Union[str, torch.device]]): Device to run the model on.
            dtype (Optional[Union[str, torch.dtype]]): Data type for model computations.
        """
        super().__init__(
            model=model,
            cosine_sim_clamp=cosine_sim_clamp,
            resolution=resolution,
            device=device,
            dtype=dtype,
            **kwargs
        )

        self.similarity_aggregation = similarity_aggregation.lower()

        if self.similarity_aggregation not in self.all_similarity_aggregations:
            raise ValueError(f"similarity_aggregation must be one of {self.all_similarity_aggregations}")

        self.closing_iterations = closing_iterations
        self.closing_kernel_size = closing_kernel_size
        self.interest_layer_indices = interest_layer_indices
        self.layer_fusion_method = layer_fusion_method.lower()
        if self.layer_fusion_method not in self.all_layer_fusion_methods:
            raise ValueError(f"layer_fusion_method must be one of {self.all_layer_fusion_methods}, "
                             f"got {self.layer_fusion_method}")

        self.coreset_ratio = coreset_ratio
        self.coreset_seed = coreset_seed
        self.knn_k = max(1, knn_k)
        self.knn_temperature = knn_temperature


    def get_prompt_features(
        self,
        prompt_images: Union[torch.Tensor, Image.Image, np.ndarray, list],
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        augmentation: bool = False,
        return_origin_dict: bool = False,
        include_last_layer: bool = True,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], VisionEncoderOutput]:
        """Build prompt features in the same shape the scoring path consumes.

        LASA's newer path returns a prepared prompt bank for multi-layer PatchIAD/
        DuoAD instead of handing a raw VisionEncoderOutput back to __call__ for
        per-batch reshaping. Keeping that behavior here makes the prompt-feature
        layout deterministic across repos for score/feature fusion.
        """
        prompt_features = super().get_prompt_features(
            prompt_images,
            resolution=resolution,
            augmentation=augmentation,
            return_origin_dict=True,
            **kwargs,
        )

        if return_origin_dict:
            return prompt_features

        feature_maps = prompt_features.feature_maps
        if not include_last_layer and len(feature_maps) > 1:
            feature_maps = feature_maps[:-1]

        layer_fusion_method = str(
            kwargs.get("layer_fusion_method", self.layer_fusion_method)
        ).lower()

        if len(feature_maps) == 1:
            return self._maybe_coreset(
                feature_maps[-1].permute(0, 2, 3, 1).flatten(0, 2)
            )

        if "score" in layer_fusion_method:
            return tuple(
                self._maybe_coreset(feature_map.permute(0, 2, 3, 1).flatten(0, 2))
                for feature_map in feature_maps
            )

        if layer_fusion_method == "feature_avg":
            prompt_feature_map = torch.stack(feature_maps, dim=0).mean(dim=0)
            return self._maybe_coreset(
                prompt_feature_map.permute(0, 2, 3, 1).flatten(0, 2)
            )

        if layer_fusion_method == "feature_concat":
            normalized_feature_maps = [
                F.normalize(feature_map, p=2, dim=1) for feature_map in feature_maps
            ]
            prompt_feature_map = torch.cat(normalized_feature_maps, dim=1)
            return self._maybe_coreset(
                prompt_feature_map.permute(0, 2, 3, 1).flatten(0, 2)
            )

        return self._maybe_coreset(
            feature_maps[-1].permute(0, 2, 3, 1).flatten(0, 2)
        )

    def _maybe_coreset(self, bank: torch.Tensor) -> torch.Tensor:
        """Apply PatchCore-style coreset subsampling to a memory bank.

        ``ratio <= 0`` (disabled) or ``>= 1`` (keep everything) are no-ops.
        For score-based fusion the bank is a tuple of per-layer ``[N, C]``
        tensors; each layer is coreset independently (they are scored
        separately and fused afterwards, so per-layer coverage is what counts).
        """
        if (
            self.coreset_ratio is None
            or self.coreset_ratio <= 0.0
            or self.coreset_ratio >= 1.0
            or bank.shape[0] < 2
        ):
            return bank
        return greedy_coreset(
            bank, self.coreset_ratio, seed=self.coreset_seed, device=bank.device
        )
    

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
            cls_token (torch.Tensor): The CLS token feature. Shape: (BS, D).
            patch_features (torch.Tensor): The patch features. Shape: (BS, N, D).
            attentions (torch.Tensor): The attention maps from the model. Not used in this function.
            eps (float): Small value to avoid division by zero. Default: 1e-6.
            **kwargs: Additional keyword arguments.
                is_closing (bool): Whether to apply morphological closing operation.
                    Default: True.

        Returns:
            torch.Tensor: The computed salient map.
        """
        is_closing = kwargs.get("is_closing", True)

        # Compute cls weighting to replace attention map in PatchEAD
        cls_sim = torch.matmul(
            F.normalize(cls_token, dim=-1).unsqueeze(1), 
            F.normalize(patch_features, dim=-1).transpose(1, 2)
        )
        batch_max = torch.amax(cls_sim, dim=-1, keepdim=True)
        cls_sim = cls_sim / (batch_max + eps)
        
        bs, h, w = shape

        cls_sim = cls_sim.reshape(bs, h, w)
        if is_closing:
            cls_sim = safe_closing_with_dilate(cls_sim, iterations=self.closing_iterations, 
                                               kernel_size=self.closing_kernel_size)

        return cls_sim
    

    def _aggregate_similarity_to_anomaly_map(
        self,
        cosine_similarity: torch.Tensor,
        similarity_aggregation: str,
        test_feature_shape: Tuple[int, int, int, int],
        **kwargs
    ) -> torch.Tensor:
        """
        Aggregate cosine similarity scores into an anomaly map using specified method.
        
        This function takes pre-computed similarity scores and applies the supported
        aggregation strategies to produce a final anomaly score per spatial location.
        
        Args:
            cosine_similarity: Shape [BS, H*W, N_patches], similarity between test and prompt patches.
            similarity_aggregation: One of "max" or "top1_mean".
            test_feature_shape: Tuple (batch_size, channels, height, width).
        
        Returns:
            Anomaly map of shape [BS, H, W], higher values indicate anomalies.
        """
        
        bs, c, h, w = test_feature_shape
        if similarity_aggregation == "max":
            anomaly = 1 - torch.amax(cosine_similarity, dim=-1)  # [BS, H*W]
        elif similarity_aggregation == "top1_mean":
            topk_sims, _ = torch.topk(cosine_similarity, k=max(1, int(0.01*h*w)), dim=-1)  # [BS, H*W, k]
            anomaly = 1 - topk_sims.mean(dim=-1)  # [BS, H*W]
        elif similarity_aggregation == "knn_weighted":
            # PatchCore-style weighted k-NN: take the k most similar memory
            # patches and weight them inversely to their distance (= softmax
            # over similarity; softmax is shift-invariant).  Robust to a single
            # spuriously-close background patch that would fool "max".
            k = max(1, min(self.knn_k, cosine_similarity.shape[-1]))  # [BS, H*W, k]
            topk_sims, _ = torch.topk(cosine_similarity, k=k, dim=-1)
            weights = torch.softmax(topk_sims / self.knn_temperature, dim=-1)
            anomaly = 1 - (topk_sims * weights).sum(dim=-1)  # [BS, H*W]
        else:
            raise ValueError(f"similarity_aggregation must be one of {self.all_similarity_aggregations}")
        
        # Reshape back to spatial dimensions
        anomaly_map = anomaly.reshape(bs, h, w)  # [BS, H, W]

        return anomaly_map
    

    def _compute_patch_anomaly_scores(
        self,
        prompt_features: torch.Tensor,
        test_feature_map: torch.Tensor,
        cosine_sim_clamp: Tuple[float, float],
        similarity_aggregation: str,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute per-patch anomaly scores by comparing test patches to prompt patches.
        
        This is a simplified helper that computes cosine similarity and converts it
        to anomaly scores using the minimum distance strategy.
        
        Args:
            prompt_features: Shape [N_patches, C], normalized prompt patch features.
            test_feature_map: Shape [BS, H, W, C], test feature map.
            cosine_sim_clamp: Tuple (min, max) for clamping cosine similarity.
            similarity_aggregation (str): Method to aggregate similarity scores.
            **kwargs: Additional arguments.
        
        Returns:
            Anomaly map of shape [BS, H, W].
        """
        N, _ = prompt_features.shape
        bs, c, h, w = test_feature_map.shape

        # Reshape test features to [BS, H*W, C] for similarity computation
        test_features_reshaped = test_feature_map.permute(0, 2, 3, 1).reshape(bs, h*w, c)  # [BS, H*W, C]

        cosine_similarity = self._calculate_cosine_similarity(
            test_features_reshaped, prompt_features, cosine_sim_clamp
        ) # [BS, H*W, N_patches]
        
        # Convert similarity to anomaly map
        anomaly_map = self._aggregate_similarity_to_anomaly_map(
            cosine_similarity,
            similarity_aggregation,
            test_feature_shape=(bs, c, h, w),
            **kwargs
        )

        return anomaly_map

    
    def _prepare_prompt_features(
        self,
        prompt_features: Union[torch.Tensor, Tuple[torch.Tensor, ...], VisionEncoderOutput],
        layer_fusion_method: str,
        is_last_layer_used: bool
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if isinstance(prompt_features, VisionEncoderOutput):
            used_feature_maps = prompt_features.feature_maps if is_last_layer_used else \
                prompt_features.feature_maps[:-1]
            if "score" in layer_fusion_method:
                prompt_features = [
                    m.permute(0, 2, 3, 1).flatten(0, 2) for m in used_feature_maps
                ]
            elif layer_fusion_method == "feature_avg":
                prompt_features = torch.stack(used_feature_maps, dim=0).mean(dim=0)  # [BS, C, H, W]
                prompt_features = prompt_features.permute(0, 2, 3, 1).flatten(0, 2)  # [BS*H*W, C]
            elif layer_fusion_method == "feature_concat":
                used_feature_maps = [F.normalize(m, p=2, dim=1) for m in used_feature_maps]
                prompt_features = torch.cat(used_feature_maps, dim=1)  # Concatenate on channel dim
                prompt_features = prompt_features.permute(0, 2, 3, 1).flatten(0, 2)  # [BS*H*W, C_total]
                
        else:
            if "score" in layer_fusion_method:
                # Wrap single tensor into a tuple for score-based fusion
                prompt_features = (prompt_features,) if isinstance(prompt_features, torch.Tensor) else prompt_features
            else:
                # Do nothing if prompt_features is already a tensor or list of tensors in feature fusion
                prompt_features = prompt_features
        return prompt_features
    

    def _compute_anomaly_map_from_features(
        self,
        test_features_dict: VisionEncoderOutput,
        prompt_features: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        layer_fusion_method: str,
        only_use_last_layer: bool,
        is_last_layer_used: bool,
        cosine_sim_clamp: Tuple[float, float],
        similarity_aggregation: str,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute anomaly map from multi-layer features with specified fusion method.
        
        This function handles the complete pipeline: feature selection → layer fusion → 
        similarity computation → anomaly score calculation. It supports multiple fusion
        strategies for combining information from different transformer layers.
        
        Args:
            test_features_dict: VisionEncoderOutput containing feature maps from multiple layers.
            prompt_features: Prepared prompt features (tensor or list based on fusion method).
            layer_fusion_method: One of "score_avg", "score_max", "feature_avg", "feature_concat".
            only_use_last_layer: Whether to use only the last layer.
            is_last_layer_used: Whether the last layer is included in fusion.
            cosine_sim_clamp: Tuple (min, max) for clamping cosine similarity.
            similarity_aggregation (str): Method to aggregate similarity scores.
            **kwargs: Additional arguments.
        
        Returns:
            Anomaly map of shape [BS, H, W].
        """
        feature_maps = test_features_dict.feature_maps if is_last_layer_used else \
            test_features_dict.feature_maps[:-1]

        if "score" in layer_fusion_method:
            anomaly_maps = []
            for i, feature_map in enumerate(feature_maps):
                anomaly_map = self._compute_patch_anomaly_scores(
                    prompt_features[i],
                    feature_map,
                    cosine_sim_clamp,
                    similarity_aggregation,
                    **kwargs
                )
                anomaly_maps.append(anomaly_map)

            if layer_fusion_method == "score_avg":
                anomaly_map = torch.stack(anomaly_maps, dim=0).mean(dim=0)
            elif layer_fusion_method == "score_max":
                anomaly_map, _ = torch.stack(anomaly_maps, dim=0).max(dim=0)
            elif only_use_last_layer:
                anomaly_map = anomaly_maps[0]

        elif layer_fusion_method == "feature_avg":
            # Average features across layers
            feature_maps = test_features_dict.feature_maps if is_last_layer_used else \
                test_features_dict.feature_maps[:-1]
            avg_feature_map = torch.stack(feature_maps, dim=0).mean(dim=0)  # [BS, C, H, W]
            anomaly_map = self._compute_patch_anomaly_scores(
                prompt_features,
                avg_feature_map,
                cosine_sim_clamp,
                similarity_aggregation,
                **kwargs
            )
        elif layer_fusion_method == "feature_concat":
            # Concatenate features across layers
            feature_maps = test_features_dict.feature_maps if is_last_layer_used else \
                test_features_dict.feature_maps[:-1]
            concat_feature_map = torch.cat(feature_maps, dim=1)# [BS, C_total, H, W]
            
            anomaly_map = self._compute_patch_anomaly_scores(
                prompt_features,
                concat_feature_map,
                cosine_sim_clamp,
                similarity_aggregation,
                **kwargs
            )

        return anomaly_map
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
        if anomaly_score_method not in ["max", "top1"]:
            raise ValueError("anomaly_score_method must be 'max' or 'top1'")

        if cosine_sim_clamp is None:
            cosine_sim_clamp = self.cosine_sim_clamp

        layer_fusion_method = kwargs.get("layer_fusion_method", self.layer_fusion_method).lower()
        if layer_fusion_method not in self.all_layer_fusion_methods:
            raise ValueError(f"layer_fusion_method must be one of {self.all_layer_fusion_methods}, "
                             f"got {layer_fusion_method}")
        
        similarity_aggregation = kwargs.get("similarity_aggregation", self.similarity_aggregation).lower()
        if similarity_aggregation not in self.all_similarity_aggregations:
            raise ValueError(f"similarity_aggregation must be one of {self.all_similarity_aggregations}, "
                             f"got {similarity_aggregation}")

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
            return_attentions=False,
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
