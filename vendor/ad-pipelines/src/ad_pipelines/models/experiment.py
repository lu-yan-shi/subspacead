from typing import Dict, Any, Optional, Tuple, Union

import hdbscan
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA
from torchvision.utils import save_image
    

def get_pca_featuremap(
    feature_map: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    n_components: int = 3,
) -> torch.Tensor:
    batch_size, channels, height, width = feature_map.shape
    if mask is not None:
        if mask.shape[-2:] != feature_map.shape[-2:]:
            raise ValueError("Mask shape must match feature map spatial dimensions. Got" \
            f"{mask.shape[-2:]} and {feature_map.shape[-2:]}")
        fg_patches = feature_map[mask].view(-1, feature_map.size(1))
    else:
        fg_patches = feature_map.view(-1, feature_map.size(1))
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(fg_patches)
    pca_feature_map = torch.from_numpy(pca.transform(fg_patches.numpy())).view(batch_size, height, width, n_components).permute(0, 3, 1, 2)
    # multiply by 2.0 and pass through a sigmoid to get vibrant colors 
    pca_feature_map = F.sigmoid(pca_feature_map.mul(2.0))
    return pca_feature_map

def get_foreground_mask(feature_map: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Get foreground mask from feature map using HDBSCAN clustering."""
    
    batch_size, channels, height, width = feature_map.shape
    
    # Reshape feature map to (batch_size * height * width, channels)
    features = feature_map.permute(0, 2, 3, 1).reshape(-1, channels).detach().cpu().numpy()
    
    # Apply HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=int(height * width * 0.01))  # 1% of total pixels
    cluster_labels = clusterer.fit_predict(features)
    
    # Convert cluster labels to RGB visualization
    unique_labels = np.unique(cluster_labels)
    num_clusters = len(unique_labels[unique_labels != -1])  # Exclude noise (-1)
    
    # Create RGB color map for clusters
    rgb_labels = torch.zeros(batch_size, height, width, 3)
    
    # Generate colors for each cluster (excluding noise)
    colors = torch.rand(num_clusters + 1, 3)  # +1 for noise cluster
    colors[0] = torch.tensor([0, 0, 0])  # Black for noise (-1)
    
    # Convert cluster labels back to tensor and reshape first
    cluster_labels_tensor = torch.from_numpy(cluster_labels).reshape(batch_size, height, width)
    
    for label in unique_labels:
        if label == -1:
            color_idx = 0  # Black for noise
        else:
            color_idx = label + 1
        mask = (cluster_labels_tensor == label)
        rgb_labels[mask] = colors[color_idx]
    
    # Convert to (batch_size, 3, height, width) format and save
    rgb_labels = rgb_labels.permute(0, 3, 1, 2)
    save_image(rgb_labels, 'cluster_labels_rgb.png')
    
    # Convert cluster labels back to tensor and reshape
    cluster_labels = torch.from_numpy(cluster_labels).reshape(batch_size, height, width)
    
    # Create foreground mask (non-noise clusters)
    foreground_mask = (cluster_labels != -1).float()
    # Save foreground mask as image
    save_image(foreground_mask.unsqueeze(1), 'foreground_mask.png')
    
    return foreground_mask