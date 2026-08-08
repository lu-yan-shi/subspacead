"""
Batch-wise Otsu's Binarization implementation in PyTorch.
Efficient GPU-accelerated thresholding for batch processing.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def batch_otsu_binarization(
    tensor: torch.Tensor,
    num_bins: int = 256,
    return_threshold: bool = False,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Apply Otsu's binarization method to a batch of tensors.
    
    Otsu's method finds the optimal threshold that maximizes the between-class
    variance (or equivalently, minimizes the within-class variance) for binary
    segmentation. This implementation processes entire batches in parallel on GPU.
    
    Args:
        tensor (torch.Tensor): Input tensor with shape (bs, ...) where values are
            in range [-1, 1]. The spatial dimensions will be flattened for processing.
        num_bins (int): Number of histogram bins. Default: 256. More bins provide
            finer threshold granularity but increase computation.
        return_threshold (bool): If True, return both binary mask and thresholds.
            Default: False.
        eps (float): Small epsilon for numerical stability. Default: 1e-10.
    
    Returns:
        torch.Tensor: Binary tensor with same shape as input, containing 0s and 1s.
            If return_threshold=True, returns tuple (binary_tensor, thresholds).
    
    Example:
        >>> x = torch.randn(4, 3, 224, 224) * 0.5  # Values roughly in [-1, 1]
        >>> binary = batch_otsu_binarization(x)
        >>> print(binary.shape)  # (4, 3, 224, 224)
        >>> print(binary.unique())  # tensor([0., 1.])
        
        >>> # Get thresholds for each batch
        >>> binary, thresholds = batch_otsu_binarization(x, return_threshold=True)
        >>> print(thresholds.shape)  # (4,)
    
    Notes:
        - Input values are first normalized to [0, 1] range for histogram computation
        - The algorithm computes optimal threshold independently for each batch item
        - Computation complexity: O(batch_size * num_bins * num_pixels)
        - Memory efficient: processes histograms in parallel without materializing
          full probability distributions
    """
    original_shape = tensor.shape
    batch_size = original_shape[0]
    device = tensor.device
    dtype = tensor.dtype
    
    # Flatten spatial dimensions: (bs, ...) -> (bs, num_pixels)
    tensor_flat = tensor.reshape(batch_size, -1)
    
    # Normalize from [-1, 1] to [0, 1] for histogram computation
    tensor_normalized = (tensor_flat + 1.0) / 2.0
    tensor_normalized = tensor_normalized.clamp(0, 1)
    
    # Compute histograms for all batches
    # Convert to bin indices: [0, num_bins-1]
    bin_indices = (tensor_normalized * (num_bins - 1)).long()
    
    # Create histograms using one-hot encoding + sum
    # Shape: (bs, num_pixels, num_bins)
    one_hot = F.one_hot(bin_indices, num_classes=num_bins).float()
    # Sum over pixels: (bs, num_bins)
    histograms = one_hot.sum(dim=1)
    
    # Normalize histograms to probabilities
    num_pixels = tensor_flat.shape[1]
    probabilities = histograms / (num_pixels + eps)  # (bs, num_bins)
    
    # Compute cumulative sums and means for each threshold position
    # P1(t) = cumulative probability up to threshold t
    cumsum_probs = torch.cumsum(probabilities, dim=1)  # (bs, num_bins)
    
    # Weighted cumulative sum: sum of i * p(i)
    bin_centers = torch.arange(num_bins, device=device, dtype=dtype).unsqueeze(0)  # (1, num_bins)
    weighted_probs = bin_centers * probabilities  # (bs, num_bins)
    cumsum_weighted = torch.cumsum(weighted_probs, dim=1)  # (bs, num_bins)
    
    # Global mean
    global_mean = cumsum_weighted[:, -1:]  # (bs, 1)
    
    # Compute between-class variance for all thresholds
    # σ²_B(t) = P1(t) * (1 - P1(t)) * (μ1(t) - μ2(t))²
    
    # Avoid division by zero
    P1 = cumsum_probs  # (bs, num_bins)
    P2 = 1.0 - P1
    
    # Mean of class 1 (below threshold)
    mu1 = cumsum_weighted / (P1 + eps)  # (bs, num_bins)
    
    # Mean of class 2 (above threshold)
    mu2 = (global_mean - cumsum_weighted) / (P2 + eps)  # (bs, num_bins)
    
    # Between-class variance
    between_class_variance = P1 * P2 * ((mu1 - mu2) ** 2)  # (bs, num_bins)
    
    # Find threshold that maximizes between-class variance
    optimal_bin_indices = torch.argmax(between_class_variance, dim=1)  # (bs,)
    
    # Convert bin indices back to [-1, 1] range thresholds
    optimal_thresholds = (optimal_bin_indices.float() / (num_bins - 1)) * 2.0 - 1.0  # (bs,)
    
    # Apply thresholding
    # Expand thresholds to match tensor shape for broadcasting
    thresholds_expanded = optimal_thresholds.view(batch_size, *([1] * (len(original_shape) - 1)))
    
    # Binarize: values >= threshold -> 1, values < threshold -> 0
    binary_tensor = (tensor >= thresholds_expanded).float()
    
    if return_threshold:
        return binary_tensor, optimal_thresholds
    return binary_tensor


def adaptive_otsu_binarization(
    tensor: torch.Tensor,
    block_size: int = 16,
    num_bins: int = 256,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Apply adaptive Otsu's binarization with local block processing.
    
    Instead of using a single global threshold, this method divides the input
    along the last dimension into blocks and computes an optimal threshold for 
    each block independently. This is useful for data with varying statistical
    properties across different regions.
    
    Args:
        tensor (torch.Tensor): Input tensor with shape (bs, ..., N) where values
            are in range [-1, 1]. The last dimension will be divided into blocks.
        block_size (int): Size of local blocks for adaptive thresholding. Default: 16.
            Smaller blocks adapt better to local variations but may be noisier.
        num_bins (int): Number of histogram bins. Default: 256.
        eps (float): Small epsilon for numerical stability. Default: 1e-10.
    
    Returns:
        torch.Tensor: Binary tensor with same shape as input.
    
    Example:
        >>> # For 1D sequence data
        >>> x = torch.randn(4, 1000) * 0.5
        >>> binary = adaptive_otsu_binarization(x, block_size=100)
        >>> print(binary.shape)  # (4, 1000)
        
        >>> # For multi-dimensional data
        >>> x = torch.randn(2, 10, 500) * 0.5
        >>> binary = adaptive_otsu_binarization(x, block_size=50)
        >>> print(binary.shape)  # (2, 10, 500)
    
    Notes:
        - Last dimension should be divisible by block_size for best results
        - If not divisible, the tensor will be padded automatically
        - Works with any number of dimensions >= 2
    """
    if tensor.dim() < 2:
        raise ValueError(f"Expected at least 2D input (bs, ..., N), got {tensor.dim()}D tensor")
    
    original_shape = tensor.shape
    batch_size = original_shape[0]
    last_dim_size = original_shape[-1]
    device = tensor.device
    
    # Flatten all dimensions except batch and last: (bs, ..., N) -> (bs, -1, N)
    if tensor.dim() > 2:
        middle_dims = original_shape[1:-1]
        middle_size = 1
        for dim in middle_dims:
            middle_size *= dim
        tensor_reshaped = tensor.reshape(batch_size, middle_size, last_dim_size)
    else:
        middle_size = 1
        tensor_reshaped = tensor.unsqueeze(1)  # (bs, N) -> (bs, 1, N)
    
    # Calculate padding needed for last dimension
    pad_size = (block_size - last_dim_size % block_size) % block_size
    
    # Pad if necessary
    if pad_size > 0:
        tensor_reshaped = F.pad(tensor_reshaped, (0, pad_size), mode='reflect')
        padded_last_dim = last_dim_size + pad_size
    else:
        padded_last_dim = last_dim_size
    
    # Reshape into blocks: (bs, middle, N) -> (bs, middle, n_blocks, block_size)
    n_blocks = padded_last_dim // block_size
    blocks = tensor_reshaped.reshape(batch_size, middle_size, n_blocks, block_size)
    
    # Flatten to process each block: (bs * middle * n_blocks, block_size)
    blocks_flat = blocks.reshape(-1, block_size)
    
    # Apply Otsu to each block
    binary_blocks = batch_otsu_binarization(blocks_flat, num_bins=num_bins, eps=eps)
    
    # Reshape back: (bs, middle, n_blocks, block_size)
    binary_blocks = binary_blocks.reshape(batch_size, middle_size, n_blocks, block_size)
    
    # Merge blocks: (bs, middle, N_padded)
    binary_tensor = binary_blocks.reshape(batch_size, middle_size, padded_last_dim)
    
    # Remove padding if it was added
    if pad_size > 0:
        binary_tensor = binary_tensor[:, :, :last_dim_size]
    
    # Reshape back to original shape
    if tensor.dim() > 2:
        binary_tensor = binary_tensor.reshape(original_shape)
    else:
        binary_tensor = binary_tensor.squeeze(1)  # (bs, 1, N) -> (bs, N)
    
    return binary_tensor


__all__ = ['batch_otsu_binarization', 'adaptive_otsu_binarization']
