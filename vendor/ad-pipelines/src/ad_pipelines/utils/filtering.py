"""
Advanced filtering operations for image processing.
Implements bilateral filter and other edge-preserving smoothing techniques.
"""

import torch
import torch.nn.functional as F
from typing import Union, Tuple, Optional
import math


def bilateral_filter(
    tensor: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]] = 3,
    iterations: int = 1,
    spatial_sigma: Optional[float] = None,
    range_sigma: Optional[float] = None,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Apply bilateral filter to preserve edges while smoothing.
    
    The bilateral filter is a non-linear, edge-preserving smoothing filter.
    It combines spatial proximity and intensity similarity to compute weighted averages,
    preserving strong edges (large intensity differences) while smoothing weak variations.
    
    Args:
        tensor (torch.Tensor): Input tensor with shape (bs, h, w).
            Values are typically in range [0, 1] for best results.
        kernel_size (int or tuple): Size of the filter kernel. If int, creates square kernel.
            If tuple, (height, width). This determines the spatial extent. Default: 3.
        iterations (int): Number of times to apply the filter. Default: 1.
        spatial_sigma (Optional[float]): Standard deviation for spatial Gaussian kernel.
            If None, auto-set to kernel_size / 3. Default: None.
        range_sigma (Optional[float]): Standard deviation for range (intensity) Gaussian.
            If None, auto-set to 0.1. Controls edge preservation. Default: None.
        eps (float): Small value for numerical stability. Default: 1e-6.
    
    Returns:
        torch.Tensor: Filtered tensor with same shape as input.
    
    Example:
        >>> x = torch.rand(2, 100, 100)
        >>> # Use same parameters as closing for easy comparison
        >>> filtered = bilateral_filter(x, kernel_size=3, iterations=3)
        >>> print(filtered.shape)  # (2, 100, 100)
    
    Notes:
        - API matches morphological operations (kernel_size, iterations)
        - Bilateral filter preserves edges better than Gaussian smoothing
        - spatial_sigma controls spatial smoothing strength
        - range_sigma controls edge preservation (smaller = stronger preservation)
        - Multiple iterations = stronger smoothing while preserving edges
    
    Reference:
        Tomasi, C., & Manduchi, R. (1998). Bilateral filtering for gray and color images.
        ICCV 1998.
    """
    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D input (bs, h, w), got {tensor.dim()}D tensor")
    
    device = tensor.device
    dtype = tensor.dtype
    bs, h, w = tensor.shape
    
    # Parse kernel_size
    if isinstance(kernel_size, int):
        kernel_h, kernel_w = kernel_size, kernel_size
    else:
        kernel_h, kernel_w = kernel_size
    
    # Auto-set spatial_sigma from kernel_size
    if spatial_sigma is None:
        spatial_sigma = max(kernel_h, kernel_w) / 3.0
    
    # Auto-set range_sigma
    # Balance between edge preservation and hole-filling
    # 0.1 = strong edge preservation but leaves holes
    # 0.3 = fills holes but destroys shape
    # 0.15-0.2 = balanced
    if range_sigma is None:
        range_sigma = 0.15  # Balanced default for anomaly detection
    
    kernel_radius = max(kernel_h, kernel_w) // 2
    kernel_size_actual = 2 * kernel_radius + 1
    kernel_radius = max(kernel_h, kernel_w) // 2
    kernel_size_actual = 2 * kernel_radius + 1
    
    # Add channel dimension: (bs, h, w) -> (bs, 1, h, w)
    result = tensor
    
    # Apply bilateral filter iteratively
    for _ in range(iterations):
        tensor_4d = result.unsqueeze(1)
        
        # Create spatial Gaussian kernel
        # Grid of relative positions
        y_grid, x_grid = torch.meshgrid(
            torch.arange(-kernel_radius, kernel_radius + 1, device=device, dtype=torch.float32),
            torch.arange(-kernel_radius, kernel_radius + 1, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Spatial weights: Gaussian based on distance
        spatial_weights = torch.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * spatial_sigma ** 2))
        spatial_weights = spatial_weights / (spatial_weights.sum() + eps)  # Normalize
        spatial_weights = spatial_weights.unsqueeze(0).unsqueeze(0)  # (1, 1, kernel_size, kernel_size)
        
        # Pad input to handle borders
        padding = kernel_radius
        padded = F.pad(tensor_4d, (padding, padding, padding, padding), mode='replicate')
        
        # Unfold to get all patches
        patches = F.unfold(
            padded,
            kernel_size=(kernel_size_actual, kernel_size_actual),
            stride=1
        )  # (bs, kernel_size*kernel_size, h*w)
        
        # Reshape for easier processing
        patches = patches.view(bs, kernel_size_actual * kernel_size_actual, h, w)  # (bs, k*k, h, w)
        
        # Compute intensity differences from center pixel
        center_values = tensor_4d  # (bs, 1, h, w)
        intensity_diff = patches - center_values  # (bs, k*k, h, w)
        
        # Range weights: Gaussian based on intensity difference
        range_weights = torch.exp(-(intensity_diff ** 2) / (2 * range_sigma ** 2))  # (bs, k*k, h, w)
        
        # Combine spatial and range weights
        spatial_weights_flat = spatial_weights.view(1, -1, 1, 1)  # (1, k*k, 1, 1)
        combined_weights = spatial_weights_flat * range_weights  # (bs, k*k, h, w)
        
        # Normalize weights
        weight_sum = combined_weights.sum(dim=1, keepdim=True) + eps  # (bs, 1, h, w)
        normalized_weights = combined_weights / weight_sum  # (bs, k*k, h, w)
        
        # Apply weighted average
        result = (patches * normalized_weights).sum(dim=1)  # (bs, h, w)
    
    return result.to(dtype)


def adaptive_bilateral_filter(
    tensor: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]] = 3,
    iterations: int = 1,
    threshold: float = 0.5,
    spatial_sigma: Optional[float] = None,
    range_sigma: Optional[float] = None,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Apply bilateral filter only to regions below a threshold.
    
    This adaptive version applies bilateral filtering selectively:
    - Low-confidence regions (below threshold): bilateral filter applied
    - High-confidence regions (above threshold): kept unchanged
    
    Useful for preserving certain regions while smoothing uncertain areas.
    
    Args:
        tensor (torch.Tensor): Input tensor with shape (bs, h, w).
        kernel_size (int or tuple): Size of the filter kernel. Default: 3.
        iterations (int): Number of times to apply the filter. Default: 1.
        threshold (float): Values below this are filtered, above are preserved. Default: 0.5.
        spatial_sigma (Optional[float]): Spatial Gaussian std. Auto-set if None. Default: None.
        range_sigma (Optional[float]): Range Gaussian std. Auto-set if None. Default: None.
        eps (float): Small value for numerical stability. Default: 1e-6.
    
    Returns:
        torch.Tensor: Adaptively filtered tensor with same shape as input.
    
    Example:
        >>> x = torch.rand(2, 100, 100)
        >>> # Only smooth low-confidence regions, use same params as closing
        >>> filtered = adaptive_bilateral_filter(x, kernel_size=3, iterations=3, threshold=0.3)
    """
    # Apply bilateral filter
    filtered = bilateral_filter(
        tensor,
        kernel_size=kernel_size,
        iterations=iterations,
        spatial_sigma=spatial_sigma,
        range_sigma=range_sigma,
        eps=eps
    )
    
    # Create mask for regions to filter
    mask = (tensor < threshold).float()
    
    # Blend filtered and original based on mask
    result = mask * filtered + (1 - mask) * tensor
    
    return result


def guided_filter(
    input_tensor: torch.Tensor,
    guide_tensor: torch.Tensor,
    radius: int = 3,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Apply guided filter using a guide image.
    
    The guided filter is an edge-preserving smoothing filter that uses a guide image
    to determine filtering behavior. It's faster than bilateral filter and produces
    similar edge-preserving results.
    
    Args:
        input_tensor (torch.Tensor): Input to be filtered, shape (bs, h, w).
        guide_tensor (torch.Tensor): Guide image, shape (bs, h, w).
            The guide determines where edges are preserved.
        radius (int): Radius of the local window. Default: 3.
        eps (float): Regularization parameter. Smaller = more edge preservation. Default: 1e-6.
    
    Returns:
        torch.Tensor: Filtered tensor with same shape as input.
    
    Example:
        >>> cls_sim = torch.rand(2, 50, 50)
        >>> patch_features_map = torch.rand(2, 50, 50)  # Some feature similarity map
        >>> # Use feature similarity to guide smoothing
        >>> filtered = guided_filter(cls_sim, patch_features_map, radius=3)
    
    Notes:
        - Guide can be the input itself (like bilateral filter)
        - Or use semantic features to guide spatial smoothing
        - Much faster than bilateral filter: O(N) vs O(N * kernel_size^2)
    
    Reference:
        He, K., Sun, J., & Tang, X. (2013). Guided image filtering.
        TPAMI 2013.
    """
    if input_tensor.dim() != 3 or guide_tensor.dim() != 3:
        raise ValueError(f"Expected 3D inputs (bs, h, w)")
    
    if input_tensor.shape != guide_tensor.shape:
        raise ValueError(f"Input and guide must have same shape")
    
    device = input_tensor.device
    dtype = input_tensor.dtype
    bs, h, w = input_tensor.shape
    
    # Add channel dimension
    p = input_tensor.unsqueeze(1)  # (bs, 1, h, w)
    I = guide_tensor.unsqueeze(1)  # (bs, 1, h, w)
    
    kernel_size = 2 * radius + 1
    
    # Box filter (mean filter)
    def box_filter(x, r):
        """Fast box filter using cumulative sum."""
        padding = r
        padded = F.pad(x, (padding, padding, padding, padding), mode='replicate')
        return F.avg_pool2d(padded, kernel_size=2*r+1, stride=1)
    
    # Compute statistics in local window
    mean_I = box_filter(I, radius)  # Mean of guide
    mean_p = box_filter(p, radius)  # Mean of input
    mean_Ip = box_filter(I * p, radius)  # Mean of guide * input
    mean_II = box_filter(I * I, radius)  # Mean of guide^2
    
    # Covariance and variance
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I
    
    # Linear coefficients
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    # Average coefficients
    mean_a = box_filter(a, radius)
    mean_b = box_filter(b, radius)
    
    # Apply linear model
    output = mean_a * I + mean_b
    
    return output.squeeze(1).to(dtype)


def gaussian_smooth(
    tensor: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]] = 3,
    iterations: int = 1,
    sigma: Optional[float] = None
) -> torch.Tensor:
    """
    Apply Gaussian smoothing filter.
    
    Simple, fast, and differentiable smoothing operation. Good baseline
    for comparison with edge-preserving filters.
    
    Args:
        tensor (torch.Tensor): Input tensor with shape (bs, h, w).
        kernel_size (int or tuple): Size of the Gaussian kernel. Default: 3.
        iterations (int): Number of times to apply smoothing. Default: 1.
        sigma (Optional[float]): Standard deviation of Gaussian. 
            If None, auto-set to kernel_size / 3. Default: None.
    
    Returns:
        torch.Tensor: Smoothed tensor with same shape as input.
    
    Example:
        >>> x = torch.rand(2, 100, 100)
        >>> # Use same parameters as closing
        >>> smoothed = gaussian_smooth(x, kernel_size=3, iterations=3)
    """
    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D input (bs, h, w), got {tensor.dim()}D tensor")
    
    device = tensor.device
    dtype = tensor.dtype
    
    # Parse kernel_size
    if isinstance(kernel_size, int):
        kernel_h, kernel_w = kernel_size, kernel_size
    else:
        kernel_h, kernel_w = kernel_size
    
    # Auto-set sigma from kernel_size
    if sigma is None:
        sigma = max(kernel_h, kernel_w) / 3.0
    
    kernel_radius = max(kernel_h, kernel_w) // 2
    kernel_size_actual = 2 * kernel_radius + 1
    
    # Create 1D Gaussian kernel
    x = torch.arange(-kernel_radius, kernel_radius + 1, device=device, dtype=torch.float32)
    gauss_1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    
    # Reshape for 2D convolution (separable filter for efficiency)
    gauss_1d = gauss_1d.view(1, 1, -1)
    
    # Apply iteratively
    result = tensor
    for _ in range(iterations):
        # Add channel dimension
        tensor_4d = result.unsqueeze(1)  # (bs, 1, h, w)
        
        # Apply horizontal convolution
        padding = kernel_radius
        smoothed = F.conv2d(
            F.pad(tensor_4d, (padding, padding, 0, 0), mode='replicate'),
            gauss_1d.unsqueeze(2),  # (1, 1, 1, kernel_size)
            padding=0
        )
        
        # Apply vertical convolution
        smoothed = F.conv2d(
            F.pad(smoothed, (0, 0, padding, padding), mode='replicate'),
            gauss_1d.unsqueeze(3),  # (1, 1, kernel_size, 1)
            padding=0
        )
        
        result = smoothed.squeeze(1)
    
    return result.to(dtype)


def safe_bilateral_filter(
    tensor: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]] = 3,
    iterations: int = 1,
    spatial_sigma: Optional[float] = None,
    range_sigma: Optional[float] = None,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Apply bilateral filter with safe padding to handle boundaries.
    
    Similar to safe_closing, this ensures proper boundary handling by
    pre-padding the input with replicate mode.
    
    Args:
        tensor (torch.Tensor): Input tensor with shape (bs, h, w).
        kernel_size (int or tuple): Size of the filter kernel. Default: 3.
        iterations (int): Number of times to apply the filter. Default: 1.
        spatial_sigma (Optional[float]): Spatial Gaussian std. Auto-set if None.
        range_sigma (Optional[float]): Range Gaussian std. Auto-set if None.
        eps (float): Small value for numerical stability. Default: 1e-6.
    
    Returns:
        torch.Tensor: Filtered tensor with same shape as input.
    
    Example:
        >>> x = torch.rand(2, 100, 100)
        >>> # Drop-in replacement for safe_closing
        >>> filtered = safe_bilateral_filter(x, kernel_size=3, iterations=3)
    """
    # Determine pad size
    if isinstance(kernel_size, int):
        kh, kw = kernel_size, kernel_size
    else:
        kh, kw = kernel_size
    
    pad_h = (kh // 2) * iterations + 1
    pad_w = (kw // 2) * iterations + 1
    
    # Pre-pad with replicate mode
    padded_tensor = F.pad(tensor, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
    
    # Apply bilateral filter
    processed = bilateral_filter(
        padded_tensor,
        kernel_size=kernel_size,
        iterations=iterations,
        spatial_sigma=spatial_sigma,
        range_sigma=range_sigma,
        eps=eps
    )
    
    # Crop back to original size
    if pad_h > 0 and pad_w > 0:
        result = processed[..., pad_h:-pad_h, pad_w:-pad_w]
    elif pad_h > 0:
        result = processed[..., pad_h:-pad_h, :]
    elif pad_w > 0:
        result = processed[..., :, pad_w:-pad_w]
    else:
        result = processed
    
    return result


def safe_gaussian_smooth(
    tensor: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]] = 3,
    iterations: int = 1,
    sigma: Optional[float] = None
) -> torch.Tensor:
    """
    Apply Gaussian smoothing with safe padding to handle boundaries.
    
    Similar to safe_closing, this ensures proper boundary handling.
    
    Args:
        tensor (torch.Tensor): Input tensor with shape (bs, h, w).
        kernel_size (int or tuple): Size of the Gaussian kernel. Default: 3.
        iterations (int): Number of times to apply smoothing. Default: 1.
        sigma (Optional[float]): Standard deviation. Auto-set if None.
    
    Returns:
        torch.Tensor: Smoothed tensor with same shape as input.
    
    Example:
        >>> x = torch.rand(2, 100, 100)
        >>> # Drop-in replacement for safe_closing
        >>> smoothed = safe_gaussian_smooth(x, kernel_size=3, iterations=3)
    """
    # Determine pad size
    if isinstance(kernel_size, int):
        kh, kw = kernel_size, kernel_size
    else:
        kh, kw = kernel_size
    
    pad_h = (kh // 2) * iterations + 1
    pad_w = (kw // 2) * iterations + 1
    
    # Pre-pad with replicate mode
    padded_tensor = F.pad(tensor, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
    
    # Apply Gaussian smoothing
    processed = gaussian_smooth(
        padded_tensor,
        kernel_size=kernel_size,
        iterations=iterations,
        sigma=sigma
    )
    
    # Crop back to original size
    if pad_h > 0 and pad_w > 0:
        result = processed[..., pad_h:-pad_h, pad_w:-pad_w]
    elif pad_h > 0:
        result = processed[..., pad_h:-pad_h, :]
    elif pad_w > 0:
        result = processed[..., :, pad_w:-pad_w]
    else:
        result = processed
    
    return result


def asymmetric_smooth(
    tensor: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]] = 3,
    iterations: int = 1,
    spatial_sigma: Optional[float] = None,
    mode: str = 'max_preserving'
) -> torch.Tensor:
    """
    Apply asymmetric smoothing where high values can spread to low values, but not vice versa.
    
    This filter allows high-confidence regions to expand into low-confidence regions
    while preventing low values from contaminating high values. Useful for filling
    holes in anomaly maps without destroying strong anomaly signals.
    
    Args:
        tensor (torch.Tensor): Input tensor with shape (bs, h, w).
        kernel_size (int or tuple): Size of the filter kernel. Default: 3.
        iterations (int): Number of times to apply the filter. Default: 1.
        spatial_sigma (Optional[float]): Spatial Gaussian std for distance weighting.
            If None, auto-set to kernel_size / 3. Default: None.
        mode (str): Smoothing mode:
            - 'max_preserving': Use weighted average only from higher neighbors
            - 'dilation_smooth': Blend between max pooling and gaussian smoothing
            Default: 'max_preserving'.
    
    Returns:
        torch.Tensor: Asymmetrically smoothed tensor with same shape as input.
    
    Example:
        >>> x = torch.rand(2, 100, 100)
        >>> # High values spread to fill holes, low values don't contaminate high regions
        >>> smoothed = asymmetric_smooth(x, kernel_size=3, iterations=3)
    
    Notes:
        - This is ideal for CLS similarity maps where high similarity should expand
        - Fills holes (low-value regions) without reducing peak values
        - More aggressive than bilateral filter in preserving maxima
    """
    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D input (bs, h, w), got {tensor.dim()}D tensor")
    
    device = tensor.device
    dtype = tensor.dtype
    bs, h, w = tensor.shape
    
    # Parse kernel_size
    if isinstance(kernel_size, int):
        kernel_h, kernel_w = kernel_size, kernel_size
    else:
        kernel_h, kernel_w = kernel_size
    
    # Auto-set spatial_sigma from kernel_size
    if spatial_sigma is None:
        spatial_sigma = max(kernel_h, kernel_w) / 3.0
    
    kernel_radius = max(kernel_h, kernel_w) // 2
    kernel_size_actual = 2 * kernel_radius + 1
    
    # Apply asymmetric smoothing iteratively
    result = tensor
    
    for _ in range(iterations):
        tensor_4d = result.unsqueeze(1)  # (bs, 1, h, w)
        
        if mode == 'max_preserving':
            # Create spatial Gaussian kernel for distance weighting
            y_grid, x_grid = torch.meshgrid(
                torch.arange(-kernel_radius, kernel_radius + 1, device=device, dtype=torch.float32),
                torch.arange(-kernel_radius, kernel_radius + 1, device=device, dtype=torch.float32),
                indexing='ij'
            )
            
            spatial_weights = torch.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * spatial_sigma ** 2))
            spatial_weights = spatial_weights / spatial_weights.sum()
            spatial_weights = spatial_weights.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)
            
            # Pad and unfold
            padding = kernel_radius
            padded = F.pad(tensor_4d, (padding, padding, padding, padding), mode='replicate')
            patches = F.unfold(
                padded,
                kernel_size=(kernel_size_actual, kernel_size_actual),
                stride=1
            )  # (bs, k*k, h*w)
            
            patches = patches.view(bs, kernel_size_actual * kernel_size_actual, h, w)  # (bs, k*k, h, w)
            
            # For each pixel, only consider neighbors >= current value
            center_values = tensor_4d  # (bs, 1, h, w)
            
            # Mask: 1 where neighbor >= center, 0 otherwise
            higher_mask = (patches >= center_values).float()  # (bs, k*k, h, w)
            
            # Apply spatial weights only to higher neighbors
            spatial_weights_flat = spatial_weights.view(1, -1, 1, 1)  # (1, k*k, 1, 1)
            weights = spatial_weights_flat * higher_mask  # (bs, k*k, h, w)
            
            # Normalize weights
            weight_sum = weights.sum(dim=1, keepdim=True) + 1e-6  # (bs, 1, h, w)
            normalized_weights = weights / weight_sum
            
            # Weighted average (only from higher neighbors)
            result = (patches * normalized_weights).sum(dim=1)  # (bs, h, w)
            
        elif mode == 'dilation_smooth':
            # Blend between max pooling (100% dilation) and gaussian smoothing
            # This is faster but less precise
            
            # Max pooling component
            padding = kernel_radius
            max_pooled = F.max_pool2d(
                F.pad(tensor_4d, (padding, padding, padding, padding), mode='replicate'),
                kernel_size=kernel_size_actual,
                stride=1
            ).squeeze(1)
            
            # Gaussian smoothing component
            gauss_smoothed = gaussian_smooth(result, kernel_size=kernel_size, iterations=1)
            
            # Blend: take max of current and smoothed (ensures high values dominate)
            result = torch.max(gauss_smoothed, result * 0.7 + max_pooled * 0.3)
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'max_preserving' or 'dilation_smooth'")
    
    return result.to(dtype)


def safe_asymmetric_smooth(
    tensor: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]] = 3,
    iterations: int = 1,
    spatial_sigma: Optional[float] = None,
    mode: str = 'max_preserving'
) -> torch.Tensor:
    """
    Apply asymmetric smoothing with safe padding to handle boundaries.
    
    Args:
        tensor (torch.Tensor): Input tensor with shape (bs, h, w).
        kernel_size (int or tuple): Size of the filter kernel. Default: 3.
        iterations (int): Number of times to apply the filter. Default: 1.
        spatial_sigma (Optional[float]): Spatial Gaussian std. Auto-set if None.
        mode (str): 'max_preserving' or 'dilation_smooth'. Default: 'max_preserving'.
    
    Returns:
        torch.Tensor: Asymmetrically smoothed tensor with same shape as input.
    
    Example:
        >>> x = torch.rand(2, 100, 100)
        >>> # Drop-in replacement for safe_closing with asymmetric behavior
        >>> smoothed = safe_asymmetric_smooth(x, kernel_size=3, iterations=3)
    """
    # Determine pad size
    if isinstance(kernel_size, int):
        kh, kw = kernel_size, kernel_size
    else:
        kh, kw = kernel_size
    
    pad_h = (kh // 2) * iterations + 1
    pad_w = (kw // 2) * iterations + 1
    
    # Pre-pad with replicate mode
    padded_tensor = F.pad(tensor, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
    
    # Apply asymmetric smoothing
    processed = asymmetric_smooth(
        padded_tensor,
        kernel_size=kernel_size,
        iterations=iterations,
        spatial_sigma=spatial_sigma,
        mode=mode
    )
    
    # Crop back to original size
    if pad_h > 0 and pad_w > 0:
        result = processed[..., pad_h:-pad_h, pad_w:-pad_w]
    elif pad_h > 0:
        result = processed[..., pad_h:-pad_h, :]
    elif pad_w > 0:
        result = processed[..., :, pad_w:-pad_w]
    else:
        result = processed
    
    return result


__all__ = [
    'bilateral_filter',
    'adaptive_bilateral_filter',
    'guided_filter',
    'gaussian_smooth',
    'safe_bilateral_filter',
    'safe_gaussian_smooth',
    'asymmetric_smooth',
    'safe_asymmetric_smooth'
]
