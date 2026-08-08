"""
Efficient batch-wise morphological operations in PyTorch.
Implements erosion, dilation, opening, and closing using GPU-accelerated convolutions.
"""

import torch
import torch.nn.functional as F
from typing import Union, Tuple, Optional


def create_circular_kernel(radius: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a circular structuring element (kernel) for morphological operations.
    
    Args:
        radius (int): Radius of the circular kernel. Kernel size will be (2*radius+1).
        device (torch.device): Device to create the kernel on. Default: None (CPU).
    
    Returns:
        torch.Tensor: Binary kernel with shape (1, 1, kernel_size, kernel_size).
    
    Example:
        >>> kernel = create_circular_kernel(radius=2)
        >>> print(kernel.shape)  # (1, 1, 5, 5)
    """
    kernel_size = 2 * radius + 1
    y, x = torch.meshgrid(
        torch.arange(kernel_size, device=device, dtype=torch.float32) - radius,
        torch.arange(kernel_size, device=device, dtype=torch.float32) - radius,
        indexing='ij'
    )
    kernel = (x ** 2 + y ** 2 <= radius ** 2).float()
    return kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


def create_rectangular_kernel(height: int, width: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a rectangular structuring element (kernel).
    
    Args:
        height (int): Height of the rectangular kernel.
        width (int): Width of the rectangular kernel.
        device (torch.device): Device to create the kernel on. Default: None (CPU).
    
    Returns:
        torch.Tensor: Binary kernel with shape (1, 1, height, width).
    """
    kernel = torch.ones(1, 1, height, width, device=device, dtype=torch.float32)
    return kernel


def erode(
    tensor: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]] = 3,
    kernel_type: str = 'rectangular',
    iterations: int = 1
) -> torch.Tensor:
    """
    Apply erosion operation to a batch of binary masks.
    
    Erosion shrinks the foreground (1s) regions by removing pixels at boundaries.
    It's implemented using minimum pooling with a structuring element.
    
    Args:
        tensor (torch.Tensor): Input binary tensor with shape (bs, h, w).
            Values should be 0 or 1.
        kernel_size (int or tuple): Size of the structuring element.
            If int, creates a square kernel. If tuple, (height, width).
            For circular kernels, this is interpreted as radius. Default: 3.
        kernel_type (str): Type of kernel: 'rectangular' or 'circular'. Default: 'rectangular'.
        iterations (int): Number of times to apply erosion. Default: 1.
    
    Returns:
        torch.Tensor: Eroded binary tensor with same shape as input.
    
    Example:
        >>> x = torch.ones(2, 100, 100)  # Binary mask
        >>> eroded = erode(x, kernel_size=3, iterations=2)
        >>> print(eroded.shape)  # (2, 100, 100)
    
    Notes:
        - Erosion removes small noise and disconnects weakly connected regions
        - Multiple iterations = stronger erosion effect
        - Circular kernels are more isotropic than rectangular ones
    """
    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D input (bs, h, w), got {tensor.dim()}D tensor")
    
    device = tensor.device
    dtype = tensor.dtype
    bs, h, w = tensor.shape
    
    # Add channel dimension: (bs, h, w) -> (bs, 1, h, w)
    tensor = tensor.unsqueeze(1)
    
    # Create kernel
    if kernel_type == 'circular':
        radius = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        kernel = create_circular_kernel(radius, device=device)
    elif kernel_type == 'rectangular':
        if isinstance(kernel_size, int):
            kh, kw = kernel_size, kernel_size
        else:
            kh, kw = kernel_size
        kernel = create_rectangular_kernel(kh, kw, device=device)
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}. Use 'rectangular' or 'circular'.")
    
    # Apply erosion (minimum pooling)
    result = tensor
    for _ in range(iterations):
        # Pad to maintain size
        padding = (kernel.shape[-1] // 2, kernel.shape[-1] // 2, 
                   kernel.shape[-2] // 2, kernel.shape[-2] // 2)
        padded = F.pad(result, padding, mode='constant', value=1)  # Pad with 1s (background)
        
        # Unfold to get patches
        patches = F.unfold(
            padded, 
            kernel_size=(kernel.shape[-2], kernel.shape[-1]),
            stride=1
        )  # (bs, kernel_h*kernel_w, h*w)
        
        # Reshape kernel to match
        kernel_flat = kernel.view(-1)  # (kernel_h*kernel_w,)
        
        # Apply kernel mask: only consider kernel positions
        masked_patches = patches * kernel_flat.unsqueeze(0).unsqueeze(-1)  # (bs, kernel_h*kernel_w, h*w)
        
        # For erosion, we want minimum where kernel is 1
        # Set non-kernel positions to 1 (so they don't affect min)
        kernel_mask = kernel_flat.unsqueeze(0).unsqueeze(-1)  # (1, kernel_h*kernel_w, 1)
        masked_patches = torch.where(
            kernel_mask > 0, 
            masked_patches, 
            torch.ones_like(masked_patches)
        )
        
        # Take minimum over kernel positions
        result = torch.min(masked_patches, dim=1, keepdim=True)[0]  # (bs, 1, h*w)
        result = result.view(bs, 1, h, w)
    
    # Remove channel dimension
    result = result.squeeze(1)
    
    return result.to(dtype)


def dilate(
    tensor: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]] = 3,
    kernel_type: str = 'rectangular',
    iterations: int = 1
) -> torch.Tensor:
    """
    Apply dilation operation to a batch of binary masks.
    
    Dilation expands the foreground (1s) regions by adding pixels at boundaries.
    It's implemented using maximum pooling with a structuring element.
    
    Args:
        tensor (torch.Tensor): Input binary tensor with shape (bs, h, w).
            Values should be 0 or 1.
        kernel_size (int or tuple): Size of the structuring element.
            If int, creates a square kernel. If tuple, (height, width).
            For circular kernels, this is interpreted as radius. Default: 3.
        kernel_type (str): Type of kernel: 'rectangular' or 'circular'. Default: 'rectangular'.
        iterations (int): Number of times to apply dilation. Default: 1.
    
    Returns:
        torch.Tensor: Dilated binary tensor with same shape as input.
    
    Example:
        >>> x = torch.zeros(2, 100, 100)
        >>> x[:, 45:55, 45:55] = 1  # Small square region
        >>> dilated = dilate(x, kernel_size=5, iterations=2)
        >>> print(dilated.shape)  # (2, 100, 100)
    
    Notes:
        - Dilation fills small holes and connects nearby regions
        - Multiple iterations = stronger dilation effect
        - Circular kernels are more isotropic than rectangular ones
    """
    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D input (bs, h, w), got {tensor.dim()}D tensor")
    
    device = tensor.device
    dtype = tensor.dtype
    bs, h, w = tensor.shape
    
    # Add channel dimension: (bs, h, w) -> (bs, 1, h, w)
    tensor = tensor.unsqueeze(1)
    
    # Create kernel
    if kernel_type == 'circular':
        radius = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        kernel = create_circular_kernel(radius, device=device)
    elif kernel_type == 'rectangular':
        if isinstance(kernel_size, int):
            kh, kw = kernel_size, kernel_size
        else:
            kh, kw = kernel_size
        kernel = create_rectangular_kernel(kh, kw, device=device)
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}. Use 'rectangular' or 'circular'.")
    
    # Apply dilation (maximum pooling)
    result = tensor
    for _ in range(iterations):
        # Pad to maintain size
        padding = (kernel.shape[-1] // 2, kernel.shape[-1] // 2, 
                   kernel.shape[-2] // 2, kernel.shape[-2] // 2)
        padded = F.pad(result, padding, mode='constant', value=0)  # Pad with 0s (background)
        
        # Unfold to get patches
        patches = F.unfold(
            padded, 
            kernel_size=(kernel.shape[-2], kernel.shape[-1]),
            stride=1
        )  # (bs, kernel_h*kernel_w, h*w)
        
        # Reshape kernel to match
        kernel_flat = kernel.view(-1)  # (kernel_h*kernel_w,)
        
        # Apply kernel mask: only consider kernel positions
        masked_patches = patches * kernel_flat.unsqueeze(0).unsqueeze(-1)  # (bs, kernel_h*kernel_w, h*w)
        
        # For dilation, we want maximum where kernel is 1
        # Set non-kernel positions to 0 (so they don't affect max)
        kernel_mask = kernel_flat.unsqueeze(0).unsqueeze(-1)  # (1, kernel_h*kernel_w, 1)
        masked_patches = torch.where(
            kernel_mask > 0, 
            masked_patches, 
            torch.zeros_like(masked_patches)
        )
        
        # Take maximum over kernel positions
        result = torch.max(masked_patches, dim=1, keepdim=True)[0]  # (bs, 1, h*w)
        result = result.view(bs, 1, h, w)
    
    # Remove channel dimension
    result = result.squeeze(1)
    
    return result.to(dtype)


def opening(
    tensor: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]] = 3,
    kernel_type: str = 'rectangular',
    iterations: int = 1
) -> torch.Tensor:
    """
    Apply morphological opening (erosion followed by dilation).
    
    Opening removes small objects and smooths boundaries while preserving
    the shape and size of larger objects.
    
    Args:
        tensor (torch.Tensor): Input binary tensor with shape (bs, h, w).
        kernel_size (int or tuple): Size of the structuring element. Default: 3.
        kernel_type (str): Type of kernel: 'rectangular' or 'circular'. Default: 'rectangular'.
        iterations (int): Number of times to apply opening. Default: 1.
    
    Returns:
        torch.Tensor: Opened binary tensor with same shape as input.
    
    Example:
        >>> x = torch.rand(2, 100, 100) > 0.5  # Noisy binary mask
        >>> opened = opening(x.float(), kernel_size=3)
        >>> print(opened.shape)  # (2, 100, 100)
    
    Notes:
        - Opening = Erosion then Dilation
        - Useful for removing small noise while preserving larger structures
        - Less aggressive than pure erosion
    """
    eroded = erode(tensor, kernel_size=kernel_size, kernel_type=kernel_type, iterations=iterations)
    opened = dilate(eroded, kernel_size=kernel_size, kernel_type=kernel_type, iterations=iterations)
    return opened


def closing(
    tensor: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]] = 3,
    kernel_type: str = 'rectangular',
    iterations: int = 1
) -> torch.Tensor:
    """
    Apply morphological closing (dilation followed by erosion).
    
    Closing fills small holes and connects nearby objects while preserving
    the shape and size of larger objects.
    
    Args:
        tensor (torch.Tensor): Input binary tensor with shape (bs, h, w).
        kernel_size (int or tuple): Size of the structuring element. Default: 3.
        kernel_type (str): Type of kernel: 'rectangular' or 'circular'. Default: 'rectangular'.
        iterations (int): Number of times to apply closing. Default: 1.
    
    Returns:
        torch.Tensor: Closed binary tensor with same shape as input.
    
    Example:
        >>> x = torch.rand(2, 100, 100) > 0.5  # Noisy binary mask with holes
        >>> closed = closing(x.float(), kernel_size=3)
        >>> print(closed.shape)  # (2, 100, 100)
    
    Notes:
        - Closing = Dilation then Erosion
        - Useful for filling small holes and connecting nearby regions
        - Less aggressive than pure dilation
    """
    dilated = dilate(tensor, kernel_size=kernel_size, kernel_type=kernel_type, iterations=iterations)
    closed = erode(dilated, kernel_size=kernel_size, kernel_type=kernel_type, iterations=iterations)
    return closed


def safe_closing(
    tensor: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]] = 3,
    kernel_type: str = 'rectangular',
    iterations: int = 1
) -> torch.Tensor:
    """
    Apply closing with safe padding to handle image boundaries correctly.
    
    This wrapper prevents artifacts at the borders when objects touch the edge,
    by using replicate padding that preserves boundary values for continuous-valued inputs.
    """
    
    # Determine pad size based on kernel size and iterations
    # We need enough padding to cover the dilation expansion
    if isinstance(kernel_size, int):
        kh, kw = kernel_size, kernel_size
    else:
        kh, kw = kernel_size
        
    if kernel_type == 'circular':
        radius = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        pad_h = pad_w = (2 * radius + 1) // 2 * iterations
    else:
        pad_h = (kh // 2) * iterations + 1
        pad_w = (kw // 2) * iterations + 1

    # 1. Pre-pad the input tensor with replicate mode for natural boundary handling
    # F.pad order is (left, right, top, bottom)
    padded_tensor = F.pad(tensor, (pad_w, pad_w, pad_h, pad_h), mode='replicate')

    # 2. Perform the standard closing operation
    # The internal erode/dilate will still pad, but they will act on our 0-padded area,
    # avoiding the boundary artifact.
    processed = closing(padded_tensor, kernel_size, kernel_type, iterations)

    # 3. Crop back to original size
    if pad_h > 0 and pad_w > 0:
        result = processed[..., pad_h:-pad_h, pad_w:-pad_w]
    elif pad_h > 0:
        result = processed[..., pad_h:-pad_h, :]
    elif pad_w > 0:
        result = processed[..., :, pad_w:-pad_w]
    else:
        result = processed

    return result


def safe_closing_with_dilate(
    tensor: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]] = 3,
    kernel_type: str = 'rectangular',
    iterations: int = 1,
) -> torch.Tensor:
    """
    Apply closing with safe padding to handle image boundaries correctly.
    
    This wrapper prevents artifacts at the borders when objects touch the edge,
    by using replicate padding that preserves boundary values for continuous-valued inputs.
    """
    
    # Determine pad size based on kernel size and iterations
    # We need enough padding to cover the dilation expansion
    if isinstance(kernel_size, int):
        kh, kw = kernel_size, kernel_size
    else:
        kh, kw = kernel_size
        
    if kernel_type == 'circular':
        radius = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        pad_h = pad_w = (2 * radius + 1) // 2 * iterations
    else:
        pad_h = (kh // 2) * iterations + 1
        pad_w = (kw // 2) * iterations + 1

    # 1. Pre-pad the input tensor with replicate mode for natural boundary handling
    # F.pad order is (left, right, top, bottom)
    padded_tensor = F.pad(tensor, (pad_w, pad_w, pad_h, pad_h), mode='replicate')

    # 2. Perform the standard closing operation
    # The internal erode/dilate will still pad, but they will act on our 0-padded area,
    # avoiding the boundary artifact.
    processed = closing(padded_tensor, kernel_size, kernel_type, max(iterations - 1, 1))
    processed = dilate(processed, kernel_size, kernel_type, 1)

    # 3. Crop back to original size
    if pad_h > 0 and pad_w > 0:
        result = processed[..., pad_h:-pad_h, pad_w:-pad_w]
    elif pad_h > 0:
        result = processed[..., pad_h:-pad_h, :]
    elif pad_w > 0:
        result = processed[..., :, pad_w:-pad_w]
    else:
        result = processed

    return result


def morphological_gradient(
    tensor: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]] = 3,
    kernel_type: str = 'rectangular'
) -> torch.Tensor:
    """
    Compute morphological gradient (dilation - erosion).
    
    The morphological gradient highlights the boundaries of objects.
    
    Args:
        tensor (torch.Tensor): Input binary tensor with shape (bs, h, w).
        kernel_size (int or tuple): Size of the structuring element. Default: 3.
        kernel_type (str): Type of kernel: 'rectangular' or 'circular'. Default: 'rectangular'.
    
    Returns:
        torch.Tensor: Gradient tensor with same shape as input.
    
    Example:
        >>> x = torch.zeros(2, 100, 100)
        >>> x[:, 30:70, 30:70] = 1  # Square region
        >>> edges = morphological_gradient(x, kernel_size=3)
        >>> print(edges.shape)  # (2, 100, 100)
    """
    dilated = dilate(tensor, kernel_size=kernel_size, kernel_type=kernel_type, iterations=1)
    eroded = erode(tensor, kernel_size=kernel_size, kernel_type=kernel_type, iterations=1)
    gradient = dilated - eroded
    return gradient


__all__ = [
    'erode', 'dilate', 'opening', 'closing', 'morphological_gradient',
    'create_circular_kernel', 'create_rectangular_kernel'
]
