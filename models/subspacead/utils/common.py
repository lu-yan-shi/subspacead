import numpy as np
import torch


def min_max_norm(
    x: torch.Tensor | np.ndarray, eps: float = 1e-8
) -> torch.Tensor | np.ndarray:
    """Min-max normalization on a tensor or numpy array."""
    is_torch = torch.is_tensor(x)

    if is_torch:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_min = torch.amin(x, dim=(-1, -2), keepdim=True)
        x_max = torch.amax(x, dim=(-1, -2), keepdim=True)
        x_norm = (x - x_min) / (x_max - x_min + eps)
        return x_norm.clamp(0.0, 1.0)
    else:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_min = np.min(x, axis=(-1, -2), keepdims=True)
        x_max = np.max(x, axis=(-1, -2), keepdims=True)
        x_norm = (x - x_min) / (x_max - x_min + eps)
        return np.clip(x_norm, 0.0, 1.0)
