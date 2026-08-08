import torch.nn as nn
from typing import Optional
import torch

def _bi_ok(x: torch.Tensor, batch_index: torch.Tensor, B: Optional[int] = None):
    assert x.dim() == 2, f"x should be [N,D], got {x.shape}"
    N = x.size(0)
    bi = batch_index.to(device=x.device, dtype=torch.long).view(-1).contiguous()
    assert bi.numel() == N, f"batch_index length {bi.numel()} != N {N}"
    mx = int(bi.max().item()); B_infer = mx + 1
    B = B_infer if B is None else int(B)
    return bi, B

def seg_sum(x: torch.Tensor, bi: torch.Tensor, B: int) -> torch.Tensor:
    N, D = x.shape
    out = torch.zeros(B, D, device=x.device, dtype=x.dtype)
    idx = bi.view(N, 1).expand(N, D)
    out.scatter_add_(0, idx, x)
    return out

def seg_max(x: torch.Tensor, bi: torch.Tensor, B: int) -> torch.Tensor:
    N, D = x.shape
    out = torch.full((B, D), float('-inf'), device=x.device, dtype=x.dtype)
    idx = bi.view(N, 1).expand(N, D)
    out = out.scatter_reduce(0, idx, x, reduce='amax', include_self=True)
    return out

def seg_mean(x: torch.Tensor, bi: torch.Tensor, B: int) -> torch.Tensor:
    s = seg_sum(x, bi, B)
    cnt = torch.bincount(bi, minlength=B).to(x.device).clamp_min(1)
    return s / cnt.view(B, 1)

def seg_softmax_1d(scores: torch.Tensor, bi: torch.Tensor, B: int, eps: float = 1e-12) -> torch.Tensor:
    # scores: [N,1]
    s32 = scores.float()
    gmax = seg_max(s32, bi, B)           # [B,1]
    shifted = s32 - gmax[bi]             # [N,1]
    expv = torch.exp(shifted)            # [N,1]
    gsum = seg_sum(expv, bi, B)          # [B,1]
    soft = expv / (gsum[bi] + eps)       # [N,1]
    return soft.to(scores.dtype)

class AdaptiveFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, z_geo, z_vis):
        fusion_input = torch.cat([z_geo, z_vis], dim=-1)  # [N, 2D]
        alpha = self.gate(fusion_input)                   # [N, 1]
        fused = alpha * z_geo + (1 - alpha) * z_vis       # [N, D]
        return fused, alpha


class GlobalAttentionPool(nn.Module):
    def __init__(self, d_model, hidden=128):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x, batch_index, B: Optional[int]=None):
        bi, B = _bi_ok(x, batch_index, B)
        attn_logits = self.att(x)               # [N,1]
        attn = seg_softmax_1d(attn_logits, bi, B)  # [N,1]
        xw = x * attn                           # [N,D]
        pooled = seg_sum(xw, bi, B)             # [B,D]
        return pooled
    
class MultiScaleGlobalPool(nn.Module):
    def __init__(self, d_model, hidden=128, out_dim=None, dropout=0.1):
        super().__init__()
        self.att_pool = GlobalAttentionPool(d_model, hidden)
        self.fc = nn.Sequential(
            nn.Linear(3 * d_model, out_dim or d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x, batch_index, B: Optional[int]=None):
        bi, B = _bi_ok(x, batch_index, B)
        mean_feat = seg_mean(x, bi, B)          # [B,D]
        max_feat  = seg_max(x, bi, B)           # [B,D]
        att_feat  = self.att_pool(x, bi, B)     # [B,D]
        fused = torch.cat([mean_feat, max_feat, att_feat], dim=-1)
        return self.fc(fused)
