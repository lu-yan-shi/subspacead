import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

def mlp(d_in, d_out, hidden=None, act=nn.ReLU, bn=True, dropout=0.0):
    layers, last = [], d_in
    if hidden is not None:
        hs = hidden if isinstance(hidden, (list, tuple)) else [hidden]
        for h in hs:
            layers += [nn.Linear(last, h)]
            if bn: layers += [nn.LayerNorm(h)]
            layers += [act()]
            if dropout > 0: layers += [nn.Dropout(dropout)]
            last = h
    layers += [nn.Linear(last, d_out)]
    return nn.Sequential(*layers)

class Pos2DEncoder(nn.Module):
    def __init__(self, d_model: int, hidden: Optional[int] = None,
                mode: str = 'mlp', fourier_bands: int = 6):
        super().__init__()
        self.mode = mode
        self.d_model = d_model
        self.fourier_bands = fourier_bands
        if mode == 'mlp':
            self.proj = mlp(4, d_model, hidden=hidden or d_model // 2, dropout=0.0)
        elif mode == 'fourier':
            in_dim = 4 * (2 * fourier_bands)
            self.register_buffer('freqs', torch.logspace(math.log10(1.0), math.log10(1000.0), fourier_bands))
            self.lin = nn.Linear(in_dim, d_model)
        else:
            raise ValueError("Unknown pos2d mode")
    def forward(self, pos2d: torch.Tensor) -> torch.Tensor:
        if self.mode == 'mlp':
            return self.proj(pos2d.float())
        else:
            B, L, _ = pos2d.shape
            Freq = self.fourier_bands
            x = pos2d.unsqueeze(-1) * self.freqs.view(1, 1, 1, Freq) * (2 * math.pi)  # [B,L,4,F]
            pe = torch.cat([torch.sin(x), torch.cos(x)], dim=-1).view(B, L, 4 * (2 * Freq))
            return self.lin(pe)

class MHAWithBias(nn.Module):
    def __init__(self, d_model, nhead, attn_dropout=0.1, proj_dropout=0.1, bias=True):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def _reshape(self, x): # [B, L, D] -> [B, H, L, Dh]
        B, L, _ = x.shape
        return x.view(B, L, self.nhead, self.d_head).transpose(1, 2).contiguous()

    def forward(self, q, k, v,
                rel_bias_heads: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None):
        B, Lq, _ = q.shape
        H, Dh = self.nhead, self.d_head

        Q = self._reshape(self.q_proj(q))  # [B, H, Lq, Dh]
        K = self._reshape(self.k_proj(k))  # [B, H, Lk, Dh]
        V = self._reshape(self.v_proj(v))  # [B, H, Lv, Dh]


        Qh = Q.reshape(B * H, Lq, Dh)
        Kh = K.reshape(B * H, K.size(2), Dh)
        Vh = V.reshape(B * H, V.size(2), Dh)

        attn_mask = None
        if rel_bias_heads is not None:
            # scaled_dot_product_attention 的 attn_mask 形状是 [B*H, Lq, Lk]
            attn_mask = rel_bias_heads.reshape(B * H, Lq, Kh.size(1)).to(Q.dtype)

        if key_padding_mask is not None:
            neg_inf = torch.finfo(Q.dtype).min
            pad = key_padding_mask.to(Q.device).unsqueeze(1).expand(B, Lq, Kh.size(1))  # [B, Lq, Lk]
            pad = pad.reshape(B, 1, Lq, Kh.size(1)).expand(B, H, Lq, Kh.size(1)).reshape(B * H, Lq, Kh.size(1))
            pad_mask = torch.zeros_like(pad, dtype=Q.dtype).masked_fill(pad, neg_inf)
            attn_mask = pad_mask if attn_mask is None else (attn_mask + pad_mask)

        out = F.scaled_dot_product_attention(
            Qh, Kh, Vh,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0
        )  # [B*H, Lq, Dh]

        out = out.reshape(B, H, Lq, Dh).transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        out = self.o_proj(out)
        out = self.proj_drop(out)
        return out
    
class RelationToBias(nn.Module):
    def __init__(self, rel_dim: int, nhead: int, hidden=None, dropout=0.0):
        super().__init__()
        self.edge2bias = mlp(rel_dim, nhead, hidden=hidden, dropout=dropout)

    @staticmethod
    def empty_bias(B, H, Lq, Lk, device, dtype):
        return torch.zeros(B, H, Lq, Lk, device=device, dtype=dtype)
    
    def _square_bias_single(self,
                           batch: Batch,
                           N_nodes: torch.Tensor,     # [B] per-graph node count (no global)
                           L_max: int,
                           nhead: int,
                           edge_attr: Optional[torch.Tensor],  # [E_total, R]
                           attn_dtype: torch.dtype
                           ) -> torch.Tensor:
        
        device = batch.edge_index.device
        B = int(batch.ptr.numel() - 1)
        bias = self.empty_bias(B, nhead, L_max, L_max, device=device, dtype=attn_dtype)

        if edge_attr is None or edge_attr.numel() == 0 or batch.edge_index.numel() == 0:
            return bias

        edge_graph = batch.batch[batch.edge_index[0]]  # [E_total]
        ptr = batch.ptr.to(device)
        start = ptr[edge_graph]
        i_loc = (batch.edge_index[0].to(device) - start).long()
        j_loc = (batch.edge_index[1].to(device) - start).long()

        Lb = (N_nodes * 1).to(device)
        valid = (i_loc < Lb[edge_graph]) & (j_loc < Lb[edge_graph])
        if not valid.any():
            return bias
        
        i_loc = i_loc[valid]; j_loc = j_loc[valid]; g = edge_graph[valid]
        hb = self.edge2bias(edge_attr[valid].to(device))

        P = i_loc.numel()
        h_idx = torch.arange(nhead, device=device).view(1, nhead).expand(P, nhead).reshape(-1)
        i_idx = i_loc.view(-1, 1).expand(P, nhead).reshape(-1)
        j_idx = j_loc.view(-1, 1).expand(P, nhead).reshape(-1)
        g_idx = g.view(-1, 1).expand(P, nhead).reshape(-1)
        val  = hb.view(P, nhead).reshape(-1)

        # bias.index_put_((g_idx, h_idx, i_idx, j_idx), val, accumulate=True)

        flat = bias.view(-1)
        lin = (((g_idx * bias.shape[1]) + h_idx) * bias.shape[2] + i_idx) * bias.shape[3] + j_idx
        flat = flat.scatter_add(0, lin, val)
        bias = flat.view_as(bias)

        return bias

    def forward_single(self, batch: Batch, N_nodes: torch.Tensor, L_max: int,
                       nhead: int, edge_attr: Optional[torch.Tensor],
                       attn_dtype: torch.dtype) -> torch.Tensor:
        return self._square_bias_single(batch, N_nodes, L_max, nhead, edge_attr, attn_dtype)

    def forward_cross(self,
                      bias_square_of_keys: torch.Tensor, 
                      Lq: int, Lk: int) -> torch.Tensor:
        B, H, _, _ = bias_square_of_keys.shape
        bias_keys = bias_square_of_keys[:, :, :Lk, :Lk]  # [B,H,Lk,Lk]
        bias_keys = bias_keys.mean(dim=2, keepdim=True)  # [B,H,1,Lk] -> 聚合到仅依赖 Key
        bias_cross = bias_keys.expand(B, H, Lq, Lk)      # [B,H,Lq,Lk]
        return bias_cross

def build_padded_streams(batch: Batch,
                         x_geo_nodes: torch.Tensor,  # [N, D]
                         x_vis_nodes: torch.Tensor,  # [N, D]
                         pos4_src: Optional[torch.Tensor] = None  # 一般可用 batch.x_geo[:, :4]
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int], int]:
    device = x_geo_nodes.device
    B = int(batch.ptr.numel() - 1)
    D = x_geo_nodes.size(1)
    ptr = batch.ptr.to(device)
    n_nodes = (ptr[1:] - ptr[:-1]).to(torch.long)  # [B]
    L_max = int(n_nodes.max().item()) if B > 0 else 0

    xg = torch.zeros((B, L_max, D), device=device, dtype=x_geo_nodes.dtype)
    xv = torch.zeros((B, L_max, D), device=device, dtype=x_vis_nodes.dtype)
    key_pad = torch.ones(B, L_max, device=device, dtype=torch.bool)
    if pos4_src is None:
        pos4_src = torch.zeros(x_geo_nodes.size(0), 4, device=device, dtype=x_geo_nodes.dtype)
    pos_b = torch.zeros(B, L_max, 4, device=device, dtype=pos4_src.dtype)

    for b in range(B):
        st, ed = int(ptr[b].item()), int(ptr[b+1].item())
        Nb = ed - st
        if Nb <= 0:
            continue 
        xg[b, :Nb, :] = x_geo_nodes[st:ed, :]
        xv[b, :Nb, :] = x_vis_nodes[st:ed, :]
        key_pad[b, :Nb] = False
        pos_b[b, :Nb] = pos4_src[st:ed]

    N_nodes_list = [int(n.item()) for n in n_nodes]
    return xg, xv, pos_b, key_pad, N_nodes_list, L_max

class CrossGraphTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff=1024, dropout=0.1,
                 rel_dim_geo=32, rel_dim_vis=32, use_cross_bias=True):
        super().__init__()
        self.nhead = nhead
        self.use_cross_bias = use_cross_bias

        # self attention
        self.geo_self = MHAWithBias(d_model, nhead, dropout, dropout)
        self.vis_self = MHAWithBias(d_model, nhead, dropout, dropout)

        # cross attention
        self.geo_from_vis = MHAWithBias(d_model, nhead, dropout, dropout)  # Q=geo, KV=vis
        self.vis_from_geo = MHAWithBias(d_model, nhead, dropout, dropout)  # Q=vis, KV=geo

        # 边特征 → bias
        self.geo_rel2bias = RelationToBias(rel_dim_geo, nhead, hidden=d_model//2, dropout=dropout)
        self.vis_rel2bias = RelationToBias(rel_dim_vis, nhead, hidden=d_model//2, dropout=dropout)

        # Norm + FFN
        self.ln_geo1 = nn.LayerNorm(d_model)
        self.ln_vis1 = nn.LayerNorm(d_model)
        self.ln_geo2 = nn.LayerNorm(d_model)
        self.ln_vis2 = nn.LayerNorm(d_model)

        self.ffn_geo = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
        self.ffn_vis = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
    
    def forward(self,
                x_geo: torch.Tensor, x_vis: torch.Tensor,          # [B,Lg,D], [B,Lv,D]
                batch: Batch,
                N_nodes: torch.Tensor,                             # [B]
                key_pad_geo: torch.Tensor, key_pad_vis: torch.Tensor,  # [B,Lg], [B,Lv]
                Lg: int, Lv: int,
                edge_geo: Optional[torch.Tensor] = None,           # [E,Rg]
                edge_vis: Optional[torch.Tensor] = None            # [E,Rv]
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, Lg_now, D = x_geo.shape
        _, Lv_now, _ = x_vis.shape
        assert Lg_now == Lg and Lv_now == Lv

        # self attention
        bias_geo_sq = self.geo_rel2bias.forward_single(batch, N_nodes, Lg, self.nhead, edge_geo, x_geo.dtype) 
        bias_vis_sq = self.vis_rel2bias.forward_single(batch, N_nodes, Lv, self.nhead, edge_vis, x_vis.dtype)
        x_geo = x_geo + self.geo_self(self.ln_geo1(x_geo), self.ln_geo1(x_geo), self.ln_geo1(x_geo),
                                      rel_bias_heads=bias_geo_sq, key_padding_mask=key_pad_geo)
        x_vis = x_vis + self.vis_self(self.ln_vis1(x_vis), self.ln_vis1(x_vis), self.ln_vis1(x_vis),
                                      rel_bias_heads=bias_vis_sq, key_padding_mask=key_pad_vis)
        
        # cross attention
        if self.use_cross_bias:
            bias_g2v = self.vis_rel2bias.forward_cross(bias_vis_sq, Lq=Lg, Lk=Lv)  # [B,H,Lg,Lv]
            bias_v2g = self.geo_rel2bias.forward_cross(bias_geo_sq, Lq=Lv, Lk=Lg)  # [B,H,Lv,Lg]
        else:
            bias_g2v = None
            bias_v2g = None

        cross_x_geo = x_geo + self.geo_from_vis(self.ln_geo2(x_geo), self.ln_vis2(x_vis), self.ln_vis2(x_vis),
                                          rel_bias_heads=bias_g2v, key_padding_mask=key_pad_vis)
        cross_x_vis = x_vis + self.vis_from_geo(self.ln_vis2(x_vis), self.ln_geo2(x_geo), self.ln_geo2(x_geo),
                                          rel_bias_heads=bias_v2g, key_padding_mask=key_pad_geo)
        
        # FFN
        out_geo = cross_x_geo + 0.1 * self.ffn_geo(self.ln_geo2(cross_x_geo))
        out_vis = cross_x_vis + 0.1 * self.ffn_vis(self.ln_vis2(cross_x_vis))
        return out_geo, out_vis

class CrossGraphTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, 
                 d_ff=1024, dropout=0.1, 
                 rel_dim_geo=32, rel_dim_vis=32,
                 use_cross_bias=True):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossGraphTransformerLayer(d_model, nhead, d_ff, dropout,
                                       rel_dim_geo, rel_dim_vis, use_cross_bias)
            for _ in range(num_layers)
        ])
        self.final_ln_geo = nn.LayerNorm(d_model)
        self.final_ln_vis = nn.LayerNorm(d_model)

    def forward(self,
                x_geo_b: torch.Tensor, x_vis_b: torch.Tensor,     # [B,Lg,D], [B,Lv,D]
                batch: Batch,
                N_nodes: torch.Tensor,                            # [B]
                key_pad_geo: torch.Tensor, key_pad_vis: torch.Tensor,  # [B,Lg], [B,Lv]
                Lg: int, Lv: int,
                edge_geo: Optional[torch.Tensor] = None,
                edge_vis: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        xg, xv = x_geo_b, x_vis_b
        for layer in self.layers:
            xg, xv = layer(xg, xv, batch, N_nodes, key_pad_geo, key_pad_vis, Lg, Lv, edge_geo, edge_vis)
        return self.final_ln_geo(xg), self.final_ln_vis(xv)

class CrossGraphLayoutTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_enc_layers=4,
                 d_ff=1024, dropout=0.1,
                 rel_dim_geo=32, rel_dim_vis=32,
                 pos2d_mode='mlp',
                 use_cross_bias=True):
        super().__init__()

        self.modal_emb = nn.Embedding(2, d_model)  # 0-geo, 1-vis
        self.pos2d = Pos2DEncoder(d_model, hidden=d_model//2, mode=pos2d_mode)
        self.encoder = CrossGraphTransformerEncoder(
            d_model, nhead, num_enc_layers, d_ff, dropout, 
            rel_dim_geo=rel_dim_geo, rel_dim_vis=rel_dim_vis,
            use_cross_bias=use_cross_bias
        )
        
    def forward(self,
                batch: Batch,
                x_geo: torch.Tensor,                 # [N, D]
                x_vis: torch.Tensor,                 # [N, D]
                edge_geo: Optional[torch.Tensor] = None,  # [E, Rg]
                edge_vis: Optional[torch.Tensor] = None   # [E, Rv]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        pos4 = batch.x_geo[:, :4]
        xg_b, xv_b, pos_b, key_pad, N_nodes_list, L_max = build_padded_streams(batch, x_geo, x_vis, pos4)
        key_pad_geo = key_pad
        key_pad_vis = key_pad
        Lg, Lv = L_max, L_max

        B, L, D = xg_b.size()
        xg_b = xg_b + self.modal_emb(torch.zeros(B, L, dtype=torch.long, device=xg_b.device))
        xv_b = xv_b + self.modal_emb(torch.ones(B, L, dtype=torch.long, device=xv_b.device))
        pos_enc = self.pos2d(pos_b)
        xg_b = xg_b + pos_enc
        xv_b = xv_b + pos_enc

        N_nodes = torch.tensor(N_nodes_list, dtype=torch.long, device=xg_b.device)
        out_geo_b, out_vis_b = self.encoder(
            xg_b, xv_b, batch, N_nodes,
            key_pad_geo, key_pad_vis, Lg, Lv,
            edge_geo=edge_geo, edge_vis=edge_vis
        )

        out_geo_nodes, out_vis_nodes = [], []
        ptr = batch.ptr
        for b in range(B):
            st, ed = int(ptr[b].item()), int(ptr[b + 1].item())
            Nb = ed - st
            if Nb <= 0:
                continue
            out_geo_nodes.append(out_geo_b[b, :Nb])
            out_vis_nodes.append(out_vis_b[b, :Nb])
        out_geo = torch.cat(out_geo_nodes, dim=0) if out_geo_nodes else x_geo.new_zeros((0, D))
        out_vis = torch.cat(out_vis_nodes, dim=0) if out_vis_nodes else x_vis.new_zeros((0, D))

        return out_geo, out_vis