import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data, Batch

from .gnn import DualTowerGNNEncoder
from .transformer import CrossGraphLayoutTransformer
from .glb import MultiScaleGlobalPool, AdaptiveFusion
from .detector import NodeMDNHead, EdgeMDNHead
from .aggregate import score_aggregate
from .tools import norm_per_graph

class LayoutAD(nn.Module):
    def __init__(self,
                d_geo: int = 17,   
                d_vis: int = 1024,
                e_geo_raw: int = 9,
                e_vis_raw: int = 2,
                d_model: int = 256,
                rel_dim_geo=32, rel_dim_vis=32,
                nhead: int = 8,    
                num_enc_layers: int = 4, 
                num_gnn_layers: int = 2,
                d_ff: int = 1024,   
                dropout: float = 0.1,
                pos2d_mode: str = 'mlp',
                noise_std: float = 0.01,
                n_components_node: int = 5,
                n_components_edge: int = 8,
                ):
        super().__init__()

        # GNN
        self.gnn = DualTowerGNNEncoder(
           d_geo=d_geo, d_vis=d_vis, e_geo_raw=e_geo_raw, e_vis_raw=e_vis_raw,
           dropout=dropout, num_layers=num_gnn_layers
        )

        # noise
        self.noise_std = noise_std

        # Transformer
        self.transformer = CrossGraphLayoutTransformer(
            d_model=d_model,
            nhead=nhead,
            num_enc_layers=num_enc_layers,
            rel_dim_geo=rel_dim_geo,
            rel_dim_vis=rel_dim_vis,
            d_ff=d_ff,
            dropout=dropout,
            pos2d_mode=pos2d_mode
        )

        # Global
        self.fusion = AdaptiveFusion(d_model=d_model)
        self.global_pool = MultiScaleGlobalPool(d_model=d_model)
        
        # Detector - node
        self.node_detector = NodeMDNHead(d_model, hidden=256, n_components=n_components_node)
        self.edge_detector = EdgeMDNHead(d_model, e_geo_raw, hidden=256, n_components=n_components_edge)

        # alpha
        self.alpha = 0.6
        self.register_buffer("mu_node", torch.zeros(1))
        self.register_buffer("sigma_node", torch.ones(1))
        self.register_buffer("mu_edge", torch.zeros(1))
        self.register_buffer("sigma_edge", torch.ones(1))
        self.edge_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.edge_bias  = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, batch: Batch):
        
        # gnn 
        xg, xs, eg, es = self.gnn(batch)

        # noise
        if self.training and self.noise_std > 0:
            xg = xg + torch.randn_like(xg) * self.noise_std
            xs = xs + torch.randn_like(xs) * self.noise_std
            eg = eg + torch.randn_like(eg) * (self.noise_std * 0.5)
            es = es + torch.randn_like(es) * (self.noise_std * 0.5)

        # transformer
        out_geo, out_vis = self.transformer(batch, xg, xs, eg, es)

        # global
        node_embed, alpha = self.fusion(out_geo, out_vis)
        B = int(batch.ptr.numel() - 1)
        global_feat = self.global_pool(node_embed, batch.batch, B)

        # detector - node and edge
        node_output = self.node_detector(out_geo, out_vis, global_feat, batch.batch)
        nll_g, nll_v, sigma_g_avg, sigma_v_avg, node_score = node_output
        edge_output = self.edge_detector(out_vis, batch.edge_index, batch.edge_attr_geo, global_feat, batch.batch)
        nll, sigma_avg = edge_output

        edge_to_node_score = score_aggregate(batch.edge_index, nll, xg.size(0), 'topk')

        # node_score_norm = norm_per_graph(node_score, batch.batch)
        # edge_score_norm = norm_per_graph(edge_to_node_score, batch.batch)
        # node_score_norm = torch.sigmoid((node_score - self.mu_node.detach()) / (self.sigma_node.detach() + 1e-6))
        # edge_score_norm = torch.sigmoid((edge_to_node_score - self.mu_edge.detach()) / (self.sigma_edge.detach() + 1e-6))

        if self.training:
            node_score_norm = torch.sigmoid((node_score - self.mu_node) / (self.sigma_node + 1e-6))
            edge_score_norm = torch.sigmoid((edge_to_node_score - self.mu_edge) / (self.sigma_edge + 1e-6))
        else:
            # print(1)
            node_score_norm = norm_per_graph(node_score, batch.batch)
            edge_score_norm = norm_per_graph(edge_to_node_score, batch.batch)

        final_score = (1 - self.alpha) * node_score_norm + self.alpha * edge_score_norm
        assert (final_score >= 0).all(), f"{node_score_norm}, {edge_score_norm}"

        return {
            'score': final_score,
            'node_output': node_output,
            'edge_output': edge_output
        }

