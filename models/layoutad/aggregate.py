import torch
from torch_scatter import scatter
import math

def score_aggregate(edge_index: torch.Tensor,
                    edge_score: torch.Tensor,
                    num_nodes: int,
                    mode: str = "logsumexp",
                    k_ratio: float = 0.2):

    i, j = edge_index
    scores = torch.cat([edge_score, edge_score], dim=0)
    nodes = torch.cat([i, j], dim = 0)

    if mode == "max":
        node_score, _ = scatter(scores, nodes, dim=0, dim_size=num_nodes, reduce='max')
    
    elif mode == "topk":
        node_score = torch.zeros(num_nodes, device=edge_score.device)
        for n in range(num_nodes):

            mask = (nodes == n)
            if not mask.any():
                continue
            vals = scores[mask]
            k = max(1, int(k_ratio * vals.numel()))
            topk_vals, _ = torch.topk(vals, k)
            node_score[n] = topk_vals.mean()
    
    elif mode == 'logsumexp':
        node_score = torch.zeros(num_nodes, device=edge_score.device)
        for n in range(num_nodes):
            mask = (nodes == n)
            if not mask.any():
                continue
            vals = -scores[mask]
            m = vals.max()
            node_score[n] = -(m + torch.log(torch.exp(vals - m).sum()))
    
    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")

    return node_score