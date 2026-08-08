import numpy as np
import cv2, torch
from torch_scatter import scatter_mean

def norm_per_graph(scores, batch_ids, eps=1e-6):
    means = scatter_mean(scores, batch_ids, dim=0)                # [num_graph]
    means_exp = means[batch_ids]                                  # [N]
    sq_means = scatter_mean(scores * scores, batch_ids, dim=0)    # E[x^2]
    vars_exp = (sq_means - means * means).clamp_min(eps)          # Var = E[x^2] - E[x]^2
    stds_exp = vars_exp.sqrt()[batch_ids]                         # [N]

    z = (scores - means_exp) / stds_exp
    return torch.sigmoid(z)

def build_pixel_score_map(node_masks, node_scores, H, W, 
                          mapping: str = "soft", sigma: float = 5.0):

    score_map = np.zeros((H, W), dtype=np.float32)
    if node_masks is None or len(node_masks) == 0:
        return score_map

    for j, m in enumerate(node_masks):
        if j >= len(node_scores):
            break
        s = float(node_scores[j])
        if mapping == "hard":
            score_map[m > 0] = np.maximum(score_map[m > 0], s)
        else:
            mask = (m > 0).astype(np.uint8)
            if mask.sum() == 0:
                continue

            dist = cv2.distanceTransform((1 - mask).astype(np.uint8), cv2.DIST_L2, 3)
            soft = np.exp(-dist / max(sigma, 1e-3))
            soft = soft / (soft.max() + 1e-8)
            score_map = np.maximum(score_map, soft * s)

    return score_map