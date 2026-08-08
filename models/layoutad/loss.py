import torch
import torch.nn.functional as F

def _ensure_batch_index(x, batch_index):
    if batch_index is None:
        return torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    return batch_index.to(dtype=torch.long, device=x.device)

def _per_graph_std_mean(x, batch_index):
    x = x.view(-1)
    batch_index = _ensure_batch_index(x, batch_index)
    if x.numel() == 0:
        return x.new_tensor(0.0)
    B = int(batch_index.max().item()) + 1
    vals = []
    for b in range(B):
        m = (batch_index == b)
        if m.any():
            xb = x[m]
            if xb.numel() >= 2:
                vals.append(xb.std())
    return torch.stack(vals).mean() if len(vals) > 0 else x.new_tensor(0.0)

@torch.no_grad()
def _per_graph_topbot_diff(scores, batch_index, top_frac=0.3):
    scores = scores.view(-1)
    batch_index = _ensure_batch_index(scores, batch_index)
    if scores.numel() == 0:
        return scores.new_zeros(0)
    B = int(batch_index.max().item()) + 1
    diffs = []
    for b in range(B):
        m = (batch_index == b)
        s = scores[m]
        n = s.numel()
        if n < 2: 
            continue
        k = max(1, min(int(n * top_frac), n // 2))
        if k == 0: 
            continue
        top_mean = torch.topk(s, k, largest=True).values.mean()
        bot_mean = torch.topk(s, k, largest=False).values.mean()
        diffs.append((top_mean - bot_mean).detach())
    if len(diffs) == 0:
        return scores.new_zeros(0)
    return torch.stack(diffs)

def _per_graph_mean(x, batch_index):
    x = x.view(-1)
    b = _ensure_batch_index(x, batch_index)
    if x.numel() == 0:
        return x.new_zeros(0)
    B = int(b.max().item()) + 1
    vals = []
    for i in range(B):
        m = (b == i)
        if m.any():
            vals.append(x[m].mean())
    return torch.stack(vals) if len(vals) > 0 else x.new_zeros(0)


def topbottom_contrastive_loss(
    scores,
    batch_index,
    top_frac: float = 0.2,
    temp: float = 0.5,
    margin_base: float = 0.2,
    margin_mult: float = 0.5,
    min_margin: float = 0.15,
    max_margin: float = 1.5,
):

    scores = scores.view(-1)
    if scores.numel() == 0:
        return scores.new_tensor(0.0)

    batch_index = _ensure_batch_index(scores, batch_index)
    B = int(batch_index.max().item()) + 1

    loss = scores.new_tensor(0.0)
    cnt = 0
    for b in range(B):
        m = (batch_index == b)
        s = scores[m]
        n = s.numel()
        if n < 2:
            continue

        k = max(1, int(n * top_frac))
        k = min(k, n // 2)  # disjoint
        if k == 0:
            continue

        top_mean = torch.topk(s, k, largest=True).values.mean()
        bot_mean = torch.topk(s, k, largest=False).values.mean()
        diff = top_mean - bot_mean

        std_g = s.std(unbiased=False) if n >= 2 else s.new_tensor(0.0)
        margin_g = (margin_base + margin_mult * std_g.detach()).clamp(min_margin, max_margin)

        logits = (diff - margin_g) / (temp + 1e-6)
        loss = loss + F.softplus(-logits)
        cnt += 1
        # print(diff, margin_g)

    if cnt == 0:
        return scores.new_tensor(0.0)
    return loss / cnt


def unsup_loss(
    node_outputs,
    edge_outputs,
    batch_index,
    edge_batch_index=None,
    lambda_rank_node: float = 1.0,
    lambda_rank_edge: float = 1.0,
    lambda_node: float = 0.005,
    lambda_edge: float = 0.005,
    reg_sigma: float = 1e-2,
    reg_std: float = 1e-2,
    node_top_frac: float = 0.40,
    node_temp: float = 15.0,
    node_margin_base: float = 5.0,
    node_margin_mult: float = 0.7,
    node_min_margin: float = 0.15,
    node_max_margin: float = 1.5,
    edge_top_frac: float = 0.35,
    edge_temp: float = 0.6,
    edge_margin_base: float = 4.0,
    edge_margin_mult: float = 1.0,
    edge_min_margin: float = 3.5,
    edge_max_margin: float = 10.0,
    adaptive_edge_margin: bool = True,
    adaptive_momentum: float = 0.6,
    adaptive_offset_in_temp: float = 0.8,
    state: dict | None = None,
):
    """
    node_outputs: (nll_g, nll_v, sigma_g_avg, sigma_v_avg, node_score)
    edge_outputs: (edge_nll, edge_sigma)
    """
    nll_g, nll_v, sigma_g, sigma_v, node_score = node_outputs
    edge_nll, edge_sigma = edge_outputs

    eb = _ensure_batch_index(edge_nll, edge_batch_index)
    B = int(eb.max().item()) + 1 if edge_nll.numel() > 0 else 0
    chunks = []
    with torch.no_grad():
        per_graph_means = []
        for b in range(B):
            m = (eb == b)
            s = edge_nll[m]
            if m.any():
                mu = s.mean().detach() 
                per_graph_means.append(mu)
                chunks.append((s - mu) + 0.0) 
    edge_for_rank = torch.cat(chunks, dim=0) if len(chunks) > 0 else edge_nll

    diag_edge = {}
    if adaptive_edge_margin:
        with torch.no_grad():
            diffs = _per_graph_topbot_diff(edge_for_rank, edge_batch_index, top_frac=edge_top_frac)
            if diffs.numel() > 0:
                batch_stat = diffs.median() 
                if state is None:
                    ema = batch_stat
                else:
                    ema_prev = state.get("edge_ema_diff", batch_stat)
                    ema = adaptive_momentum * ema_prev + (1 - adaptive_momentum) * batch_stat
                    state["edge_ema_diff"] = ema
                base = (ema - adaptive_offset_in_temp * edge_temp).clamp(min=0.0)

                e_min = (base - 0.2 * edge_temp).clamp(min=0.0)
                e_max = (ema + 2.0 * edge_temp)

                edge_margin_base = float(base.item())
                edge_min_margin  = float(e_min.item())
                edge_max_margin  = float(e_max.item())

                diag_edge.update({
                    "edge_diff_median": float(batch_stat.item()),
                    "edge_ema_diff": float(ema.item()),
                    "edge_margin_base_now": edge_margin_base,
                    "edge_min_margin_now": edge_min_margin,
                    "edge_max_margin_now": edge_max_margin,
                })

    L_rank_node = topbottom_contrastive_loss(
        node_score, batch_index,
        top_frac=node_top_frac, temp=node_temp,
        margin_base=node_margin_base, margin_mult=node_margin_mult,
        min_margin=node_min_margin, max_margin=node_max_margin
    )
    L_rank_edge = topbottom_contrastive_loss(
        edge_for_rank, edge_batch_index,
        top_frac=edge_top_frac, temp=edge_temp,
        margin_base=edge_margin_base, margin_mult=edge_margin_mult,
        min_margin=edge_min_margin, max_margin=edge_max_margin
    )

    L_node = (nll_g.mean() + nll_v.mean())
    L_edge = edge_nll.mean()

    L_sigma = ((sigma_g - 1.0) ** 2).mean() + ((sigma_v - 1.0) ** 2).mean() + ((edge_sigma - 1.0) ** 2).mean()

    std_node = _per_graph_std_mean(node_score, batch_index)
    std_edge = _per_graph_std_mean(edge_nll, edge_batch_index)
    L_std = (std_node - 1.0).pow(2) + (std_edge - 1.0).pow(2)

    total = (
        lambda_rank_node * L_rank_node
        + lambda_rank_edge * L_rank_edge
        + lambda_node * L_node
        + lambda_edge * L_edge
        + reg_sigma * L_sigma
        + reg_std * L_std
    )

    def _item(x):
        try:
            return float(x.detach().item())
        except Exception:
            return float('nan')
    
    # print(f"rank node={_item(L_rank_node):.4f}, rank edge={_item(L_rank_edge):.4f}, node={_item(L_node):.4f}, edge={_item(L_edge):.4f} std={_item(L_std):.4f}, sigma={_item(L_sigma):.4f}, total={_item(total):.4f}")

    return {
        "loss_total": total,
        "L_rank_node": _item(L_rank_node),
        "L_rank_edge": _item(L_rank_edge),
        "L_node": _item(L_node),
        "L_edge": _item(L_edge),
        "L_sigma": _item(L_sigma),
        "L_std": _item(L_std),
    }

