import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG2PI = math.log(2 * math.pi)

class NodeMDNHead(nn.Module):
    def __init__(self, d_model: int, n_components: int = 5, hidden: int = 256,
                 sigma_min: float = 0.3, sigma_max: float = 3.0,
                 cov_type: str = 'diag'):
        super().__init__()
        assert cov_type in ['diag', 'full'], "cov_type must be 'diag' or 'full'"
        self.n_components = n_components
        self.d_model = d_model
        self.cov_type = cov_type
        self.log_s_min = math.log(sigma_min)
        self.log_s_max = math.log(sigma_max)

        # 输入维度包含 global context
        in_dim = 2 * d_model 

        # vis -> geo
        self.net_g = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )

        # geo -> vis
        self.net_v = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )

        self.fc_mu_g = nn.Linear(hidden, n_components * d_model)
        self.fc_pi_g = nn.Linear(hidden, n_components)
        self.fc_mu_v = nn.Linear(hidden, n_components * d_model)
        self.fc_pi_v = nn.Linear(hidden, n_components)

        if cov_type == 'diag':
            self.fc_cov_g = nn.Linear(hidden, n_components * d_model)
            self.fc_cov_v = nn.Linear(hidden, n_components * d_model)
        else:
            self.fc_cov_g = nn.Linear(hidden, n_components * (d_model * (d_model + 1) // 2))
            self.fc_cov_v = nn.Linear(hidden, n_components * (d_model * (d_model + 1) // 2))

    def _get_mixture_params(self, h, net_type: str):
        B = h.size(0)
        if net_type == 'g':
            mu = self.fc_mu_g(h).view(B, self.n_components, self.d_model)
            pi = F.softmax(self.fc_pi_g(h), dim=-1)
            cov_layer = self.fc_cov_g
        else:
            mu = self.fc_mu_v(h).view(B, self.n_components, self.d_model)
            pi = F.softmax(self.fc_pi_v(h), dim=-1)
            cov_layer = self.fc_cov_v

        if self.cov_type == 'diag':
            log_sigma_raw = cov_layer(h)
            half = 0.5 * (self.log_s_max - self.log_s_min)
            mid  = 0.5 * (self.log_s_max + self.log_s_min)
            log_sigma = mid + half * torch.tanh(log_sigma_raw)
            log_sigma = log_sigma.view(B, self.n_components, self.d_model)
            return pi, mu, log_sigma
        else:
            L_flat = cov_layer(h)
            L = torch.zeros(B, self.n_components, self.d_model, self.d_model, device=h.device)
            tril_idx = torch.tril_indices(self.d_model, self.d_model, device=h.device)
            L[:, :, tril_idx[0], tril_idx[1]] = L_flat.view(B, self.n_components, -1)
            diag_idx = torch.arange(self.d_model, device=h.device)
            L[:, :, diag_idx, diag_idx] = torch.exp(L[:, :, diag_idx, diag_idx]).clamp(
                math.exp(self.log_s_min), math.exp(self.log_s_max)
            )
            return pi, mu, L

    def _nll_diag(self, target, pi, mu, log_sigma):
    
        target = target.unsqueeze(1).expand_as(mu)
        inv_var = torch.exp(-2 * log_sigma)
        log_prob = -0.5 * (((target - mu) ** 2) * inv_var).sum(-1)
        log_prob -= 0.5 * (self.d_model * LOG2PI + 2 * log_sigma.sum(-1))
        log_prob = torch.log(pi + 1e-8) + log_prob
        nll = -torch.logsumexp(log_prob, dim=-1)
        nll = torch.relu(nll)
        return nll

    def _nll_full(self, target, pi, mu, L):
        N, K, D = mu.shape
        x = target.unsqueeze(1) - mu                     # [N, K, D]

        L_flat = L.reshape(N * K, D, D).contiguous()     # [N*K, D, D]
        x_flat = x.reshape(N * K, D).contiguous()        # [N*K, D]

        y_flat = torch.linalg.solve_triangular(
            L_flat, x_flat.unsqueeze(-1), lower=True
        ).squeeze(-1)                                    # [N*K, D]

        # log|Σ| = 2 * sum(log(diag(L)))
        diag_L = torch.diagonal(L_flat, dim1=-2, dim2=-1)          # [N*K, D]
        log_det_flat = 2.0 * torch.log(diag_L).sum(-1)

        quad_flat = (y_flat ** 2).sum(-1)                           # [N*K]
        log_prob_flat = -0.5 * (D * LOG2PI + quad_flat) - log_det_flat  # [N*K]
        log_prob = log_prob_flat.view(N, K)                         # [N, K]

        log_mix = torch.log(pi + 1e-8)                              # [N, K]
        nll = -torch.logsumexp(log_prob + log_mix, dim=-1)          # [N]
        # nll = torch.relu(nll)
        return nll

    def forward(self,
                z_geo: torch.Tensor,       # [N, D_model]
                z_vis: torch.Tensor,       # [N, D_model]
                global_feat: torch.Tensor, # [B, D_model]
                batch_index: torch.Tensor  # [N], graph id for each node
                ):
        gctx = global_feat[batch_index]              # [N, D_model]
        cond_g = torch.cat([z_vis, gctx], dim=-1)    # vis->geo : [N, 2*D_model]
        cond_v = torch.cat([z_geo, gctx], dim=-1)    # geo->vis : [N, 2*D_model]

        # --- vis -> geo ---
        h_g = self.net_g(cond_g)
        params_g = self._get_mixture_params(h_g, 'g')

        # --- geo -> vis ---
        h_v = self.net_v(cond_v)
        params_v = self._get_mixture_params(h_v, 'v')
        
        if self.cov_type == 'diag':
            pi_g, mu_g, log_sigma_g = params_g
            pi_v, mu_v, log_sigma_v = params_v
            nll_g = self._nll_diag(z_geo, pi_g, mu_g, log_sigma_g)
            nll_v = self._nll_diag(z_vis, pi_v, mu_v, log_sigma_v)
            sigma_g_avg = torch.exp(log_sigma_g).mean(dim=(1, 2))
            sigma_v_avg = torch.exp(log_sigma_v).mean(dim=(1, 2))
        else:
            pi_g, mu_g, Lg = params_g
            pi_v, mu_v, Lv = params_v
            nll_g = self._nll_full(z_geo, pi_g, mu_g, Lg)
            nll_v = self._nll_full(z_vis, pi_v, mu_v, Lv)
            diag_g = torch.diagonal(Lg, dim1=-2, dim2=-1)
            diag_v = torch.diagonal(Lv, dim1=-2, dim2=-1)
            sigma_g_avg = diag_g.mean(dim=(1, 2))
            sigma_v_avg = diag_v.mean(dim=(1, 2))

        node_score = 0.6 * nll_g + 0.4 * nll_v
        return nll_g, nll_v, sigma_g_avg, sigma_v_avg, node_score
    
class EdgeMDNHead(nn.Module):
    def __init__(self, d_model: int, rel_dim: int, 
                 n_components: int = 5,
                 hidden: int = 256, 
                 sigma_min=0.1, sigma_max=5.0,
                 cov_type: str = 'diag'):  # diag full
        super().__init__()
        assert cov_type in ['diag', 'full'], "cov_type must be 'diag' or 'full'"
        self.n_components = n_components
        self.rel_dim = rel_dim
        self.cov_type = cov_type
        self.log_s_min = math.log(sigma_min)
        self.log_s_max = math.log(sigma_max)

        in_dim = 3 * d_model
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )

        self.fc_mu = nn.Linear(hidden, n_components * rel_dim)
        self.fc_pi = nn.Linear(hidden, n_components)

        if cov_type == 'diag':
            self.fc_cov = nn.Linear(hidden, n_components * rel_dim)  # log_sigma
        else:
            self.fc_cov = nn.Linear(hidden, n_components * (rel_dim * (rel_dim + 1) // 2))

    def _get_mixture_params(self, h):
        B = h.size(0)
        mu = self.fc_mu(h).view(B, self.n_components, self.rel_dim)
        pi = F.softmax(self.fc_pi(h), dim=-1) 

        if self.cov_type == 'diag':
            log_sigma_raw = self.fc_cov(h)
            half = 0.5 * (self.log_s_max - self.log_s_min)
            mid  = 0.5 * (self.log_s_max + self.log_s_min)
            log_sigma = mid + half * torch.tanh(log_sigma_raw)
            log_sigma = log_sigma.view(B, self.n_components, self.rel_dim)
            return pi, mu, log_sigma
        else:
            L_flat = self.fc_cov(h)
            L = torch.zeros(B, self.n_components, self.rel_dim, self.rel_dim, device=h.device)
            tril_idx = torch.tril_indices(self.rel_dim, self.rel_dim, device=h.device)
            L[:, :, tril_idx[0], tril_idx[1]] = L_flat.view(B, self.n_components, -1)
            diag_idx = torch.arange(self.rel_dim, device=h.device)
            L[:, :, diag_idx, diag_idx] = torch.exp(L[:, :, diag_idx, diag_idx]).clamp(
                math.exp(self.log_s_min), math.exp(self.log_s_max)
            )
            return pi, mu, L

    def _nll_diag(self, target, pi, mu, log_sigma):
        """
        target: [E, R]
        pi:     [E, K]
        mu:     [E, K, R]
        log_sigma: [E, K, R]
        """
        target = target.unsqueeze(1).expand_as(mu)
        inv_var = torch.exp(-2 * log_sigma)
        log_prob = -0.5 * (((target - mu) ** 2) * inv_var).sum(-1)
        log_prob -= 0.5 * (self.rel_dim * LOG2PI + 2 * log_sigma.sum(-1))
        log_prob = torch.log(pi + 1e-8) + log_prob
        nll = -torch.logsumexp(log_prob, dim=-1)
        # nll = torch.relu(nll)
        return nll

    def _nll_full(self, target, pi, mu, L):
        E, K, R = mu.shape
        x = target.unsqueeze(1) - mu                                # [E, K, R]

        L_flat = L.reshape(E * K, R, R).contiguous()                # [E*K, R, R]
        x_flat = x.reshape(E * K, R).contiguous()                   # [E*K, R]

        y_flat = torch.linalg.solve_triangular(
            L_flat, x_flat.unsqueeze(-1), lower=True
        ).squeeze(-1)                                               # [E*K, R]

        diag_L = torch.diagonal(L_flat, dim1=-2, dim2=-1)           # [E*K, R]
        log_det_flat = 2.0 * torch.log(diag_L).sum(-1)              # [E*K]
        quad_flat = (y_flat ** 2).sum(-1)                           # [E*K]

        log_prob_flat = -0.5 * (R * LOG2PI + quad_flat) - log_det_flat  # [E*K]
        log_prob = log_prob_flat.view(E, K)                         # [E, K]

        log_mix = torch.log(pi + 1e-8)
        nll = -torch.logsumexp(log_prob + log_mix, dim=-1)          # [E]
        nll = torch.relu(nll)
        return nll

    def forward(self,
                z_vis: torch.Tensor,        # [N, D]
                edge_index_vis: torch.Tensor,  # [2, E]
                edge_attr_geo: torch.Tensor,   # [E, R]
                global_feat: torch.Tensor,     # [B, D]
                batch_index: torch.Tensor      # [N]
                ):
        src, dst = edge_index_vis
        gctx = global_feat[batch_index[src]]               # [E, D]
        cond = torch.cat([z_vis[src], z_vis[dst], gctx], dim=-1)
        h = self.net(cond)

        params = self._get_mixture_params(h)
        if self.cov_type == 'diag':
            pi, mu, log_sigma = params
            nll = self._nll_diag(edge_attr_geo, pi, mu, log_sigma)
            sigma_avg = torch.exp(log_sigma).mean(dim=(1, 2))
        else:
            pi, mu, L = params
            nll = self._nll_full(edge_attr_geo, pi, mu, L)
            diag_vals = torch.diagonal(L, dim1=-2, dim2=-1)
            sigma_avg = diag_vals.mean(dim=(1, 2))

        return nll, sigma_avg
