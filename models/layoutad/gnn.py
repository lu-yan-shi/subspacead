import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, MessagePassing   

def mlp(d_in, d_out, hidden=None, act=nn.ReLU, bn=True):
    layers, last = [], d_in
    if hidden is not None:
        for h in (hidden if isinstance(hidden, (list, tuple)) else [hidden]):
            layers += [nn.Linear(last, h)]
            if bn: layers += [nn.LayerNorm(h)]
            layers += [act()]
            last = h
    layers += [nn.Linear(last, d_out)]
    return nn.Sequential(*layers)

class EdgeProjector(nn.Module):
    def __init__(self, e_raw, edge_dim, hidden=None):
        super().__init__()
        self.net = mlp(e_raw, edge_dim, hidden)
    def forward(self, e):
        if e is None or e.numel() == 0:
            return None
        return self.net(e.float())

class EdgeMP(nn.Module):
    def __init__(self, d_node, d_edge_in, d_edge_out, hidden=128, dropout=0.0):
        super().__init__()
        fin = 2 * d_node + (d_edge_in or 0)
        self.mlp = mlp(fin, d_edge_out, hidden)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_edge_out)
        self.out_dim = d_edge_out
        self.use_e = (d_edge_in is not None)

    def forward(self, x, edge_index, e=None):
        E = edge_index.size(1)
        if E == 0:
            if e is not None:
                return e
            return x.new_zeros((0, self.out_dim))

        src, dst = edge_index[0], edge_index[1]
        x_j, x_i = x[src], x[dst]                 # [E, d_node], [E, d_node]
        parts = [x_i, x_j]
        if self.use_e and (e is not None):
            parts.append(e)
        upd = self.drop(self.mlp(torch.cat(parts, dim=-1)))  # [E, d_edge_out]

        if e is None or not self.use_e:
            return self.norm(upd)
        else:
            return self.norm(e + upd)

class NodeEdgeBlock_MP(nn.Module):

    def __init__(self, hid, heads=4, edge_dim=None,  edge_hidden=128, dropout=0.0):
        super().__init__()
        assert hid % heads == 0

        kwargs = dict(in_channels=hid,
                      out_channels=hid // heads,
                      heads=heads,
                      concat=True,
                      dropout=dropout)
        if edge_dim is not None:
            kwargs['edge_dim'] = edge_dim
        self.gat = GATv2Conv(**kwargs)
        self.use_edge_in_gat = (edge_dim is not None)

        self.norm = nn.LayerNorm(hid)
        self.drop = nn.Dropout(dropout)

        self.edge_dim = edge_dim
        if edge_dim is not None:
            self.edge_mp = EdgeMP(d_node=hid, d_edge_in=edge_dim,
                                  d_edge_out=edge_dim, hidden=edge_hidden,
                                  dropout=dropout)
        else:
            self.edge_mp = None

    def forward(self, x, edge_index, e=None):
        # node message
        if self.use_edge_in_gat and (e is not None):
            h = self.gat(x, edge_index, edge_attr=e)
        else:
            h = self.gat(x, edge_index)
        x = self.norm(x + self.drop(h))
        # edge message
        if self.edge_mp is not None and e is not None:
            e = self.edge_mp(x, edge_index, e)
        return x, e

class DualTowerGNNEncoder(nn.Module):
    def __init__(self,
                 d_geo: int,
                 d_vis: int,
                 e_geo_raw: int = 9,
                 e_vis_raw: int = 2,
                 hid: int = 256,
                 num_layers: int = 3,
                 heads: int = 4,
                 edge_dim_geo: int = 32,
                 edge_dim_vis: int = 32,
                 dropout: float = 0.1,
                 edge_hidden=None,
                 ):
        super().__init__()

        # projection
        self.geo_proj = mlp(d_geo, hid, [64, 128, 256])
        self.vis_proj = mlp(d_vis, hid, [1024, 512, 256])

        self.edge_geo = EdgeProjector(e_geo_raw, edge_dim_geo, edge_hidden)
        self.edge_vis = EdgeProjector(e_vis_raw, edge_dim_vis, edge_hidden)

        self.geo_blocks = nn.ModuleList([
            NodeEdgeBlock_MP(hid, heads=heads, edge_dim=(edge_dim_geo if self.edge_geo else None),
                             edge_hidden=(edge_hidden or 128), dropout=dropout)
            for _ in range(num_layers)
        ])
        self.vis_blocks = nn.ModuleList([
            NodeEdgeBlock_MP(hid, heads=heads, edge_dim=(edge_dim_vis if self.edge_vis else None),
                             edge_hidden=(edge_hidden or 128), dropout=dropout)
            for _ in range(num_layers)
        ])
    
    @staticmethod
    def _check(batch):
        need = ['x_geo', 'x_vis', 'edge_index']
        for k in need:
            assert hasattr(batch, k), f"batch.{k} missing"
        if hasattr(batch, 'edge_attr_geo') and batch.edge_attr_geo is not None:
            assert batch.edge_attr_geo.size(0) == batch.edge_index.size(1), "edge_attr_geo rows must equal E"
        if hasattr(batch, 'edge_attr_vis') and batch.edge_attr_vis is not None:
            assert batch.edge_attr_vis.size(0) == batch.edge_index.size(1), "edge_attr_vis rows must equal E"
    
    def forward(self, batch):
        """
        input:
          - x_geo: [N, d_geo]
          - x_vis: [N, d_vis]
          - edge_index: [2, E]
          - edge_attr_geo: [E, e_geo_raw]
          - edge_attr_vis: [E, e_vis_raw]
        output:
          - xg: [N, hid], xs: [N, hid]
          - e_geo: [E, edge_dim_geo]
          - e_vis: [E, edge_dim_vis]
        """
        # print(batch.x_geo)
        self._check(batch)
        ei = batch.edge_index

        xg = self.geo_proj(batch.x_geo.float())
        xs = self.vis_proj(batch.x_vis.float())

        e_geo = self.edge_geo(batch.edge_attr_geo) if (self.edge_geo and batch.edge_attr_geo is not None) else None
        e_vis = self.edge_vis(batch.edge_attr_vis) if (self.edge_vis and batch.edge_attr_vis is not None) else None
        
        for bg, bs in zip(self.geo_blocks, self.vis_blocks):
            xg, e_geo = bg(xg, ei, e_geo)
            xs, e_vis = bs(xs, ei, e_vis)
        
        return xg, xs, e_geo, e_vis
