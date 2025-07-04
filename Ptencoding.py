import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph, MessagePassing
from torch_geometric.utils import softmax

def topk_per_batch(dst: torch.Tensor, scores: torch.Tensor, k: int) -> torch.Tensor:
    """
    Given:
      dst    – a LongTensor of shape [E], the target node index for each edge
      scores – a FloatTensor of shape [E], the score of each edge
      k      – how many top edges to keep per target node
    Returns:
      mask – a BoolTensor of shape [E], True for the top-k edges for each dst node
    """
    device = dst.device
    E = dst.size(0)
    mask = torch.zeros(E, dtype=torch.bool, device=device)

    # get all unique destination nodes
    unique_dst = torch.unique(dst)
    for node in unique_dst:
        # indices of edges that point to this node
        idxs = (dst == node).nonzero(as_tuple=True)[0]
        if idxs.numel() <= k:
            # keep them all
            mask[idxs] = True
        else:
            # pick top-k among these
            node_scores = scores[idxs]
            topk_vals, topk_idxs = torch.topk(node_scores, k, largest=True)
            topk_edge_idxs = idxs[topk_idxs]
            mask[topk_edge_idxs] = True

    return mask

class SampleNet(nn.Module):
    """
    Learns a score for each candidate neighbor j of i, then takes top-k.
    """
    def __init__(self, in_dim, hidden_dim, k):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Linear(2*in_dim + 3, hidden_dim),  # x_i||x_j||||pos_i - pos_j||
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, pos, full_edge_index):
        # full_edge_index: maybe large radius graph
        src, dst = full_edge_index
        # compute scores for every edge
        rel_pos = pos[src] - pos[dst]
        feats = torch.cat([x[src], x[dst], rel_pos], dim=-1)
        scores = self.net(feats).squeeze(-1)      # [E]
        
        # for each dst (center) pick top-k src
        # we'll assume batch-wise single graph for clarity
        # group scores by dst and select top-k indices
        # (in practice use torch.topk per group or a radius+topk trick)
        topk_mask = topk_per_batch(dst, scores, self.k)  # boolean mask over edges
        return full_edge_index[:, topk_mask]

class GeoPosNet(nn.Module):
    """
    V3-style positional encoding network that
    learns higher-order geometry.
    Extraido del artículo de V3 pointtransformer el ordenamiento posicional
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, rel_pos):
        # Here you could also append norm(rel_pos)² or angles
        return self.net(rel_pos)
    
class PointTransformerV3(MessagePassing):
    def __init__(self, in_channels, out_channels, hidden_channels, k):
        super().__init__(aggr='add')  # o 'sum' según quieras
        self.k = k
        self.sample_net = SampleNet(in_channels, hidden_channels, k)
        self.geo_pos    = GeoPosNet(hidden_channels)
        self.lin_q      = nn.Linear(in_channels, hidden_channels)
        self.lin_k      = nn.Linear(in_channels, hidden_channels)
        self.lin_v      = nn.Linear(in_channels, hidden_channels)
        self.out_proj   = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index=None, batch=None):
        # extraemos posiciones de los 3 primeros channels
        pos = x[:, :3]
        # construimos grafo denso y luego muestreamos top-k
        full_edge_index = knn_graph(pos, k=self.k * 3, batch=batch, loop=False)
        edge_index = self.sample_net(x, pos, full_edge_index)
        return self.propagate(edge_index, x=x, pos=pos) #se propaga 

    def message(self, x_i, x_j, pos_i, pos_j, index):
        q = self.lin_q(x_i)     #vectores en el transformer
        k = self.lin_k(x_j)
        v = self.lin_v(x_j)
        # bias posicional
        pe = self.geo_pos(pos_i - pos_j)
        # score de atención
        alpha = (q * (k + pe)).sum(dim=-1, keepdim=True)
        # softmax por destino
        alpha = softmax(alpha, index)
        return alpha * v     #nota esto estaba como tal cual el articulo

    def update(self, aggr_out):
        return self.out_proj(aggr_out)