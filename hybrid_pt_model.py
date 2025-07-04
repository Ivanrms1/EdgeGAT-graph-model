## hybrid_pt_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, GATConv, PointTransformerConv
from torch_geometric.nn import knn_graph

class HybridEdgeGATPointTransformer(nn.Module):
    """
    Híbrido: EdgeConv → GAT → PointTransformerConv → Classifier
    """
    def __init__(self,
                 in_channels,       # dim. de entrada 
                 hidden_channels,   # canales ocultos
                 out_channels,      # num de clases
                 k=16,              # vecinos para knn
                 dropout=0.2):
        super().__init__()
        self.k = k

        # 1) Dos capas EdgeConv
        self.edgeconv1 = EdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * in_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels),
            ),
            aggr='sum'
        )
        self.edgeconv2 = EdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels),
            ),
            aggr='mean'
        )

        # 2) Dos capas GAT
        
        self.gat1 = GATConv(2 * hidden_channels, hidden_channels, heads=4, concat=True)
        self.gat2 = GATConv(hidden_channels * 4, hidden_channels, heads=4, concat=False)

        # 3) MLPs for PointTransformerConv
        self.pos_nn = nn.Sequential(
            nn.Linear(3, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.attn_nn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
        # 4) Bloque PointTransformerConv
        self.pt_conv = PointTransformerConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            pos_nn=self.pos_nn,
            attn_nn=self.attn_nn
        )

        # 5) Clasificador final
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(self, x, edge_index, batch):


        # 1) EdgeConv layers
        x1 = F.leaky_relu(self.edgeconv1(x, edge_index))
        x2 = F.leaky_relu(self.edgeconv2(x1, edge_index))
        x_cat = torch.cat([x1, x2], dim=-1)                     # [N, 2*hidden]

        # 2) GAT layers
        x_g = F.elu(self.gat1(x_cat, edge_index))               # [N, hidden*heads]
        x_g = F.elu(self.gat2(x_g, edge_index))                 # [N, hidden]

        # 3) PointTransformerConv
        #   
        pos = x[:, :3]                                          # [N,3]
        x_pt = F.elu(self.pt_conv(x_g, pos, edge_index))        # [N, hidden]

        # 4) Classifier
        out = self.classifier(x_pt)                             # [N, out_channels]
        return out
