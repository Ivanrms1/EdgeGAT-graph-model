
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphUNet

class GCNUNet2(nn.Module):
    """
    Graph U-Net con GCN interno, orientado a clasificación a nivel de nodo.
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 depth=3, pool_ratios=0.5, 
                 use_bn=True, dropout=0.2):
        """
        Args:
            in_channels (int): initial features.
            hidden_channels (int): middle layers
            out_channels (int): number of classes for classification (node-level).
            depth (int): number of pooling/unpooling.
            pool_ratios (float or list):
            use_bn (bool): for batchnorm between layers
            dropout (float): adjust dropout value
        """
        super().__init__()
        
        self.unet = GraphUNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            depth=depth,
            pool_ratios=pool_ratios,
            sum_res=True  
        )
        
        self.use_bn = use_bn
        self.dropout = dropout
        
        if use_bn:
            self.bn = nn.BatchNorm1d(hidden_channels)
        else:
            self.bn = None

    def forward(self, x, edge_index, batch=None):
        x = self.unet(x, edge_index, batch=batch)  # [N, out_channels]

        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x
