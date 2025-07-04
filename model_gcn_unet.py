
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
            in_channels (int): Dimensión de entrada (features iniciales).
            hidden_channels (int): Dimensión intermedia del GCNUNet.
            out_channels (int): Número de clases para la segmentación (node-level).
            depth (int): Número de niveles de pooling/unpooling.
            pool_ratios (float or list): fracción de nodos que retienes en cada pooling.
            use_bn (bool): si deseas usar BatchNorm en las capas.
            dropout (float): tasa de dropout a aplicar a las activaciones.
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
        """
        Retorna un embedding [num_nodes, out_channels] para cada nodo.
        Llamada ideal para clasificación/segmentación de nodos:
            out = model(batch.x, batch.edge_index, batch=batch.batch)
            loss = criterion(out, batch.y)
        """

        
        x = self.unet(x, edge_index, batch=batch)  # [N, out_channels]

        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x
