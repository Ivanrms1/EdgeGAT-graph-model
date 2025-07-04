import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphUNet

class GCNUNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, depth=3, pool_ratios=0.5):
        """
        Graph U-Net con capas basadas en GCN. 
        'depth' indica cuántos niveles de poolin
        'pool_ratios' puede ser un float o lista
        """
        super().__init__()
        self.unet = GraphUNet(in_channels, hidden_channels, out_channels,
                              depth=depth, pool_ratios=pool_ratios,
                              sum_res=True)  

    def forward(self, x, edge_index):
        """
        Retorna un embedding [num_nodes, out_channels] para cada nodo
        """
        x = self.unet(x, edge_index)
        # x = F.log_softmax(x, dim=-1)  
        return x
