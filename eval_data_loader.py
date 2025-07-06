import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, radius_graph
from feature_extractor import compute_pca_features 
from torch_geometric.nn.pool import fps  # Importamos FPS para el downsampling

def get_edge_index(x, batch=None, graph_type="fixed", k=16, radius=0.15):
    """
    Genera el edge_index usando knn_graph o radius_graph.
    """
    if graph_type.lower() == "fixed":
        return knn_graph(x, k=k, batch=batch, loop=False)
    elif graph_type.lower() == "dynamic":
        return radius_graph(x, r=radius, batch=batch, loop=False)
    else:
        raise ValueError("graph_type must be 'fixed' or 'dynamic'")

def load_txt_as_data_eval(file_path, k=16, features="ALL", graph_type="fixed", radius=0.15, max_points=None):
    """
    Data loader de evaluación.
    
    Si max_points se especifica y el número de puntos es mayor, se realiza un downsampling
    homogéneo mediante farthest point sampling.
    
    Retorna un objeto Data de PyG.
    """
    # 1. Cargar datos raw
    raw = np.loadtxt(file_path)
    if raw.shape[1] < 4:
        points = raw[:, :3]
        labels = np.zeros((raw.shape[0],), dtype=int)
    else:
        points = raw[:, :3]
        labels = raw[:, 3].astype(int)
        labels = labels - 1  # Ajuste de etiquetas si es necesario

    # 2. Determinar qué features extraer.
    if features.upper() == "ALL":
        feature_flags = "nclpsoae"
    else:
        feature_flags = features.lower()

    # 3. Calcular características adicionales si es necesario.
    normals = curvature = linearity = planarity = scattering = omnivariance = anisotropy = eigenentropy = None
    if any(ch in feature_flags for ch in "nclpsoae"):
        normals, curvature, linearity, planarity, scattering, omnivariance, anisotropy, eigenentropy = compute_pca_features(points, k=k)
    
    # 4. Armar la lista de características (siempre se incluyen las coordenadas)
    feat_list = [points]
    if "n" in feature_flags and normals is not None:
        feat_list.append(normals)
    if "c" in feature_flags and curvature is not None:
        feat_list.append(curvature.reshape(-1, 1))
    if "l" in feature_flags and linearity is not None:
        feat_list.append(linearity.reshape(-1, 1))
    if "p" in feature_flags and planarity is not None:
        feat_list.append(planarity.reshape(-1, 1))
    if "s" in feature_flags and scattering is not None:
        feat_list.append(scattering.reshape(-1, 1))
    if "o" in feature_flags and omnivariance is not None:
        feat_list.append(omnivariance.reshape(-1, 1))
    if "a" in feature_flags and anisotropy is not None:
        feat_list.append(anisotropy.reshape(-1, 1))
    if "e" in feature_flags and eigenentropy is not None:
        feat_list.append(eigenentropy.reshape(-1, 1))
    
    # 5. Concatenar todas las features horizontalmente.
    x_np = np.hstack(feat_list)
    x_torch = torch.from_numpy(x_np).float()

    # 6. Si se especifica max_points y el número de puntos es mayor, aplicar FPS para downsampling.
    if max_points is not None and x_torch.size(0) > max_points:
        ratio = max_points / float(x_torch.size(0))
        indices = fps(x_torch, batch=None, ratio=ratio)
        x_torch = x_torch[indices]
        pts_torch = torch.from_numpy(points).float()[indices]
        y_torch = torch.from_numpy(labels).long()[indices]
    else:
        pts_torch = torch.from_numpy(points).float()
        y_torch = torch.from_numpy(labels).long()

    # 7. Construir la conectividad del grafo usando las coordenadas
    edge_index = get_edge_index(pts_torch, batch=None, graph_type=graph_type, k=k, radius=radius)

    # 8. Crear y retornar el objeto data
    data = Data(x=x_torch, edge_index=edge_index, y=y_torch)
    return data
