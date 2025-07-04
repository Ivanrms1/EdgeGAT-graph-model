# data_loader.py
import os
import glob
import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, radius_graph
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from feature_extractor import compute_pca_features

def adjacency_from_edge_index(edge_index, num_nodes):
    """
    Construye la matriz de adyacencia A (sparse COO) de un grafo no dirigido
    a partir de edge_index [2, E].
    """
    row, col = edge_index.cpu().numpy()
    data = np.ones(len(row), dtype=np.float32)
    A = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    # Hacemosla simétrica:
    A = A + A.T.multiply(A.T > A) - A.multiply(A.T > A)
    return A

def normalized_laplacian(A):
    """
    Devuelve la matriz de Laplaciano normalizado simétrico:
      L_sym = I - D^{-1/2} A D^{-1/2}
    """
    deg = np.array(A.sum(axis=1)).flatten()
    # Evitar división por cero:
    deg_inv_sqrt = 1.0 / np.sqrt(deg + 1e-12)
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    I = sp.eye(A.shape[0], format='csr')
    L = I - D_inv_sqrt @ A @ D_inv_sqrt
    return L

def laplacian_eigen_features(L_sym, k=3):
    """
    Calcula los primeros k+1 autovectores de L_sym y devuelve
    las últimas k (ignorando el primero constante).
    """
    # which='SM' → los k+1 autovalores más pequeños
    vals, vecs = eigsh(L_sym, k=k+1, which='SM', tol=1e-2)
    # vecs: (N, k+1). Quitamos la primera columna (modo trivial):
    return vecs[:, 1:]  # shape (N, k)

def get_edge_index(x, batch=None, graph_type="fixed", k=16, radius=0.15):
    """
    Igual que antes, construye knn_graph o radius_graph.
    """
    if not hasattr(get_edge_index, "has_printed"):
        msg = "[Graph Selection] Usando "
        msg += f"knn_graph k={k}" if graph_type=="fixed" else f"radius_graph r={radius}"
        print(msg)
        get_edge_index.has_printed = True

    if graph_type=="fixed":
        return knn_graph(x, k=k, batch=batch, loop=False)
    else:
        return radius_graph(x, r=radius, batch=batch, loop=False)

def load_txt_as_data(file_path,
                     k=16,
                     features="xyz_nc",
                     graph_type="fixed",
                     radius=0.15,
                     use_laplacian=False,
                     laplacian_k=3):
    """
    Lee un .txt (x,y,z,label) o solo (x,y,z) → Data(x, edge_index, y).
    Parámetros nuevos:
      - use_laplacian (bool): si True, extrae laplacian_k autovectores del Laplaciano.
      - laplacian_k (int): número de vectores propios a usar.
    """
    # 1. Cargo datos
    raw = np.loadtxt(file_path)
    if raw.ndim == 1: raw = raw.reshape(1, -1)
    if raw.shape[1] < 4:
        points = raw[:, :3]
        labels = np.zeros(points.shape[0], dtype=int)
    else:
        points = raw[:, :3]
        labels = raw[:, 3].astype(int)   -1 #Agrerar -1 dependiendo de escala del labeling (Ao dataset, pheno4d si esta ordenado)

    N = points.shape[0]

    # PCA-features opcional
    normals = curvature = linearity = planarity = scattering = omnivariance = anisotropy = eigenentropy = None
    if any(ch in features for ch in "nclpsoae"):
        normals, curvature, linearity, planarity, scattering, omnivariance, anisotropy, eigenentropy = \
            compute_pca_features(points, k=k)

    # 3. edge_index
    pts_t = torch.from_numpy(points).float()
    edge_index = get_edge_index(pts_t, None, graph_type, k, radius)

    # 4. Feature list
    feat_list = [points]
    if "n" in features and normals is not None: feat_list.append(normals)
    if "c" in features and curvature is not None: feat_list.append(curvature.reshape(-1,1))
    if "l" in features and linearity is not None: feat_list.append(linearity.reshape(-1,1))
    if "p" in features and planarity is not None: feat_list.append(planarity.reshape(-1,1))
    if "s" in features and scattering is not None: feat_list.append(scattering.reshape(-1,1))
    if "o" in features and omnivariance is not None: feat_list.append(omnivariance.reshape(-1,1))
    if "a" in features and anisotropy is not None: feat_list.append(anisotropy.reshape(-1,1))
    if "e" in features and eigenentropy is not None: feat_list.append(eigenentropy.reshape(-1,1))

    # 5. Laplacian features opcional
    if use_laplacian:
        A = adjacency_from_edge_index(edge_index, N)
        L = normalized_laplacian(A)
        lap_feats = laplacian_eigen_features(L, k=laplacian_k)  # (N, laplacian_k)
        feat_list.append(lap_feats)

    # 5 NORMALIZAR CADA BLOQUE DE FEATURES
    """Este paso muy importante, anteriormente no lo inclui y afecto enormemente dado que xyz estaban 
    influyendo mayoritariamente mientras que PCA y laplacian no"""
    norm_feat_list = []
    eps = 1e-12
    for feat in feat_list:
        # feat: (N, d_i) — p.ej. (N,3), (N,1) o (N, laplacian_k)
        mu  = feat.mean(axis=0, keepdims=True)      # (1, d_i)
        std = feat.std(axis=0, keepdims=True) + eps  # (1, d_i)
        norm_feat_list.append((feat - mu) / std)
    feat_list = norm_feat_list

    # 6. Concatenar y crear tensores
    x_np = np.hstack(feat_list)     # shape: (N, total_features)
    x_t  = torch.from_numpy(x_np).float()
    y_t  = torch.from_numpy(labels).long()

    return Data(x=x_t, edge_index=edge_index, y=y_t)



def load_folder_as_dataset(folder_path, **kwargs):
    files = glob.glob(os.path.join(folder_path, "*.txt"))
    return [load_txt_as_data(f, **kwargs) for f in files]

def get_datasets(train_folder, val_folder, test_folder, **kwargs):
    return (load_folder_as_dataset(train_folder, **kwargs),
            load_folder_as_dataset(val_folder, **kwargs),
            load_folder_as_dataset(test_folder, **kwargs))
