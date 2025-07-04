import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

def compute_pca_features(points, k=16):
    """
    Calcula, para cada punto, las siguientes características:
      - Normales (3 valores)
      - Curvatura (1 valor)
      - Linealidad (1 valor)
      - Planaridad (1 valor)
      - Scattering (1 valor)
      - Omnivariance (1 valor)
      - Anisotropía (1 valor)
      - Eigenentropy (1 valor)
    
    Retorna:
      - normals: array de forma (N, 3)
      - curvature: array de forma (N,)
      - linearity: array de forma (N,)
      - planarity: array de forma (N,)
      - scattering: array de forma (N,)
      - omnivariance: array de forma (N,)
      - anisotropy: array de forma (N,)
      - eigenentropy: array de forma (N,)
    """
    N = points.shape[0]
    normals = np.zeros((N, 3))
    curvature = np.zeros(N)
    linearity = np.zeros(N)
    planarity = np.zeros(N)
    scattering = np.zeros(N)
    omnivariance = np.zeros(N)
    anisotropy = np.zeros(N)
    eigenentropy = np.zeros(N)
    
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    for i in range(N):
        neighbor_idx = indices[i]
        neighbor_points = points[neighbor_idx]  # (k, 3)
        pca = PCA(n_components=3)
        pca.fit(neighbor_points)
        eigenvalues = pca.singular_values_
        eigenvectors = pca.components_
        
        # Se asume que los eigenvalores están ordenados de mayor a menor siendo lam3 el menor
        lam1, lam2, lam3 = eigenvalues
        
        # Normal: vector asociado al menor eigenvalor en este caso lam3, dado la lista es el -1
        normals[i] = eigenvectors[-1]
        
        # Curvatura
        curvature[i] = lam3 / (lam1 + lam2 + lam3 + 1e-12)
        
        # Descriptores derivados:
        linearity[i] = (lam1 - lam2) / (lam1 + 1e-12)
        planarity[i] = (lam2 - lam3) / (lam1 + 1e-12)
        scattering[i] = lam3 / (lam1 + 1e-12)
        omnivariance[i] = (lam1 * lam2 * lam3) ** (1/3)
        anisotropy[i] = (lam1 - lam3) / (lam1 + 1e-12)
        eigenentropy[i] = - (lam1 * np.log(lam1 + 1e-12) + lam2 * np.log(lam2 + 1e-12) + lam3 * np.log(lam3 + 1e-12))
    
    return normals, curvature, linearity, planarity, scattering, omnivariance, anisotropy, eigenentropy
