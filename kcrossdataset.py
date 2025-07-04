# kcrossdataset.py

import os
import glob
import numpy as np
from sklearn.model_selection import StratifiedKFold

def _extract_label_from_txt(path):
    """
    Lee la primera línea de path y extrae la etiqueta (columna 4) si existe,
    devolviendo 0 si no hay etiqueta.
    """
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] < 4:
        return 0
    else:
        # restamos 1 para pasar etiquetas [1,2,3] → [0,1,2]
        return int(raw[0, 3]) - 1

def get_cv_splits(nubes_folder, test_folder,
                  n_splits=5, shuffle=True, random_state=42):
    """
    Devuelve:
      - folds: lista de tuplas (train_paths, val_paths), una por cada split
      - test_paths: lista de todos los archivos .txt en test_folder
    """
    # 1) Lista y ordena los archivos de entrenamiento/validación
    all_paths = sorted(glob.glob(os.path.join(nubes_folder, "*.txt")))
    # 2) Extrae etiquetas para StratifiedKFold
    labels = [_extract_label_from_txt(p) for p in all_paths]

    skf = StratifiedKFold(n_splits=n_splits,
                          shuffle=shuffle,
                          random_state=random_state)

    folds = []
    for train_idx, val_idx in skf.split(all_paths, labels):
        train_paths = [all_paths[i] for i in train_idx]
        val_paths   = [all_paths[i] for i in val_idx]
        folds.append((train_paths, val_paths))

    # 3) Lista de test (sin hacer split)
    test_paths = sorted(glob.glob(os.path.join(test_folder, "*.txt")))

    return folds, test_paths
