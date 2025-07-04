#!/usr/bin/env python
import os
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch_geometric.loader import DataLoader
from hybrid_pt_model import HybridEdgeGATPointTransformer
from kcrossdataset import get_cv_splits
from data_loader import load_txt_as_data
from model import build_model          # GAT original
from model3 import GCNUNet             # Graph U-Net (simple)
from model_gcn_unet import GCNUNet2    # Graph U-Net modificado
from model5 import SimpleGCN           # GCN simple
from GATedgeconv import build_edgegat_model
from Gattnetconv import GATTNetConvHybrid
from Ptencoding import PointTransformerV3
#from GatTransformer import HybridGATTransformer
from metricas import (
    calcular_metricas,
    plot_metrics_over_epochs,
    save_metrics_to_csv
)

def compute_in_channels(feats: str):
    """
    Cuenta cuántos canales resultan de la combinación de features.
    """
    c = 3  # xyz
    if "n" in feats: c += 3
    if "c" in feats: c += 1
    if "l" in feats: c += 1
    if "p" in feats: c += 1
    if "s" in feats: c += 1
    if "o" in feats: c += 1
    if "a" in feats: c += 1
    if "e" in feats: c += 1
    return c

def train_kfold(
    ARCHITECTURE="GATNET",
    FEATURES="xyz_nclpsoae",
    n_splits=5,
    nubes_folder="kcrossdataset/phenonubes",
    test_folder="kcrossdataset/phenotest"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Architecture: {ARCHITECTURE}")
    print(f"Features: {FEATURES}")
    print(f"Using Dataset from: {nubes_folder}")

    # Opciones de Laplaciano
    use_laplacian = False
    laplacian_k   = 3

    # Computar canales de entrada
    in_channels = compute_in_channels(FEATURES)
    if use_laplacian:
        in_channels += laplacian_k
    print(f"in_channels = {in_channels} (laplacian={use_laplacian})")

    # Hiperparámetros
    hidden_channels = 32
    out_channels    = 3
    heads           = 2
    lr              = 5e-4
    epochs          = 100
    k               = 16
    batch_size      = 4
    graph_type      = "fixed"
    radius          = 0.15

    # Obtener folds + test fijo
    folds, test_paths = get_cv_splits(
        nubes_folder, test_folder,
        n_splits=n_splits, shuffle=True, random_state=42
    )
    print(f"Found {len(folds)} folds, {len(test_paths)} test files.")

    base_cv_dir = "results_cv"
    os.makedirs(base_cv_dir, exist_ok=True)

    metrics_all_folds = []  # para summary
    fold_mious = []
    fold_times = []

    # --- Loop sobre folds ---
    for idx, (train_paths, val_paths) in enumerate(folds, start=1):
        print(f"\n=== FOLD {idx}/{n_splits} ===")

        # Preparar DataLoaders
        loader_kwargs = {
            "k": k,
            "features": FEATURES,
            "graph_type": graph_type,
            "radius": radius,
            "use_laplacian": use_laplacian,
            "laplacian_k": laplacian_k
        }
        train_data = [load_txt_as_data(p, **loader_kwargs) for p in train_paths]
        val_data   = [load_txt_as_data(p, **loader_kwargs) for p in val_paths]
        test_data  = [load_txt_as_data(p, **loader_kwargs) for p in test_paths]

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False)

        # Instanciar modelo según ARCHITECTURE
        arch = ARCHITECTURE.upper()
        if arch == "DEFAULT":
            print("Using GAT original")
            model = build_model(in_channels, hidden_channels, out_channels, heads)
        elif arch == "GCN":
            print("Using SimpleGCN")
            model = SimpleGCN(in_channels, out_channels, dropout=0.2)
        elif arch == "UNET":
            print("Using GCNUNet (simple)")
            model = GCNUNet(in_channels, hidden_channels, out_channels, depth=3, pool_ratios=0.5)
        elif arch == "UNET2":
            print("Using GCNUNet2 (modified)")
            model = GCNUNet2(in_channels, hidden_channels, out_channels, depth=3, pool_ratios=0.5)
        elif arch == "PT":  # o cualquier clave
            print("Using Hybrid EdgeConv + GAT + PointTransformer")
            model = HybridEdgeGATPointTransformer(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                dropout=0.2
            )

        elif arch == "PTV3":
            print("Using PointTransformerV3")
            # __init__(in_channels, out_channels, hidden_channels, k)
            model = PointTransformerV3(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                k=k,
            )

        elif arch == "EDGEGAT":
            print("Using EDGEGAT")
            model = build_edgegat_model(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                heads=heads,
                k=k,
                dynamic_graph=False
            )
        elif arch == "GATNET":
            print("Using GATNetConvHybrid")
            model = GATTNetConvHybrid(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                heads=heads,
                k=k,
                dynamic_graph=False,
                dropout=0.2
            )
        elif arch == "HYBRID":
            print("Using hybrid GAT + Transformer block")
            model = HybridGATTransformer(
                in_dim=in_channels,
                hid_dim=hidden_channels,
                out_dim=out_channels,
                heads=heads,
                dropout=0.1
            )
        else:
            raise ValueError(f"Unsupported architecture: {ARCHITECTURE}")

        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        all_metrics = []
        best_miou   = 0.0
        best_epoch  = 0

        # --- Entrenamiento por época ---
        t0 = time.time()
        for epoch in range(1, epochs+1):
            # Train
            model.train()
            train_losses, train_preds, train_labels = [], [], []
            for batch in train_loader:
                batch = batch.to(device)
                if arch == "PTV3":
                    x, pos = batch.x, batch.x[:, :3]
                    out = model(x, pos)
                else:
                    # tus casos anteriores
                    try:
                        out = model(batch.x, batch.edge_index, batch.batch)
                    except TypeError:
                        out = model(batch.x, batch.edge_index)
                loss = criterion(out, batch.y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                preds = out.argmax(dim=1).cpu().numpy()
                train_preds.append(preds)
                train_labels.append(batch.y.cpu().numpy())

            train_preds  = np.concatenate(train_preds)
            train_labels = np.concatenate(train_labels)
            train_loss   = np.mean(train_losses)
            train_m      = calcular_metricas(
                y_true=train_labels,
                y_pred=train_preds,
                loss_value=train_loss,
                average='macro'
            )

            # Validation
            model.eval()
            val_losses, val_preds, val_labels = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    try:
                        out = model(batch.x, batch.edge_index, batch.batch)
                    except TypeError:
                        out = model(batch.x, batch.edge_index)
                    loss = criterion(out, batch.y)
                    val_losses.append(loss.item())
                    preds = out.argmax(dim=1).cpu().numpy()
                    val_preds.append(preds)
                    val_labels.append(batch.y.cpu().numpy())

            val_preds  = np.concatenate(val_preds)
            val_labels = np.concatenate(val_labels)
            val_loss   = np.mean(val_losses)
            val_m      = calcular_metricas(
                y_true=val_labels,
                y_pred=val_preds,
                average='macro'
            )

            # Guardar métricas
            all_metrics.append({
                'epoch': epoch,
                'train_loss':   train_m['loss'],
                'val_loss':     val_loss,
                'val_accuracy': val_m['accuracy'],
                'val_precision':val_m['precision'],
                'val_recall':   val_m['recall'],
                'val_mIoU':     val_m['mIoU']
            })

            print(f"Epoch {epoch:02d}/{epochs} | "
                  f"TrainLoss={train_m['loss']:.4f} | "
                  f"ValmIoU={val_m['mIoU']:.4f}")

            # Mejor modelo por fold
            if val_m['mIoU'] > best_miou:
                best_miou  = val_m['mIoU']
                best_epoch = epoch
        tf = time.time()
        elapsed = tf - t0
        fold_times.append(elapsed)
        print(f"Fold {idx} training time: {elapsed:.1f}s — best mIoU: {best_miou:.4f}")
        fold_mious.append(best_miou)

        # Guardar métricas de este fold
        fold_dir = os.path.join(base_cv_dir, f"fold_{idx}")
        os.makedirs(fold_dir, exist_ok=True)
        plot_metrics_over_epochs(all_metrics, save_path=os.path.join(fold_dir, "curvas.png"))
        save_metrics_to_csv(all_metrics, os.path.join(fold_dir, "metrics.csv"))
        metrics_all_folds.append(all_metrics)

    # --- Resumen cross-validation por época ---------------------------------------
    metric_keys = ['train_loss','val_loss','val_accuracy','val_precision','val_recall','val_mIoU']
    summary = {ep: {mk: [] for mk in metric_keys} for ep in range(1, epochs+1)}

    for fold_metrics in metrics_all_folds:
        for ep_dict in fold_metrics:
            ep = ep_dict['epoch']
            for mk in metric_keys:
                summary[ep][mk].append(ep_dict[mk])

    # Guardar CSV de resumen
    summary_dir = os.path.join(base_cv_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    summary_csv = os.path.join(summary_dir, "cv_epoch_summary.csv")

    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['epoch'] + [f"{mk}_mean" for mk in metric_keys] + [f"{mk}_std" for mk in metric_keys]
        writer.writerow(header)
        for ep in range(1, epochs+1):
            row = [ep]
            for mk in metric_keys:
                vals = np.array(summary[ep][mk])
                row += [f"{vals.mean():.4f}", f"{vals.std():.4f}"]
            writer.writerow(row)

    print(f"Saved per-epoch summary CSV to {summary_csv}")

    # (Opcional) Graficar cada métrica con barras de error
    for mk in metric_keys:
        means = [np.mean(summary[ep][mk]) for ep in range(1, epochs+1)]
        stds  = [np.std (summary[ep][mk]) for ep in range(1, epochs+1)]
        plt.figure(figsize=(6,4))
        plt.errorbar(range(1, epochs+1), means, yerr=stds, fmt='-o', capsize=3)
        plt.title(f"{mk} over epochs (mean ± std)")
        plt.xlabel("Epoch")
        plt.ylabel(mk)
        plt.tight_layout()
        plot_path = os.path.join(summary_dir, f"cv_{mk}_curve.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot {plot_path}")

    # Reporte final
    mean_time = np.mean(fold_times)
    std_time = np.std(fold_times)
    mean_miou = np.mean(fold_mious)
    std_miou  = np.std(fold_mious)
    print(f"\n=== CROSS-VALIDATION SUMMARY ({n_splits} folds) ===")
    print(f"\nCross-validation mIoU: {mean_miou:.4f} ± {std_miou:.4f}")
    print(f" Training time: {mean_time:.1f}s ± {std_time:.1f}s")

if __name__ == "__main__":
    # Uso: python train_k.py <ARCH> <FEATURES> [<n_splits>]
    arch   = sys.argv[1] if len(sys.argv) > 1 else "GATNET"
    feats  = sys.argv[2] if len(sys.argv) > 2 else "xyz_nclpsoae"
    splits = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    train_kfold(arch, feats, n_splits=splits)
