import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
import sys
from torch_geometric.loader import DataLoader
# Archivos internos del proyecto
from data_loader import get_datasets   
from model import build_model          # GAT original
from model3 import GCNUNet             # Graph U-Net (simple)
from model_gcn_unet import GCNUNet2    # Graph U-Net modificado
from model5 import SimpleGCN
from GATedgeconv import build_edgegat_model
from Gattnetconv import GATTNetConvHybrid

from metricas import (
    calcular_metricas,
    plot_metrics_over_epochs,
    plot_confusion_matrix_custom,
    save_metrics_to_csv
)

def compute_in_channels(feats: str):
    """
    Cuenta cuántos canales resultan de la combinación de features.
    Ejemplos:
      "xyz"          -> 3
      "xyz_n"        -> 3 + 3 = 6
      "xyz_nc"       -> 3 + 3 + 1 = 7
      "xyz_nclpsoae" -> 3 + 3 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = 13
    """
    c = 3  # siempre se incluyen xyz
    if "n" in feats:
        c += 3
    if "c" in feats:
        c += 1
    if "l" in feats:
        c += 1
    if "p" in feats:
        c += 1
    if "s" in feats:
        c += 1
    if "o" in feats:
        c += 1
    if "a" in feats:
        c += 1
    if "e" in feats:
        c += 1
    return c

def train(ARCHITECTURE="DEFAULT", FEATURES="xyz_nc"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    print(f"Arquitectura seleccionada: {ARCHITECTURE}")
    print(f"Features seleccionados: {FEATURES}")

    use_laplacian = False      
    laplacian_k   = 3         # número de vectores propios a extraer

    in_channels = compute_in_channels(FEATURES)
    if use_laplacian:
        in_channels += laplacian_k
    print(f"in_channels = {in_channels}  (incluyendo{' ' if use_laplacian else ' no '}Laplacian)")

    hidden_channels = 32
    out_channels = 3
    heads = 2
    lr = 0.001
    epochs = 100
    k = 16  #ignore if graphtype is "dynamic"
    batch_size = 4
    graphtype = "fixed"  #use fixed or dynamic, if dynamic adjust radius
    radius_k=0.16



    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"resultados_{ARCHITECTURE}_{FEATURES}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # Cargar datasets usando la función get_datasets 
    train_data_list, val_data_list, test_data_list = get_datasets(
        train_folder="Etiquetas_txt/nubes_train",
        val_folder="Etiquetas_txt/nubes_val",
        test_folder="Etiquetas_txt/nubes_test",
        k=k,
        features=FEATURES,
        graph_type=graphtype,
        radius=radius_k,
        use_laplacian=use_laplacian,
        laplacian_k=laplacian_k
    )
    
    print(f"Train nubes: {len(train_data_list)}")
    print(f"Val nubes:   {len(val_data_list)}")
    print(f"Test nubes:  {len(test_data_list)}")

    train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data_list, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_data_list, batch_size=batch_size, shuffle=False)
    
    # Seleccionar y crear el modelo
    if ARCHITECTURE.upper() == "DEFAULT":
        print("Usando modelo DEFAULT (build_model)")
        model = build_model(in_channels, hidden_channels, out_channels, heads)
    elif ARCHITECTURE.upper() == "EDGEGAT":
        print("Usando EDGEGAT")
        dynamic_graph = False
        model = build_edgegat_model(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            heads=heads,
            k=k,
            dynamic_graph=dynamic_graph
        )
    elif ARCHITECTURE.upper() == "GCN":
        print("Usando GCN simple")
        model = SimpleGCN(in_channels, out_channels, dropout=0.2)
    elif ARCHITECTURE.upper() == "UNET2":
        print("Versión modificada de UNET")
        model = GCNUNet2(in_channels, hidden_channels, out_channels, depth=3, pool_ratios=0.5)
    elif ARCHITECTURE.upper() == "UNET":
        print("Usando GCNUNet (Graph U-Net)")
        model = GCNUNet(in_channels, hidden_channels, out_channels, depth=3, pool_ratios=0.5)

    elif ARCHITECTURE.upper() == "GATNET":
        print("Usando el modelo de GATEDGECONV con Tnet")
        model = GATTNetConvHybrid(in_channels=in_channels,hidden_channels=hidden_channels,
        out_channels=out_channels,
        heads=heads,
        k=k,
        dynamic_graph= False,
        dropout=0.2
    )
    
    else:
        raise ValueError(f"Arquitectura desconocida: {ARCHITECTURE}")

    model.to(device)
    no_batch_models = {"DEFAULT", "GCN", "UNET", "UNET2", "GATTRANSFORMER", "GATRELTRANSFORMER", "GATRELMH"}
    needs_batch = ARCHITECTURE.upper() not in no_batch_models
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    all_metrics = []
    best_val_metric = 0.0
    best_epoch = 0
    
    print("Iniciando entrenamiento...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_preds_total = []
        train_labels_total = []
        
        for batch in train_loader:
            batch = batch.to(device)
            #out = model(batch.x, batch.edge_index) #para caso de GATRELMH
            #out = model(batch.x, batch.edge_index, batch.batch)
            if needs_batch:
                out = model(batch.x, batch.edge_index, batch.batch)
            else:
                out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            train_losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            preds = out.argmax(dim=1)
            train_preds_total.append(preds.cpu().numpy())
            train_labels_total.append(batch.y.cpu().numpy())
        
        train_preds_total = np.concatenate(train_preds_total)
        train_labels_total = np.concatenate(train_labels_total)
        train_loss_mean = np.mean(train_losses)
        train_m = calcular_metricas(
            y_true=train_labels_total,
            y_pred=train_preds_total,
            loss_value=train_loss_mean,
            average='macro'
        )
        
        model.eval()
        val_preds_total = []
        val_labels_total = []
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                if needs_batch:                 #dado que unos piden 3 y otros dos argumentos
                    out_val = model(batch.x, batch.edge_index, batch.batch)
                else:
                    out_val = model(batch.x, batch.edge_index)
                loss_val = criterion(out_val, batch.y)
                val_losses.append(loss_val.item())
                preds_val = out_val.argmax(dim=1)
                val_preds_total.append(preds_val.cpu().numpy())
                val_labels_total.append(batch.y.cpu().numpy())
        
        val_preds_total = np.concatenate(val_preds_total)
        val_labels_total = np.concatenate(val_labels_total)
        val_loss_mean = np.mean(val_losses)
        val_m = calcular_metricas(
            y_true=val_labels_total,
            y_pred=val_preds_total,
            average='macro'
        )
        
        epoch_dict = {
            'epoch': epoch + 1,
            'train_loss': train_m['loss'],
            'val_loss': val_loss_mean,
            'val_accuracy': val_m['accuracy'],
            'val_precision': val_m['precision'],
            'val_recall': val_m['recall'],
            'val_mIoU': val_m['mIoU']
        }
        all_metrics.append(epoch_dict)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"TrainLoss={epoch_dict['train_loss']:.4f} | "
              f"ValLoss={val_loss_mean:.4f} | "
              f"ValAcc={val_m['accuracy']:.4f} | "
              f"ValPrec={val_m['precision']:.4f} | "
              f"ValRecall={val_m['recall']:.4f} | "
              f"ValmIoU={val_m['mIoU']:.4f}")
        
        current_val_metric = val_m['mIoU']
        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            best_epoch = epoch + 1
            best_path = os.path.join(save_dir, "model_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f" [*] Nuevo mejor modelo guardado en epoch {best_epoch} "
                  f"con val_mIoU={best_val_metric:.4f}")
    
    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nEntrenamiento finalizado en {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    model.eval()
    test_preds_total = []
    test_labels_total = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            if needs_batch:
                out_test = model(batch.x, batch.edge_index, batch.batch)
            else:
                out_test = model(batch.x, batch.edge_index)
            preds_test = out_test.argmax(dim=1)
            test_preds_total.append(preds_test.cpu().numpy())
            test_labels_total.append(batch.y.cpu().numpy())
    
    test_preds_total = np.concatenate(test_preds_total)
    test_labels_total = np.concatenate(test_labels_total)
    test_m = calcular_metricas(
        y_true=test_labels_total,
        y_pred=test_preds_total,
        average='macro'
    )
    print("\nMétricas en Test:")
    print(f" Accuracy={test_m['accuracy']:.4f}, "
          f"Precision={test_m['precision']:.4f}, "
          f"Recall={test_m['recall']:.4f}, "
          f"mIoU={test_m['mIoU']:.4f}")
    
    plot_path = os.path.join(save_dir, "entrenamiento_curvas.png")
    plot_metrics_over_epochs(all_metrics, save_path=plot_path)
    
    csv_path = os.path.join(save_dir, "entrenamiento_metricas.csv")
    save_metrics_to_csv(all_metrics, csv_path=csv_path)
    
    cm_path = os.path.join(save_dir, "confusion_matrix_test.png")
    classes = ["Clase 0", "Clase 1", "Clase 2"]
    plot_confusion_matrix_custom(
        y_true=test_labels_total,
        y_pred=test_preds_total,
        classes=classes,
        normalize='true',
        save_path=cm_path
    )
    
    print(f"\nMejor modelo guardado en epoch {best_epoch} con val_metric={best_val_metric:.4f}")
    print(f"Archivos guardados en: {save_dir}")

if __name__ == "__main__":
    
    #   python train.py EDGEGAT xyz_n --> Usa coordenadas, normales y curvatura
    #   python train.py EDGEGAT all     --> Usa todas las features
    if len(sys.argv) > 1:
        arch = sys.argv[1]
    else:
        arch = "DEFAULT"

    if len(sys.argv) > 2:
        feats = sys.argv[2]
        if feats.lower() == "all":
            feats = "xyz_nclpsoae"
    else:
        feats = "xyz_nc"

    train(ARCHITECTURE=arch, FEATURES=feats)
