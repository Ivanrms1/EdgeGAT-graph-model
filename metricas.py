# metricas.py

import matplotlib.pyplot as plt
import numpy as np
import csv

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    jaccard_score,  # Para IoU
    confusion_matrix,
    ConfusionMatrixDisplay
)

def calcular_metricas(y_true, y_pred, loss_value=None, average='macro'):
    """
    Calcula varias métricas (Accuracy, Precision, Recall, mean IoU) para
    un problema de clasificación multi-clase, y opcionalmente incluye la pérdida.
    
    Args:
        y_true (array-like): Etiquetas reales, tamaño [N].
        y_pred (array-like): Etiquetas predichas, tamaño [N].
        loss_value (float, opcional): valor de la pérdida (ej. cross-entropy).
        average (str): 'macro', 'weighted', etc. para promediar las métricas multi-clase.
    
    Returns:
        dict: {
            'accuracy': float,
            'precision': float,
            'recall': float,
            'mIoU': float,
            'loss': float (opcional)
        }
    """
    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    
    # Precision macro-average por defecto
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    
    # Recall macro-average por defecto
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    
    # Mean IoU usando jaccard_score
    # El IoU para multi-clase también depende de "average":
    miou = jaccard_score(y_true, y_pred, average=average)
    
    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'mIoU': miou
    }
    if loss_value is not None:
        metrics['loss'] = loss_value
    
    return metrics


def plot_metrics_over_epochs(metrics_list, save_path=None):
    """
    Dibuja la evolución de varias métricas (accuracy, precision, recall, mIoU, loss)
    a lo largo de las épocas. 
    metrics_list es una lista de diccionarios, uno por epoch, con claves:
    ['accuracy', 'precision', 'recall', 'mIoU', 'loss' (opcional)].
    
    Args:
        metrics_list (list[dict]): Cada elemento corresponde a una época.
            Ejemplo: metrics_list[epoch] = {
                'accuracy': 0.88,
                'precision': 0.85,
                'recall': 0.86,
                'mIoU': 0.80,
                'loss': 0.34
            }
        save_path (str, opcional): si se indica, guarda la figura en esa ruta.
    """
    if not metrics_list:
        print("No hay métricas para graficar.")
        return

    epochs = range(1, len(metrics_list) + 1)
    
    # Extraer cada métrica en forma de lista
    accuracy_values  = [m['accuracy']  for m in metrics_list if 'accuracy'  in m]
    precision_values = [m['precision'] for m in metrics_list if 'precision' in m]
    recall_values    = [m['recall']    for m in metrics_list if 'recall'    in m]
    miou_values      = [m['mIoU']      for m in metrics_list if 'mIoU'      in m]
    loss_values      = [m['loss']      for m in metrics_list if 'loss'      in m]

    # Crear subplots (2x3)
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    # (0,0) -> Accuracy, (0,1) -> Precision, (0,2) -> Recall
    # (1,0) -> mIoU,     (1,1) -> Loss,      (1,2) -> vacío

    # Accuracy
    if accuracy_values:
        axs[0,0].plot(epochs, accuracy_values, marker='o', label='Accuracy')
        axs[0,0].set_title("Accuracy")
        axs[0,0].set_xlabel("Epoch")
        axs[0,0].set_ylabel("Value")
    else:
        axs[0,0].set_visible(False)

    # Precision
    if precision_values:
        axs[0,1].plot(epochs, precision_values, marker='o', color='orange', label='Precision')
        axs[0,1].set_title("Precision")
        axs[0,1].set_xlabel("Epoch")
        axs[0,1].set_ylabel("Value")
    else:
        axs[0,1].set_visible(False)

    # Recall
    if recall_values:
        axs[0,2].plot(epochs, recall_values, marker='o', color='green', label='Recall')
        axs[0,2].set_title("Recall")
        axs[0,2].set_xlabel("Epoch")
        axs[0,2].set_ylabel("Value")
    else:
        axs[0,2].set_visible(False)

    # mIoU
    if miou_values:
        axs[1,0].plot(epochs, miou_values, marker='o', color='red', label='mIoU')
        axs[1,0].set_title("Mean IoU")
        axs[1,0].set_xlabel("Epoch")
        axs[1,0].set_ylabel("Value")
    else:
        axs[1,0].set_visible(False)

    # Loss
    if loss_values:
        axs[1,1].plot(epochs, loss_values, marker='o', color='purple', label='Loss')
        axs[1,1].set_title("Loss")
        axs[1,1].set_xlabel("Epoch")
        axs[1,1].set_ylabel("Value")
    else:
        axs[1,1].set_visible(False)

    # Último subplot vacío
    axs[1,2].set_visible(False)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def plot_confusion_matrix_custom(y_true, y_pred, classes=None, normalize=None, save_path=None):
    """
    Grafica la matriz de confusión, con la opción de normalizar.
    Usa sklearn.metrics.confusion_matrix y ConfusionMatrixDisplay.
    
    Args:
        y_true (array-like): Etiquetas reales (N).
        y_pred (array-like): Etiquetas predichas (N).
        classes (list[str], opcional): nombres de las clases.
        normalize (str, opcional): 'true', 'pred', 'all' o None (ver sklearn docs).
        save_path (str, opcional): ruta para guardar la figura.
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format=".2f" if normalize else "d")
    plt.title("Matriz de Confusión")
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def save_metrics_to_csv(metrics_list, csv_path="training_metrics.csv"):
    """
    Guarda en CSV el contenido de metrics_list, donde cada elemento es un dict con las 
    métricas de cada epoch.
    Ejemplo de un elemento: {
        'epoch': 1,
        'train_loss': 0.45,
        'val_accuracy': 0.78,
        'val_precision': 0.80,
        ...
    }
    
    Args:
        metrics_list (list[dict]): Lista de diccionarios con las métricas de cada epoch.
        csv_path (str): Ruta para el archivo CSV de salida.
    """
    if not metrics_list:
        print("No hay métricas para guardar en CSV.")
        return

    # Obtener las columnas a partir de las llaves del primer diccionario
    fieldnames = list(metrics_list[0].keys())

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_list:
            writer.writerow(row)
    
    print(f"Métricas guardadas en {csv_path}")
