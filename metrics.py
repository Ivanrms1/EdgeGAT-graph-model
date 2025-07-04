# metrics.py
import torch
import numpy as np

def accuracy(predictions, labels):
    correct = (predictions == labels).sum().item()
    total = labels.numel()
    return correct / total

def recall_per_class(predictions, labels, num_classes):
    recall_vals = []
    for c in range(num_classes):
        tp = ((predictions == c) & (labels == c)).sum().item()
        fn = ((predictions != c) & (labels == c)).sum().item()
        recall_class = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recall_vals.append(recall_class)
    return np.mean(recall_vals)  # Solo retornamos el promedio

def iou_per_class(predictions, labels, num_classes):
    iou_vals = []
    for c in range(num_classes):
        tp = ((predictions == c) & (labels == c)).sum().item()
        fp = ((predictions == c) & (labels != c)).sum().item()
        fn = ((predictions != c) & (labels == c)).sum().item()
        denom = (tp + fp + fn)
        iou_class = tp / denom if denom > 0 else 0.0
        iou_vals.append(iou_class)
    return np.mean(iou_vals)  # Solo retornamos el promedio

def evaluate_metrics(outputs, labels, criterion, num_classes):
    # outputs: [B, N, C]
    # labels: [B, N]
    # Calcular loss
    B, N, C = outputs.shape
    #loss = criterion(outputs.view(-1, C), labels.view(-1))
    loss = criterion(outputs.reshape(-1, C), labels.reshape(-1))

    preds = outputs.argmax(dim=-1)  # [B, N]

    acc = accuracy(preds, labels)
    recall_mean = recall_per_class(preds, labels, num_classes)
    iou_mean = iou_per_class(preds, labels, num_classes)

    return {
        'loss': loss.item(),
        'accuracy': acc,
        'recall_mean': recall_mean,
        'iou_mean': iou_mean
    }
