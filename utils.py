"""Activation functions for neural networks.

Author:
Vojta Havlicek, 2025
"""
import os
import random

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from constants import DEVICE, SEED


# --- ACTIVATION FUNCTIONS ---
def ReLU(x:np.ndarray) -> np.ndarray:  # noqa: N802,
    """ReLU activation function.

    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Output array after applying ReLU.

    """
    return np.maximum(0, x)


def softmax(x) -> np.ndarray:
    """Softmax activation function.

    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Output array after applying softmax.

    """
    e_x = np.exp(x-np.max(x)) # for numerical stability
    return e_x / e_x.sum(axis=0, keepdims=True)


# --- LOSS FUNCTIONS ---
def cross_entropy(logits: np.ndarray, y:np.ndarray) -> float: 
    """Cross-entropy loss function.

    Args:
        logits (np.ndarray): Logits from the model.
        y (np.ndarray): True labels (one-hot encoded).

    Returns:
        float: Cross-entropy loss value.

    """
    log_probs = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
    return -np.sum(y * log_probs)


# --- TRAINING UTILITIES ---
def batch_confusion_matrix(predictions: torch.Tensor,
                           targets: torch.Tensor,
                           num_classes: int = 10) -> torch.Tensor:
    """Compute confusion matrix for a batch.

    Args:
        predictions (torch.Tensor): Predictions, shape (N,).
        targets (torch.Tensor): Ground-truth targets, shape (N,).
        num_classes (int): Number of classes.

    Returns:
        torch.Tensor: Confusion matrix, shape (num_classes, num_classes).

    """
    with torch.no_grad():
        k = (targets * num_classes + predictions).to(torch.int64)
        bincount = torch.bincount(k, minlength=num_classes**2)
        return bincount.reshape(num_classes, num_classes).to(torch.int64)


def train_epoch(model:nn.Module,
                loader:DataLoader,
                optimizer:optim.Optimizer,
                loss:nn.Module) -> tuple:
    """Trains one epoch.

    Args:
        model (nn.Module): Model to train.
        loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer to use.
        loss (callable): Loss function.

    Returns:
        tuple: (average loss, accuracy)

    """
    model.train()
    total_loss, total, correct = 0.0, 0, 0
    for (x, y) in loader:
        x_device, y_device = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_device)
        batch_loss = loss(logits, y_device)
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item() * y_device.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y_device).sum().item()
        total += x_device.size(0)

    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model:nn.Module,
             loader:DataLoader,
             loss:nn.Module,
             *, # forces keyword-only arguments
             compute_confmat: bool = False) -> tuple:
    """Evaluate model on the given DataLoader.

    Args:
        model (nn.Module): _description_
        loader (DataLoader): _description_
        loss (nn.Module): _description_
        compute_confmat (bool, optional): _description_. Defaults to False.

    Returns:
        tuple: _description_

    """
    model.eval()
    total_loss, total, correct = 0.0, 0, 0
    if compute_confmat:
        c = torch.zeros((10,10), dtype=torch.int32)

    for x, y in loader:
        x_device, y_device = x.to(DEVICE), y.to(DEVICE)
        logits = model(x_device)
        loss_value = loss(logits, y_device) # Note: loss takes logits/targets, not probabilities
        total_loss += loss_value.item() * y_device.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y_device).sum().item()
        total += x_device.size(0)

        if compute_confmat:
            c += batch_confusion_matrix(preds.cpu(), y_device.cpu(), num_classes=10)

    if compute_confmat:
        diag = c.diagonal().to(torch.float32)
        support = c.sum(dim=1).to(torch.float32)
        per_class_acc = (diag / support).tolist()
        return total_loss / total, correct / total, c, per_class_acc

    return total_loss / total, correct / total


# ---- Reproducibility ----
def set_seed(seed:int=SEED) -> None:
    """Set random seed for reproducibility.

    Args:
        seed (int, optional): Defaults to 42.

    """
    random.seed(seed); 
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    torch.backends.mps.deterministic = True
    torch.backends.mps.benchmark = False
set_seed(SEED)


