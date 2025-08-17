"""Activation functions for neural networks.

Author:
Vojta Havlicek, 2025
"""

import numpy as np

# --- ACTIVATION FUNCTIONS ---
def ReLU(x) -> np.ndarray:  # noqa: N802,
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



