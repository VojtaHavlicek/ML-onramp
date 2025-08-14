import numpy as np

def ReLU(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x-np.max(x)) # for numerical stability
    return e_x / e_x.sum(axis=0, keepdims=True)