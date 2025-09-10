# mlp.py
"""A simple 2-layer Multi-Layer Perceptron (MLP) class."""

from constants import *
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from utils import ReLU, softmax, cross_entropy
from typing import List, Dict


class MLP(nn.Module):
    def __init__(self,
                 input_dimension=28*28,
                 hidden_dimensions=(256,128),
                 output_dimension=10,
                 droupout_probability=0.1,
                 use_batchnorm=False):
        """Pytorch Implementation of a one hidden layer MLP. Initializes weights and biases for the two layers at random.

        Args:
            input_dimension (int): Dimensionality of input data (D).
            hidden_dimensions (tuple of int): Dimensionality of hidden layers.
            output_dimension (int): Number of classes (C).
            droupout_probability (float): Dropout probability, between 0 and 1.
            use_batchnorm (bool): Whether to use batch normalization after each hidden layer.

        """
        super().__init__()
        layers = []

        last_dimension = input_dimension
        for hidden_dimension in hidden_dimensions:
            layers.append(nn.Linear(last_dimension, hidden_dimension))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dimension))
            layers.append(nn.ReLU())
            if droupout_probability > 0:
                layers.append(nn.Dropout(droupout_probability))
            last_dimension = hidden_dimension

        layers.append(nn.Linear(last_dimension, output_dimension))
        self.net = nn.Sequential(*layers)


    def forward(self, X):
        N = X.shape[0]
        X = X.view(N, -1) # reshape (N, 28, 28) -> (N, 28*28)
        return self.net(X)


