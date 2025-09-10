import torch

DEVICE = "mps" if torch.mps.is_available() else "cpu"
SEED = 42