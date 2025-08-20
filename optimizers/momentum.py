"""Momentum SGD Optimizer."""

import numpy as np
from typing import Any, Callable, List, Sequence, Union, Tuple, Dict, Optional

class Momentum:
    """Stochastic Gradient Descent optimizer with momentum (and optional Nesterov)."""

    def __init__(self, 
                 params:List[Dict[str, np.ndarray]],
                 lr=0.01,
                 beta: float = 0.9,
                 nesterov: bool = False,
                 eps:float = 0.0,
                 weight_decay: float = 0.0) -> None:
        """Initialize the Momentum SGD optimizer.

        Args:
            params: list of dicts with {"value": np.ndarray, "grad": np.ndarray}
            lr: learning rate
            beta: momentum factor (0.0 <= momentum < 1.0)
            nesterov: use Nesterov momentum (default: False)
            eps: dampening factor (0.0 <= dampening < 1.0)
            weight_decay: L2 weight decay (default: 0.0)

        """
        if not (0.0 <= beta < 1.0):
            raise ValueError("Momentum must be in [0.0, 1.0).") # noqa: EM101, TRY003
        if not (0.0 <= eps < 1.0):
            raise ValueError("Dampening must be in [0.0, 1.0).") # noqa: EM101, TRY003
        if weight_decay < 0.0:
            raise ValueError("Weight decay must be non-negative.") # noqa: EM101, TRY003

        self.params = params
        self.lr = lr
        self.beta = beta
        self.nesterov = nesterov
        self.eps = eps
        self.weight_decay = weight_decay

        # Per-parameter momentum buffers
        self._vel: List[Optional[np.ndarray]] = [None] * len(params)

    def step(self) -> None:
        """Update parameters in-place."""

        for index, parameter in enumerate(self.params):
            gradient = parameter["grad"]
            if gradient is None:
                continue

            # L2 weight decay (coupled)
            if self.weight_decay > 0.0:
                gradient += self.weight_decay * parameter["value"]

            # Init / update velocity buffer
            v = self._vel[index]
            if v is None:
                v = np.zeros_like(parameter["value"])

            # Apply momentum
            v = self.beta * v + (1.0 - self.eps) * gradient
            g_eff = gradient + self.beta * v if self.nesterov else v

            # Update parameter
            parameter["value"] -= self.lr * g_eff
            self._vel[index] = v


    def zero_grad(self) -> None:
        """Reset gradients to zero (important for next step)."""
        for param in self.params:
            if param["grad"] is not None:
                param["grad"].fill(0)
