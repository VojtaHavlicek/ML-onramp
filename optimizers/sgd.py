"""Stochastic Gradient Descent (SGD) optimizer implementation."""

class SGD:
    """Stochastic Gradient Descent optimizer."""

    def __init__(self,
                 params, 
                 lr=0.01) -> None:# noqa: ANN001
        """Initialize the SGD optimizer.

        Args:
            params: list of dicts with {"value": np.ndarray, "grad": np.ndarray}
            lr: learning rate

        """
        self.params = params
        self.lr = lr

    def step(self) -> None:
        """Update parameters in-place."""
        for param in self.params:
            if param["grad"] is None:
                continue # Skip if no gradient was computed.
            param["value"] -= self.lr * param["grad"]

    def zero_grad(self) -> None:
        """Reset gradients to zero (important for next step)."""
        for param in self.params:
            if param["grad"] is not None:
                param["grad"].fill(0)
