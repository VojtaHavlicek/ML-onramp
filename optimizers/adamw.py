"""AdamW Optimizer Implementation."""

import numpy as np

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if not (0.0 <= betas[0] < 1.0) or not (0.0 <= betas[1] < 1.0):
            raise ValueError("Momentum (betas) must be in [0.0, 1.0).") # noqa: EM101, TRY003
        if not (0.0 <= eps < 1.0):
            raise ValueError("Dampening (eps) must be in [0.0, 1.0).") # noqa: EM101, TRY003
        if weight_decay < 0.0:
            raise ValueError("Weight decay must be non-negative.") # noqa: EM101, TRY003

        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p["value"]) for p in params]
        self.v = [np.zeros_like(p["value"]) for p in params]


    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p["grad"]
            if g is None: continue

            # moment estimates
            # 1. (Momentum) use exponential moving average for momentum
            self.m[i] = self.beta1 * self.m[i] + (1-self.beta1) * g 

            # 2. (RMSProp) use exponential moving average for squared gradients
            self.v[i] = self.beta2 * self.v[i] + (1-self.beta2) * (g*g)

            # bias correction
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # update: Adam step + decoupled weight decay
            p["value"] -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * p["value"])

            # This means that the weight decay is applied directly to the parameters,
            # rather than to the gradients, which is the key difference from standard Adam.

    def zero_grad(self):
        for p in self.params:
            if p["grad"] is not None:
                p["grad"].fill(0.0)