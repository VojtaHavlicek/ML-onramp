"""AdamW Optimizer Implementation."""

import numpy as np

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
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
            self.m[i] = self.beta1 * self.m[i] + (1-self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1-self.beta2) * (g*g)

            # bias correction
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # update: Adam step + decoupled weight decay
            p["value"] -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * p["value"])

    def zero_grad(self):
        for p in self.params:
            if p["grad"] is not None:
                p["grad"].fill(0.0)