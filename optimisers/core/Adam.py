import numpy as np
from optimisers.config.RegisterDecorator import register_optimiser
from optimisers.tools.optimiserParent import OptimiserParentClass
from optimisers.tools.utilities import Utils


@register_optimiser('Adam')
class Adam(OptimiserParentClass):
    def __init__(self, params, weight_decay, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params, lr)
        self.weight_decay = weight_decay
        self.utils = Utils()

        self.bias_1 = beta1  # decay factor
        self.bias_2 = beta2  # decay factor
        self.epsilon = eps  # for avoiding div by 0

        self.moment_1 = {id(p): np.zeros_like(p) for p in self.params}  # exponential moving average of gradients
        self.moment_2 = {id(p): np.zeros_like(p) for p in
                         self.params}  # exponential moving average of squared gradients

        self.t = 0  # step count for bias correction

    def update_param(self, p):
        grad = p.grad

        if self.weight_decay > 0:
            grad = self.utils.weight_decay(self.weight_decay, grad, p)

        pid = id(p)
        m = self.m[pid]
        v = self.v[pid]

        # Update biased first moment estimate
        m = self.beta1 * m + (1 - self.beta1) * grad
        # Update biased second moment estimate
        v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

        # Store back
        self.m[pid] = m
        self.v[pid] = v

        # Bias correction/ normalisation of both moments
        self.t += 1
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)

        # Update parameter
        p.data = p.data - self.lr * (m_hat / (np.sqrt(v_hat) + self.eps))

        return p.data
