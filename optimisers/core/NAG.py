import numpy as np
from optimisers.config.RegisterDecorator import register_optimiser
from optimisers.tools.optimiserParent import OptimiserParentClass
from optimisers.tools.utilities import Utils


@register_optimiser('NAG')
class NAG(OptimiserParentClass):
    def __init__(self, parameters, momentum, lr, weight_decay):
        super().__init__(parameters, lr)
        self.weight_decay = float(weight_decay)
        self.utils = Utils()
        self.momentum = float(momentum)
        self.velocity = {id(p): np.zeros_like(p.data) for p in self.params}  # 0 for every param in params with lookup

    def update_param(self, p):
        grad = p.grad

        if self.weight_decay > 0:
            grad = self.utils.weight_decay(self.weight_decay, grad, p)  # Handles the weight decay

        v = self.velocity[id(p)]  # Lookup velocity in dict

        v = self.momentum * v + grad
        d_p = grad + self.momentum * v
        p.data = p.data - self.lr * d_p  # update parameter to move away from increasing gradient

        self.velocity[id(p)] = v  # Update velocity in lookup to new velocity after momentum

        return p.data
