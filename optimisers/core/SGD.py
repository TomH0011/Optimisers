from optimisers.config.RegisterDecorator import register_optimiser
from optimisers.tools.optimiserParent import OptimiserParentClass
from optimisers.tools.utilities import Utils


@register_optimiser('SGD')
class SGD(OptimiserParentClass):
    def __init__(self, params, lr, weight_decay):
        super().__init__(params, lr)
        self.weight_decay = weight_decay
        self.utils = Utils()

    def update_param(self, p):
        grad = p.grad

        if self.weight_decay > 0:
            grad = self.utils.weight_decay(self.weight_decay, grad, p)  # Handles the weight decay

        p.data = p.data - self.lr * grad

        return p.data
