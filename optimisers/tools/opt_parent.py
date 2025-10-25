# Aim is to create a parent class which has tools for all optimisers
# e.g. contains methods like step - zero_grad should also go here
# also contain method for storing parameters
from optimisers.config.RegisterDecorator import OPTIMISER_REGISTRY


class OptimiserParentClass:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    # Generalised step method for all optimisers
    def step(self):
        for p in self.params:
            if getattr(p, 'grad', None) is None:
                continue
            # Calls upon the child optimiser to  use their update param method else use the one in the parent class
            # The parent class update_param method raises an error
            self.update_param(p)
        return None

    # Only really needs using when user is also using auto_grad = True from Pytorch
    def zero_grad(self):
        for p in self.params:
            g = getattr(p, "grad", None)
            if g is None:
                continue

            # NumPy case
            if hasattr(g, "fill"):
                g.fill(0.0)

            # PyTorch case
            elif hasattr(g, "zero_"):
                g.zero_()

    def store_parameters(self):
        raise NotImplementedError('Method has not been created... YET!')

    def update_param(self, p):
        # Each optimiser subclass must implement this
        raise NotImplementedError


# Try and build the optimiser requested from registry
def build(name, params, **kwargs):
    # If optimiser isnt made... dont try and build it
    if name not in OPTIMISER_REGISTRY:
        raise ValueError(f'Optimiser {name} not found in registry')
    return OPTIMISER_REGISTRY[name](params, **kwargs)
