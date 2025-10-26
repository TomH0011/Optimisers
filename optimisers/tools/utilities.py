# Contains optional use cases as a tool for each optimiser
# This includes Weight decay
# Gradient Clipping
# Learning rate scheduler

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class Utils:
    def __init__(self):
        self.TorchTensor = None
        pass

    def weight_decay(self, decay, gradient, param):
        decayed_gradient = gradient + decay * param.data

        # Check type
        valid_types = (np.ndarray,)
        if TORCH_AVAILABLE:
            valid_types = valid_types + (torch.Tensor,)

        if isinstance(decayed_gradient, valid_types):
            return decayed_gradient
        else:
            raise TypeError(
                f"decayed_gradient has wrong type: {type(decayed_gradient)}. "
                f"Expected one of {valid_types}."
            )
