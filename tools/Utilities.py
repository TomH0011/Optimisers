# Contains optional use cases as a tool for each optimiser
# This includes Weight decay
# Gradient Clipping
# Learning rate scheduler

class Utils:
    def __init__(self):
        self.TorchTensor = None
        pass

    def weight_decay(self, decay, gradient, parameters) -> (list, tuple, self.TorchTensor):
        for p in parameters:
            decayed_gradient = gradient + decay * p.data
        if isinstance(decayed_gradient, (list, tuple, self.TorchTensor)):
            return decayed_gradient
        else:
            print(f'Type error for decayed gradient in class utils weight_decay method, ensure decayed_gradients is '
                  f'of correct type')
            raise TypeError
