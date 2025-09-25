from .SGD import SGD
from .NAG import NAG


from Config.RegisterDecorator import register_optimiser

__all__ = ["SGD", "Adagrad", "NAG"]