from .SGD import SGD
from .NAG import NAG
from .Adam import Adam


from Config.RegisterDecorator import register_optimiser

__all__ = ["SGD", "Adagrad", "NAG", "Adam"]