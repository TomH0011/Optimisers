from optimisers.core import Adagrad
from optimisers.core.SGD import SGD
from optimisers.core.NAG import NAG
from optimisers.core.Adam import Adam
from optimisers.core.Muon import Muon

from config.RegisterDecorator import register_optimiser

__all__ = ["SGD", "Adagrad", "NAG", "Adam", "Muon"]
