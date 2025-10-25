import numpy as np
from optimisers.config.RegisterDecorator import register_optimiser
from optimisers.tools.optimiserParent import OptimiserParentClass
from optimisers.tools.utilities import Utils
from optimisers.MathematicsTools import SingularValueDecomposition, NewtonSchulz
import torch


@register_optimiser('Muon')
class Muon(OptimiserParentClass):
    def __init__(self, params, weight_decay, lr, beta, numIters):
        super().__init__(params, lr)

        self.weight_decay = weight_decay
        self.beta = beta
        self.numIters = numIters
        self.utils = Utils()

        self.ns = NewtonSchulz
        self.svd = SingularValueDecomposition

        self.momentum = {id(p): torch.zeros_like(p.data) for p in self.params}

    def update_param(self, p):
        grad = p.grad

        if grad is None:
            return p.data

        if isinstance(grad, np.ndarray):
            grad = torch.from_numpy(grad).to(dtype=torch.float32)

        if self.weight_decay > 0:
            grad = self.utils.weight_decay(self.weight_decay, grad, p)

        pid = id(p)
        m_prev = self.momentum[pid]
        m_t = self.beta * m_prev + (1 - self.beta) * grad
        self.momentum[pid] = m_t  # store momentum

        # Orthogonalisation step
        A = m_t.T @ m_t

        if self.use_newton:
            inv_sqrt = NewtonSchulz.inv_sqrt(A, self.num_iters)
            O_t = m_t @ inv_sqrt
        else:
            O_t = self.svd.SingularValueDecomposition.svd_inv_sqrt(m_t)

        p.data = p.data - self.lr * O_t

        return p.data
