import torch


class SingularValueDecomposition:
    @staticmethod
    def svd_inv_sqrt(A, eps=1e-8):
        U, S, Vh = torch.linalg.svd(A)
        S_inv_sqrt = torch.diag(1.0 / torch.sqrt(S + eps))
        return Vh.T @ S_inv_sqrt @ U.T
