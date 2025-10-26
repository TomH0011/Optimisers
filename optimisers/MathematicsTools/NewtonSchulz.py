import torch

class NewtonSchulz:
    @staticmethod
    def inv_sqrt(A, numIters):
        # Keep numIters at 5
        # Normalise matrix by its Frobenius norm to stabilise iteration
        normA = torch.norm(A)
        Y = A / normA
        # both identity tensors of size A on same device as A
        I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        Z = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)

        for _ in range(numIters):
            T = 0.5 * (3 * I - Z @ Y)
            Y = Y @ T
            Z = T @ Z

        # Now Z approximates A^{-1/2}, rescale by sqrt(normA)
        return Z / torch.sqrt(normA)
