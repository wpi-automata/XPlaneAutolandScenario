"""
sketch_ops.py
"""
import numpy as np
import torch
import torch.nn as nn

from .utils import idct


class SketchOperator(nn.Module):
    """
    implements d x N linear operator for sketching
    """

    def __init__(self, d, N):
        """
        d x N operator
        """
        self.d = d
        self.N = N
        super().__init__()

    def forward(self, M, transpose=False):
        """
        implements \tilde{M} = S M, right multiplication by M

        assumes M of a compatible shape, i.e. M is (N, ..)

        if transpose, then compute \tilde{M}= M S^T (assumes M is (..., N))
        """
        raise NotImplementedError


class GaussianSketchOp(SketchOperator):
    """
    Gaussian sketch
    """

    def __init__(self, d, N, device=torch.device("cpu")):
        super().__init__(d, N)
        self.test_matrix = nn.Parameter(
            torch.randn(d, N, dtype=torch.float, device=device), requires_grad=False
        )

    @torch.no_grad()
    def forward(self, M, transpose=False):
        if transpose:
            return M @ self.test_matrix.t()
        return self.test_matrix @ M


class SRFTSketchOp(SketchOperator):
    """
    SRFT sketch
    """

    def __init__(self, d, N, device=torch.device("cpu")):
        super().__init__(d, N)
        self.D = nn.Parameter(
            2 * (torch.rand(N, device=device) > 0.5).float() - 1, requires_grad=False
        )
        self.P = np.random.choice(N, d)  # choose d elements from n

    @torch.no_grad()
    def forward(self, M, transpose=False):
        if transpose:
            M = M.t()

        if M.dim() == 2:
            result = idct((self.D[:, None] * M).t()).t()[self.P, :]
        elif M.dim() == 1:
            result = idct(self.D * M)[self.P]
        else:
            raise ValueError

        if transpose:
            result = result.t()

        return result
