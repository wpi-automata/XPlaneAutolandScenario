"""
Tools for extracting low-rank approximations from sketched matrices

Not quite PCA because many of these algorithms don't center the data
"""

import torch
from .sketch_ops import GaussianSketchOp, SRFTSketchOp
from .ipca import IncrementalPCA as iPCA



@torch.no_grad()
def low_rank_approx(Y, W, Psi_fn):
    """
    given Y = A @ Om, (N, k)
    and W = Psi @ A, (l, M)
    and Psi_fn(X) = Psi @ X, (N,...) -> (l,...)
    where Om and Psi and random sketching operators

    returns Q (N x k), X (k x M) such that A ~= QX
    """
    Q, _ = torch.linalg.qr(Y, "reduced")  # (N, k)
    U, T = torch.linalg.qr(Psi_fn(Q), "reduced")  # (l, k), (k, k)
    X = torch.linalg.solve_triangular(T, U.t() @ W, upper=True) # (k, N)

    return Q, X


@torch.no_grad()
def fixed_rank_svd_approx(Y, W, Psi_fn, r):
    """
    given Y = A @ Om, (N, k)
    and W = Psi @ A, (l, M)
    and Psi_fn(X) = Psi @ X, (N,...) -> (l,...)
    where Om and Psi and random sketching operators

    and a choice of r < k

    returns U (N x r), S (r,), V (M x r) such that A ~= U diag(S) V.T
    """
    Q, X = low_rank_approx(Y, W, Psi_fn)
    U, S, Vh = torch.lianlg.svd(X)
    U = Q @ U[:, :r]

    return U, S[:r], Vh[:r, :]


@torch.no_grad()
def sym_low_rank_approx(Y, W, Psi_fn):
    """
    given Y = A @ Om, (N, k)
    and W = Psi @ A, (l, N)
    and Psi_fn(X) = Psi @ X, (N,...) -> (l,...)
    where Om and Psi and random sketching operators

    returns U (N x 2k), S (2k x 2k) such that A ~= U S U^T
    """
    Q, X = low_rank_approx(Y, W, Psi_fn)
    k = Q.shape[-1]
    U, T = torch.linalg.qr(torch.cat([Q, X.t()], dim=1), "reduced")  # (N, 2k), (2k, 2k)
    del Q, X
    T1 = T[:, :k]  # (2k, k)
    T2 = T[:, k : 2 * k]  # (2k, k)
    S = (T1 @ T2.t() + T2 @ T1.t()) / 2  # (2k, 2k)

    return U, S


@torch.no_grad()
def fixed_rank_eig_approx(Y, W, Psi_fn, r):
    """
    returns U (N x r), D (r) such that A ~= U diag(D) U^T
    """
    U, S = sym_low_rank_approx(Y, W, Psi_fn)
    D, V = torch.linalg.eigh(S)  # (2k), (2k, 2k)
    D = D[-r:]
    V = V[:, -r:]  # (2k, r)
    U = U @ V  # (N, r)
    return U, D


class SinglePassPCA:
    """
    computes a sketch of AA^T when presented columns of A sequentially
    then, uses eigenvalue decomp of sketch to compute
    rank r range basis
    """

    def __init__(
        self,
        N,  # A.shape[0]
        r,  # number of eigenvectors to collect
        T=None,  # total sketch dimension to use
        sketch_op_class=GaussianSketchOp,  # sketch operator to use
        device=torch.device("cpu"),  # device on which to create sketch
    ):
        self.N = N
        self.r = r
        self.T = T
        if T is None:
            self.T = 6 * r + 4
        print("using T =", self.T)

        self.device = device

        self.k = max(self.r + 2, (self.T - 1) // 3)
        self.l = self.T - self.k

        # construct sketching operators
        self.Om = sketch_op_class(self.k, self.N, device=self.device)
        self.Psi = sketch_op_class(self.l, self.N, device=self.device)

        # sketch data
        self.Y = torch.zeros(self.N, self.k, dtype=torch.float, device=self.device)
        self.W = torch.zeros(self.l, self.N, dtype=torch.float, device=self.device)

        super().__init__()

    @torch.no_grad()
    def low_rank_update(self, v, weight=1.0):
        """
        processes v (nparam x d) the a batch of columns of matrix A
        self.Y += weight v v^T Om
        self.W += weight Psi v v^T
        """
        v = v.to(self.device)
        torch.addmm(
            self.Y, weight * v, self.Om(v.t(), transpose=True), alpha=1, out=self.Y
        )
        torch.addmm(self.W, weight * self.Psi(v), v.t(), alpha=1, out=self.W)

    @torch.no_grad()
    def eigs(self):
        """
        returns a basis for the range of the top r left singular vectors
        (right now, returns all 2k eigenvectors)
        returns D, the eigenvalues, and U, the eigenvectors, in ascending order
        """
        self.Y = self.Y.cpu()
        self.W = self.W.cpu()
        self.Om.cpu()
        self.Psi.cpu()
        U, D = fixed_rank_eig_approx(self.Y, self.W, self.Psi, 2 * self.k)
        return D, U


class SRFT_SinglePassPCA(SinglePassPCA):
    """
    computes a subsampled randomized fourier transform sketch
    of AA^T when presented columns of A sequentially.

    then, uses eigen decomp of sketch to compute
    rank r range basis
    """

    def __init__(self, N, r, T=None, device=torch.device("cpu")):
        super().__init__(N, r, T, device=device, sketch_op_class=SRFTSketchOp)


class IncrementalPCA(iPCA):
    """
    implements SinglePassPCA interface using iPCA backend
    """

    def __init__(self, N, r, T=None, device=torch.device("cpu")):
        super().__init__(feature_dim=N, n_components=r, device=device)
        self.counter = 0
        if T is None:
            T = 50
        self.batch_size = T
        self.batch = []

    @torch.no_grad()
    def low_rank_update(self, v):
        r"""
        if v is a vector, then it is straightforward

        how to handle jacobian matrix?
            J = \sum_j \sigma_j u_j v_j^T
            if we want V L V^T -> \sum_i J_i^T J_i
             -> \sum_i \sum_j v_j \sigma_{i,j} u_j^T u_j v_j^T
             -> \sum_i \sum_j v_{i,j} \sigma_{i,j} v_{i,j}

        i.e., given v, compute u,s,vt = SVD(v)
            then, treat s[:,None]*vt as a batch of inputs
        """
        self.batch.append(v.T)
        self.counter += 1

        if self.counter % self.batch_size == 0:
            # evaluate batch
            batch = torch.stack(self.batch)
            self.batch = []  # reset for next batch

            _, d, N = batch.shape
            if d > 1:
                _, s, Vt = torch.linalg.svd(batch, full_matrices=False)
                batch = s[:, :, None] * Vt

            print(batch.shape)
            self.forward(batch.reshape(-1, N))

    def eigs(self):
        """
        returns a basis for the range of the top r left singular vectors
        (right now, returns all 2k eigenvectors)
        returns D, the eigenvalues, and U, the eigenvectors, in ascending order
        """
        return (
            torch.flip(self.singular_vals**2, [0]),
            torch.flip(self.components, [0]).T,
        )


alg_registry = {
    "gaussian": SinglePassPCA,
    "srft": SRFT_SinglePassPCA,
    "iPCA": IncrementalPCA,
}
