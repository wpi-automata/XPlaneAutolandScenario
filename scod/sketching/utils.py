"""
Convenience functions for sketching / randomized PCA
"""
import torch
import numpy as np


def idct(X, norm=None):
    """
    based on https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py
    updated to work with more recent versions of pytorch which moved fft functionality to
    the torch.fft module
    """
    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == "ortho":
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = (
        torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :]
        * np.pi
        / (2 * N)
    )
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)

    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]

    return x.view(*x_shape)


def random_subslice(
    tensor: torch.Tensor, dim: int, k: int, scale=False, return_idx=False
):
    """
    returns a random slice of tensor by choosing random indices in dim
    NOTE: the subselected rows will be ordered as they originally were
    if k >= tensor.shape[dim], we skip computation entirely

    if scale is True, then we multiply by sqrt(T)/sqrt(k), such that
    if P is the matrix implementing this project, E[P^T P] = I
    """
    assert -len(tensor.shape) < dim < len(tensor.shape)
    if tensor.shape[dim] <= k:
        return tensor

    indices = torch.argsort(torch.rand(tensor.shape[dim], device=tensor.device))[:k]
    result = torch.index_select(tensor, dim, indices)

    if scale:
        factor = np.sqrt(tensor.shape[dim] / k)
        result *= factor

    if return_idx:
        return result, indices

    return result
