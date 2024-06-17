"""
ipca.py
"""
import torch
from torch import nn
import numpy as np


class IncrementalPCA(nn.Module):
    """
    Performs incremental PCA, building a fixed-rank approxiamtion to the data covariance matrix.
    """
    def __init__(
        self, feature_dim, n_components, center=False, device=torch.device("cpu")
    ):
        super().__init__()

        self.n_samples = 0  # how many samples we've seen
        self.n_components = n_components
        self.feature_dim = feature_dim

        self.center = center

        self.running_mean = torch.zeros(feature_dim, device=device)  # mean feature

        # top eigenvectors of past data
        self.components = torch.zeros(
            n_components, feature_dim, device=device
        )

        # singular values of past data
        self.singular_vals = torch.randn(
            n_components, device=device
        )

    def forward(self, batch_inputs):
        """
        processes batch of inputs:
            batch_inputs: [n_batch, feature_dim]
        """
        n_batch, feature_dim = batch_inputs.shape
        assert feature_dim == self.feature_dim

        n_total = self.n_samples + n_batch

        batch_mean = torch.zeros_like(self.running_mean)
        mean = batch_mean
        if self.center:
            batch_mean = batch_inputs.mean(dim=0)

            mean = self.n_samples * self.running_mean + n_batch * batch_mean
            mean /= n_total

        mean_correction = np.sqrt((self.n_samples * n_batch) / n_total) * (
            self.running_mean - batch_mean
        )

        X = torch.cat(
            [
                self.singular_vals[:, None] * self.components,
                (batch_inputs - batch_mean[None, :]),
                mean_correction[None, :],
            ],
            dim=0,
        )

        _, S, Vt = torch.linalg.svd(X, full_matrices=False)

        # flip vectors such that max absolute value entry is postive
        max_abs_rows = torch.max(torch.abs(Vt), dim=1).indices
        signs = torch.sign(Vt[torch.arange(Vt.shape[0]), max_abs_rows])
        Vt *= signs[:, None]

        explained_variance = (S**2) / (n_total - 1)
        noise_variance = explained_variance[
            self.n_components :
        ].mean()  # unexplained noise

        # update parameters
        self.n_samples = n_total
        if self.center:
            self.running_mean = mean

        self.components = Vt[: self.n_components, :]
        self.singular_vals = S[: self.n_components]

        return noise_variance

    def get_components(self):
        """
        returns current value of singular vals and components
        """
        return self.singular_vals, self.components
