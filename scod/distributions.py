"""
this file implements different output distributional families, with their loss functions and Fisher matrices

specifically, for each family, if F(theta) = LL^T
we implement left multiplication by L^T using the function

apply_sqrt_F

we also implement:

marginalize(var): which returns the distribution if the parameters used to construct the original 
    distribution were normally distributed with diagonal variance given by var

merge_batch(): returns the distribution approximating the mixture distribution constructed by 
    summing across the first batch dimension. useful if you construct this distribution with a batch
    of parameters corresponding to MC samples, and you want a single distribution approximation of the mixture.

metric(label): which returns a more human friendly measure of error between the distribution and the label
    for Normal distributions, this is the MSE, while for discrete distributions, this yields the 0-1 error

"""
from abc import abstractmethod
from typing import Tuple
import torch
import numpy as np
from torch import distributions
from torch import nn

class DistributionLayer(nn.Module):
    """
    A layer mapping network output to a distribution object    
    """
    @abstractmethod
    def forward(self, z : torch.Tensor) -> distributions.Distribution:
        """
        Returns torch.Distribution object specified by z

        Args:
            z (torch.Tensor): parameter of output distribution
        """
        raise NotImplementedError

    @abstractmethod
    def marginalize_gaussian(self, z_mean : torch.Tensor, z_var : torch.Tensor) -> distributions.Distribution:
        """
        If z \sim N(z_mean, z_var), estimates p(y) = E_z [ p(y \mid z) ]

        Args:
            z_mean (torch.Tensor): mean of z
            z_var (torch.Tensor): diagonal variance of z (same size as z_mean)

        Returns:
            distributions.Distribution: p(y)
        """
        raise NotImplementedError

    @abstractmethod
    def marginalize_samples(self, z_samples : torch.Tensor, batch_idx : int = 0) -> distributions.Distribution:
        """
        Given samples of z, estimates p(y) = E_z [ p(y \mid z) ] over empirical distribution

        Args:
            z_samples (torch.Tensor): [..., M, ..., z_dim]
            batch_idx (int): index over which to marginalize, assumed to be 0

        Returns:
            distributions.Distribution: p(y)
        """
        raise NotImplementedError

    @abstractmethod
    def apply_sqrt_F(self, z : torch.Tensor) -> torch.Tensor:
        """
        The Fisher Information Matrix of the output distribution is given by
            $$ F = E_{y \sim p(y | z)}[ d^2/dz^2 \log p (y \mid z)] $$
        If we factor F(z) = L(z) L(z)^T
        This function returns [ L(z)^T ].detach() @ z 

        Args:
            z (torch.Tensor): parameter of p(y | z)

        Returns:
            torch.Tensor: parameter scaled by the square root of the fisher matrix, [ L(z)^T ].detach() z 
        """
        raise NotImplementedError

    @abstractmethod
    def metric(self, z : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
        """
        Returns a metric of Error(y,z), e.g. MSE for regression, 0-1 error for classification

        Args:
            z (torch.Tensor): parameter of distribution
            y (torch.Tensor): target

        Returns:
            torch.Tensor: Error(y, z)
        """
        raise NotImplementedError

    def log_prob(self, z : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
        """
        Returns log p( y | z)

        Args:
            z (torch.Tensor): parameter of distribution
            y (torch.Tensor): target

        Returns:
            torch.Tensor: log p ( labels | dist )
        """
        return self.forward(z).log_prob(y)

    def validated_log_prob(self, z : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
        """
        Checks if each element of y is in the support of the distribution specified by z
        Computes mean log_prob only on the elements 

        Args:
            z (torch.Tensor): parameter of distribution
            y (torch.Tensor): target

        Returns:
            torch.Tensor: log p ( valid_y | corresponding_z )
        """
        dist = self.forward(z)
        valid_idx = dist.support.check(y)
        # raise error if there are no valid datapoints in the batch
        assert torch.sum(valid_idx) > 0

        # construct dist keeping only valid slice
        valid_y = y[valid_idx]
        valid_z = z[valid_idx]
        return self.forward(valid_z).log_prob(valid_y)

class BernoulliLogitsLayer(DistributionLayer):
    """
    Implements Bernoulli RV parameterized by logits.
    """
    def forward(self, z: torch.Tensor) -> distributions.Distribution:
        return distributions.Bernoulli(logits = z)
    
    def marginalize_gaussian(self, z_mean: torch.Tensor, z_var: torch.Tensor) -> distributions.Distribution:
        kappa = 1.0 / torch.sqrt(1. + np.pi / 8 * z_var)
        return distributions.Bernoulli(logits=kappa*z_mean)

    def marginalize_samples(self, z_samples: torch.Tensor, batch_idx : int = 0) -> distributions.Distribution:
        probs = torch.sigmoid(z_samples).mean(dim=batch_idx)
        return distributions.Bernoulli(probs=probs)

    def apply_sqrt_F(self, z: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(z)
        L = torch.sqrt(p * (1 - p)) + 1e-8 # for stability
        return L * z

    def metric(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes 0-1 classification error
        """
        return ( (self.z >= 0) != y ).float()
    
class NormalMeanParamLayer(DistributionLayer):
    """
    Implements Normal RV parameterized by the mean. The variance is a parameter of the layer.
    """
    def __init__(self, init_log_variance : torch.Tensor = torch.zeros(1)) -> None:
        super().__init__()
        self.log_variance = nn.Parameter(init_log_variance)
    
    @property
    def std_dev(self) -> torch.Tensor:
        return torch.exp(0.5*self.log_variance)

    @property
    def var(self) -> torch.Tensor:
        return torch.exp(self.log_variance)

    def forward(self, z: torch.Tensor) -> distributions.Distribution:
        return distributions.Independent(
                    distributions.Normal(
                        loc = z, 
                        scale = self.std_dev.broadcast_to(z.size())
                        ),
                    reinterpreted_batch_ndims=1
                )
    
    def marginalize_gaussian(self, z_mean: torch.Tensor, z_var: torch.Tensor) -> distributions.Distribution:
        combined_std_dev = torch.sqrt(self.std_dev**2 + z_var)
        return distributions.Independent(
                distributions.Normal(loc=z_mean, scale=combined_std_dev), 
                reinterpreted_batch_ndims=1
            )

    def marginalize_samples(self, z_samples: torch.Tensor, batch_idx: int = 0) -> distributions.Distribution:
        combined_mean = z_samples.mean(batch_idx)
        combined_std_dev = torch.sqrt(
            (z_samples ** 2).mean(batch_idx)
            - combined_mean ** 2
            + self.var
        )
        return distributions.Independent(
                distributions.Normal(loc=combined_mean, scale=combined_std_dev), 
                reinterpreted_batch_ndims=1
            )
    
    def apply_sqrt_F(self, z: torch.Tensor) -> torch.Tensor:
        return z / self.std_dev

    def metric(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return ((y - z)**2).sum(-1)


class NormalMeanDiagVarParamLayer(DistributionLayer):
    """
    Implements Normal RV where both mean and log var are input as parameters
    Assumes first half of z is mean, second half is log_var
    Only performs Fisher computation around mean.
    """
    def _get_mean_logvar(self, z : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N = z.shape[-1]
        assert N % 2 == 0
        return z[...,:N//2], z[...,N//2:]

    def forward(self, z: torch.Tensor) -> distributions.Distribution:
        mean, logvar = self._get_mean_logvar(z)
        return distributions.Independent(
                    distributions.Normal(
                        loc = mean, 
                        scale = torch.exp(0.5*logvar)
                        ),
                    reinterpreted_batch_ndims=1
                )
    
    def marginalize_gaussian(self, z_mean: torch.Tensor, z_var: torch.Tensor) -> distributions.Distribution:
        mean_mean, logvar_mean = self._get_mean_logvar(z_mean)
        mean_var, logvar_var = self._get_mean_logvar(z_var)

        combined_std_dev = torch.sqrt(torch.exp(logvar_mean) + mean_var)
        return distributions.Independent(
                distributions.Normal(loc=mean_mean, scale=combined_std_dev), 
                reinterpreted_batch_ndims=1
            )

    def marginalize_samples(self, z_samples: torch.Tensor, batch_idx: int = 0) -> distributions.Distribution:
        mean_samples, logvar_samples = self._get_mean_logvar(z_samples)
        var_mean = torch.exp(logvar_samples).mean(batch_idx)
        combined_mean = mean_samples.mean(batch_idx)
        combined_std_dev = torch.sqrt(
            (z_samples ** 2).mean(batch_idx)
            - combined_mean ** 2
            + var_mean
        )
        return distributions.Independent(
                distributions.Normal(loc=combined_mean, scale=combined_std_dev), 
                reinterpreted_batch_ndims=1
            )
    
    def apply_sqrt_F(self, z: torch.Tensor) -> torch.Tensor:
        mean, logvar = self._get_mean_logvar(z)
        return mean / torch.exp(0.5*logvar).detach()

    def metric(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mean, logvar = self._get_mean_logvar(z)
        return ((y - mean)**2).sum(-1)


class CategoricalLogitLayer(DistributionLayer):
    """
    Implements Categorical distribution parameterized by logits
    """
    def forward(self, z: torch.Tensor) -> distributions.Distribution:
        return distributions.Categorical(logits = z)
    
    def marginalize_gaussian(self, z_mean: torch.Tensor, z_var: torch.Tensor) -> distributions.Distribution:
        kappa = 1.0 / torch.sqrt(1. + np.pi / 8 * z_var)
        return distributions.Categorical(logits=kappa*z_mean)

    def marginalize_samples(self, z_samples: torch.Tensor, batch_idx : int = 0) -> distributions.Distribution:
        probs = torch.softmax(z_samples, -1).mean(dim=batch_idx)
        return distributions.Categorical(probs=probs)

    def apply_sqrt_F(self, z: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(z, -1).detach()
        z_bar = (p * z).sum(-1, keepdim=True)
        return torch.sqrt(p) * ( z - z_bar )

    def metric(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes 0-1 classification error
        """
        return (torch.argmax(z, dim=-1) != y).float()

class ExtendedDistribution:
    """
    AbstractBaseClass
    """

    @abstractmethod
    def apply_sqrt_F(self, vec):
        """
        if the Fisher matrix in terms of self._param is LL^T,
        return L^T vec, blocking gradients through L

        Args:
            vec (torch.Tensor): vector

        Returns:
            L^T vec (torch.Tensor)
        """
        raise NotImplementedError

    @abstractmethod
    def marginalize(self, diag_var):
        """
        returns an approximation to the marginal distribution if the parameter
        used to initialize this distribution was distributed according to a Gaussian
        with mean and a diagonal variance as given

        inputs:
            diag_var: variance of parameter (1,)
        """
        raise NotImplementedError

    @abstractmethod
    def merge_batch(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        raise NotImplementedError

    @abstractmethod
    def metric(self, y):
        """
        Computes an alternative metric for an obervation under this distribution,
        e.g., MSE for Normal distribution, or 0-1 error for categorical and bernoulli distributions.

        Returns:
            torch.Tensor:
        """
        raise NotImplementedError

    @abstractmethod
    def validated_log_prob(self, labels):
        """Returns log prob after throwing out labels which are outside
        the support of the distribution.

        Args:
            labels (torch.Tensor): observations of the distribution

        Returns:
            torch.Tensor: log p ( labels | dist )
        """
        raise NotImplementedError


class Bernoulli(distributions.Bernoulli, ExtendedDistribution):
    """
    Implements Bernoulli RV specified either through a logit or a probability directly.
    """

    def __init__(self, probs=None, logits=None, validate_args=None):
        self.use_logits = False
        if probs is None:
            self.use_logits = True
        super().__init__(probs=probs, logits=logits, validate_args=validate_args)

    def apply_sqrt_F(self, vec):
        """
        if the Fisher matrix in terms of self._param is LL^T,
        return L^T vec, blocking gradients through L

        if self._param is probs, then
            F = 1/(p(1-p))
        if self._param is logits, then
            F = p(1-p), where p = sigmoid(logit)
        """
        p = self.probs.detach()
        L = torch.sqrt(p * (1 - p)) + 1e-10  # for stability

        if self.use_logits:
            return L * vec
        else:
            return vec / L

    def marginalize(self, diag_var):
        """
        returns an approximation to the marginal distribution if the parameter
        used to initialize this distribution was distributed according to a Gaussian
        with mean and a diagonal variance as given

        inputs:
            diag_var: variance of parameter (1,)
        """
        if self.use_logits:
            kappa = 1.0 / torch.sqrt(1.0 + np.pi / 8 * diag_var)
            p = torch.sigmoid(kappa * self.logits)
        else:
            p = self.probs  # gaussian posterior in probability space is not useful
        return Bernoulli(probs=p)

    def merge_batch(self):
        p_mean = self.probs.mean(dim=0)
        return Bernoulli(probs=p_mean)

    def metric(self, y):
        """
        classification error (1- accuracy)
        """
        return ((self.probs >= 0.5) != y).float()

    def validated_log_prob(self, labels):
        """
        computes log prob, ignoring contribution from labels which are out of the support
        """
        valid_idx = self.support.check(labels)
        # raise error if there are no valid datapoints in the batch
        assert torch.sum(valid_idx) > 0

        # construct dist keeping only valid slice
        labels_valid = labels[valid_idx]
        if self.use_logits:
            logits_valid = self.logits[valid_idx]
            return Bernoulli(logits=logits_valid).log_prob(labels_valid)
        else:
            probs_valid = self.probs[valid_idx]
            return Bernoulli(probs=probs_valid).log_prob(labels_valid)


class Normal(distributions.normal.Normal, ExtendedDistribution):
    """
    Implemetns a Normal distribution specified by mean and standard deviation
    Only supports diagonal variance matrices
    """

    def __init__(self, loc, scale, validate_args=None):
        """Creates Normal distribution object

        Args:
            loc (_type_): mean of normal distribution
            scale (_type_): standard deviation of normal distribution
        """
        super().__init__(loc, scale, validate_args=validate_args)

    def apply_sqrt_F(self, vec):
        """
        if the Fisher matrix in terms of self._param is LL^T,
        return L^T vec, blocking gradients through L

        Here, we assume only loc varies, and do not consider cov as a backpropable parameter

        F = Sigma^{-1}

        """
        return vec / self.stddev.detach()

    def marginalize(self, diag_var):
        """
        returns an approximation to the marginal distribution if the parameter
        used to initialize this distribution was distributed according to a Gaussian
        with a diagonal variance as given

        inputs:
            diag_var: variance of parameter (d,)
        """
        stdev = torch.sqrt(self.variance + diag_var)
        return Normal(loc=self.mean, scale=stdev)

    def merge_batch(self):
        diag_var = (
            torch.mean(self.mean**2, dim=0)
            - self.mean.mean(dim=0) ** 2
            + self.variance.mean(dim=0)
        )
        return Normal(loc=self.mean.mean(dim=0), scale=torch.sqrt(diag_var))

    def metric(self, y):
        return torch.mean(torch.sum((y - self.mean) ** 2, dim=-1))

    def validated_log_prob(self, labels):
        # all labels are in the support of a Normal distribution
        return self.log_prob(labels)


class Categorical(distributions.categorical.Categorical, ExtendedDistribution):
    """
    Implements Categorical distribution specified by either probabilities directly or logits.
    """

    def __init__(self, probs=None, logits=None, validate_args=None):
        self.use_logits = False
        if probs is None:
            self.use_logits = True
        super().__init__(probs=probs, logits=logits, validate_args=validate_args)

    def apply_sqrt_F(self, vec):
        """
        if the Fisher matrix in terms of self._param is LL^T,
        return L^T vec, blocking gradients through L

        if self._param is probs, then
            F = (diag(p^{-1}))
        if self._param is logits, then
            F = (diag(p) - pp^T) = LL^T
        """
        p = self.probs.detach()
        if self.use_logits:
            vec_bar = torch.sum(p * vec, dim=-1, keepdim=True)
            return torch.sqrt(p) * (vec - vec_bar)
        else:
            return vec / (torch.sqrt(p) + 1e-8)

    def marginalize(self, diag_var):
        """
        returns an approximation to the marginal distribution if the parameter
        used to initialize this distribution was distributed according to a Gaussian
        with a diagonal variance as given

        inputs:
            diag_var: variance of parameter (d,)
        """
        if self.use_logits:
            # TODO: allow selecting this via an argument
            # probit approx
            kappa = 1.0 / torch.sqrt(1.0 + np.pi / 8 * diag_var)
            scaled_logits = kappa * self.logits
            dist = Categorical(logits=scaled_logits)

            # laplace bridge
            # d = diag_var.shape[-1]
            # sum_exp = torch.sum(torch.exp(-self.logits), dim=-1, keepdim=True)
            # alpha = 1. / diag_var * (1 - 2./d + torch.exp(self.logits)/(d**2) * sum_exp)
            # dist = distributions.Dirichlet(alpha)
            # return distributions.Categorical(probs=torch.nan_to_num(dist.mean, nan=1.0))
        else:
            p = self.probs  # gaussian posterior in probability space is not useful
            return Categorical(probs=p)
        return dist

    def validated_log_prob(self, labels):
        """
        computes log prob, ignoring contribution from labels which are out of the support
        """
        valid_idx = self.support().check(labels)
        # raise error if there are no valid datapoints in the batch
        if torch.sum(valid_idx) == 0:
            return torch.zeros_like(valid_idx).float()

        # construct dist keeping only valid slice
        labels_valid = labels[valid_idx]
        if self.use_logits:
            logits_valid = self.logits[valid_idx, ...]
            return Categorical(logits=logits_valid).log_prob(labels_valid)
        else:
            probs_valid = self.probs[valid_idx, ...]
            return Categorical(probs=probs_valid).log_prob(labels_valid)

    def merge_batch(self):
        p_mean = self.probs.mean(dim=0)
        return Categorical(probs=p_mean)

    def metric(self, y):
        """
        classification error (1- accuracy)
        """
        return (torch.argmax(self.probs, dim=-1) != y).float()
