"""
SCOD: Sketching Curvature for OOD Detection
"""
from typing import Optional, Tuple, List
from copy import deepcopy, copy
import math

import torch
from torch import nn
from torch.autograd import grad
from torch.cuda.amp.autocast_mode import autocast

from tqdm.autonotebook import tqdm

from .sketching.sketched_pca import alg_registry
from .sketching.utils import random_subslice
from .distributions import DistributionLayer


base_config = {
    "num_samples": None,  # sketch size T (T)
    "num_eigs": 10,  # low rank estimate to recover (k)
    "prior_type": "scalar",  # options are 'scalar' (isotropic prior), 'per_parameter', 'per_weight'
    "sketch_type": "gaussian",  # sketch type
    "offline_proj_dim": None,  # whether to subsample rows during offline computation
    "online_proj_dim": None,  # whether to project output down before taking gradients at test time
    "online_proj_type": "gaussian",  # how to do output projection
    "metric_threshold": None, # if not None, expects a float indicating whether to ignore a training point
}


class SCOD(nn.Module):
    """
    Wraps a trained model with functionality for adding epistemic uncertainty estimation.
    """

    def __init__(
        self,
        model: nn.Module,
        args: Optional[dict] = None,
        parameters: Optional[nn.ParameterList] = None,
    ) -> None:
        """Initializes SCOD module as a wrapper around an existing, pre-trained DNN.

        Args:
            model (nn.Module): Pre-trained DNN
            args (dict, optional):
                Configuration parameters for SCOD. If None, uses default settings. Defaults to None.
                Valid parameters, and their defaults, are given below:
                    "num_eigs": 10,  # low rank estimate to recover (k)
                    "num_samples": None,  # sketch size T (T), if None, uses 
                    "prior_type": "scalar",  # options are 'scalar' (isotropic prior), 'per_parameter', 'per_weight'
                    "sketch_type": "gaussian",  # sketch type
                    "offline_proj_dim": None,  # whether to subsample rows during offline computation
                    "online_proj_dim": None,  # whether to project output down before taking gradients at test time
                    "online_proj_type": "gaussian",  # how to do output projection
                    "metric_threshold": None, # if not None, expects a float indicating whether to ignore a training point

            parameters (nn.ParameterList, optional):
                The parameters of model to consider when computing approximate Bayesian posterior.
                If None, uses all model.parameters() for which requires_grad is True.  Defaults to None.
        """
        super().__init__()

        self.config = deepcopy(base_config)
        if args is not None:
            self.config.update(args)

        self.model = model

        # extract device from model
        device = next(model.parameters()).device

        # extract parameters to consider in sketch - keep all that yield valid gradients
        if parameters is None:
            self.trainable_params = list(
                filter(lambda x: x.requires_grad, self.model.parameters())
            )
        else:
            self.trainable_params = list(parameters)
        self.n_weights_per_param = list(p.numel() for p in self.trainable_params)
        self.n_weights = int(sum(self.n_weights_per_param))
        print("Weight space dimension: %1.3e" % self.n_weights)

        self.num_samples = self.config["num_samples"]
        self.num_eigs = self.config["num_eigs"]

        if self.num_samples is None:
            self.num_samples = 6 * self.num_eigs + 4

        self.configured = nn.Parameter(
            torch.zeros(1, dtype=torch.bool), requires_grad=False
        )

        # ---------------------------
        # options for prior representation and computation
        self.prior_type = self.config["prior_type"]
        self.log_prior_scale = nn.Parameter(self._init_log_prior(device), requires_grad=True)

        # --------------------------------
        # options for offline computation
        self.offline_projection_dim = T = self.config["offline_proj_dim"]

        # Empirical Fisher:
        # Whether we should just use empirical fisher, i.e. outer products
        # of gradients of the negative log prob
        self.use_empirical_fisher = T is not None and T == 0

        # Random proj:
        # Rather than compute the whole Fisher, instead, randomly subsample rows of the jacobian
        # This subsampling isn't performed if T is equal or larger than the output dimension
        self.use_random_proj = T is not None and T > 0

        # Approximation alg:
        # Determines how to aggregate per-datapoint information into final low-rank GGN approx
        self.sketch_class = alg_registry[self.config["sketch_type"]]

        # Points to consider:
        # If not None, then SCOD ignores points where metric > threshold
        self.metric_threshold = self.config["metric_threshold"]

        # --------------------------------
        # options for online computation
        self.online_projection_dim = self.config["online_proj_dim"]
        self.online_projection_type = self.config["online_proj_type"]

        # --------------------------------
        # Parameters to be saved as part of self.state_dict() for easy reloading:
        # Low-rank approx of Gauss Newton, filled in by self.process_dataset
        self.GGN_eigs = nn.Parameter(
            torch.zeros(self.num_eigs, device=device), requires_grad=False
        )

        self.GGN_basis = nn.Parameter(
            torch.zeros(self.n_weights, self.num_eigs, device=device),
            requires_grad=False,
        )

        # stores what prior scales were used in GGN computation
        self.GGN_sqrt_prior = nn.Parameter(copy(self.sqrt_prior), requires_grad=False)
        self.GGN_is_aligned = True

        self.hyperparameters = [self.log_prior_scale]

    def _init_log_prior(self, device : torch.DeviceObjType) -> torch.Tensor:
        """Returns intial value of log_prior parameter, depending on self.prior_type

        Raises:
            ValueError: If self.prior_type is not an understood value.

        Returns:
            torch.Tensor: torch.zeros() of the correct shape
        """
        if self.prior_type == "per_parameter":
            n_prior_params = len(self.trainable_params)
        elif self.prior_type == "per_weight":
            n_prior_params = self.n_weights
        elif self.prior_type == "scalar":
            n_prior_params = 1
        else:
            raise ValueError(
                "prior_type must be one of (scalar, per_parameter, per_weight)"
            )
        return torch.zeros(n_prior_params, device=device)

    def _broadcast_to_n_weights(self, v: torch.Tensor) -> torch.Tensor:
        """Broadcasts a vector to be the length of self.n_weights
        The vector must be one of the following lengths:
          - 1 -> broadcasts as usual
          - len(self.trainable_parameters) -> each element is repeated
                for the number of weights in each parameter
          - n_weights -> returned as is

        Args:
            v (torch.Tensor): vector to be expanded

        Raises:
            ValueError: v is not of a compatible length

        Returns:
            torch.Tensor: expanded view into v
        """
        assert len(v.shape) == 1
        k = v.shape[0]
        if k == 1:
            return v.expand(self.n_weights)

        if k == len(self.trainable_params):
            return torch.cat(
                [vi.expand(n) for vi, n in zip(v, self.n_weights_per_param)]
            )

        if k == self.n_weights:
            return v

        raise ValueError(
            "Input vector is not compatible with self.n_weights or self.trainable_params"
        )

    @property
    def sqrt_prior(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: Learned elements of diagonal part of Sigma_0^{1/2}
        """
        return torch.exp(0.5 * self.log_prior_scale)

    def _get_weight_jacobian(
        self, vec: torch.Tensor, scaled_by_prior: bool = True
    ) -> torch.Tensor:
        """Returns a k x n_param matrix, with each row being d(vec[i])/d(weights) for i = 1, .., k
        (Currently just loops over dimensionality of vec, as batch jacobians aren't supported in pytorch)

        if scaled_by_prior is True, then scales jacobian by self.sqrt_prior before returning

        Args:
            vec (torch.Tensor): function output to differentiate via backprop
            scaled_by_prior (bool, optional): whether to scale jacobian by prior. Defaults to True.

        Returns:
            torch.Tensor: J = df/dweights
        """
        assert len(vec.shape) == 1
        grad_vecs = []
        for j in range(vec.shape[0]):
            grads = grad(
                vec[j],
                self.trainable_params,
                retain_graph=True,
                only_inputs=True,
                allow_unused=True,
            )
            g = torch.cat([gr.contiguous().view(-1) for gr in grads]).detach()
            if scaled_by_prior:
                g = g * self._broadcast_to_n_weights(self.sqrt_prior)
            grad_vecs.append(g)

        return torch.stack(grad_vecs)

    def _predictive_var(
        self, J: torch.Tensor, n_eigs: Optional[int] = None, prior_only: bool = False
    ):
        r"""
        Returns the predictive covariance matrix
            Sigma_z = J ( I + G )^{-1} J.T  (shape: d x d)

        where J are scaled Jacobian matrices and G is the approximated Gauss Newton matrix
        assumes J = [d x n_weights]

        n_eigs dictates how many eigenvalues of G to use, by default equal to self.num_eigs

        prior_only: if True, ignores G, returning the prior predictive variance

        ---

        Interally, G is represented as U diag(eigs) U.T
        Thus, by the Woodbury identity, (I + G)^{-1} is equal to
            I - U diag( eigs / (eigs + 1) ) U.T

        Assumes J is the jacobian already scaled by the sqrt prior covariance.

        Returns diag( J ( I - U D U^T ) J^T )
        where D = diag( eigs / (eigs + 1))

        and U diag(eigs) U^T ~= G, the Gauss Newton matrix computed using gradients
        already scaled by the sqrt prior covariance

        that is, we let J = (df/dw) \Sigma_0^{1/2}
        and G = sum( J_i^T H_i J_i )
        """
        JJT = J @ J.T
        if not self.configured or prior_only:
            return JJT

        if n_eigs is None:
            n_eigs = self.num_eigs

        # pylint: disable=invalid-unary-operand-type
        basis = self.GGN_basis[:, -n_eigs:]
        eigs = torch.clamp(self.GGN_eigs[-n_eigs:], min=0.0)

        if self.GGN_is_aligned:
            scaling = eigs / (eigs + 1.0)
            neg_term = J @ basis
            neg_term = neg_term @ (scaling[:, None] * neg_term.T)
        else:
            # G = rescaling*U D (rescaling*U).T
            # the rescaling breaks the orthogonality of the basis, need to do matrix inversion
            rescaling = self._broadcast_to_n_weights(
                self.sqrt_prior / self.GGN_sqrt_prior
            )
            basis = rescaling[:, None] * basis
            inv_term = torch.linalg.inv(torch.diag_embed(1.0 / eigs) + basis.T @ basis)
            scaled_jac = J @ basis
            neg_term = scaled_jac @ inv_term @ scaled_jac.T

        return JJT - neg_term

    def _kernel_matrix(self, inputs, prior_only=True, n_eigs=None):
        """
        returns a kernel (gram) matrix for the set of inputs

        returns a matrix of size: [KD x KD] where inputs is [X, d]
        """
        N = inputs.shape[0]  # batch size

        z = self.model(inputs)  # batch of outputs
        flat_z = z.view(N, -1)  # batch of flattened outputs
        jacs = []
        for j in range(N):
            J = self._get_weight_jacobian(flat_z[j, :], scaled_by_prior=True)
            jacs.append(J)

        jacs = torch.cat(jacs, dim=0)  # (KD x N)
        return self._predictive_var(
            jacs, n_eigs=n_eigs, prior_only=prior_only
        )  # (KD x KD)

    def _output_projection(self, output_size, T, device):
        if self.online_projection_type == "PCA":
            return self.online_proj_basis[:, -T:].detach()
        elif self.online_projection_type == "blocked":
            P = torch.eq(
                torch.floor(
                    torch.arange(output_size, device=device)[:, None]
                    / (output_size // T)
                ),
                torch.arange(T, device=device)[None, :],
            ).float()
            P /= torch.norm(P, dim=0, keepdim=True)
            return P
        else:
            P = torch.randn(output_size, T, device=device)
            P, _ = torch.linalg.qr(P, "reduced")
            return P

    def process_dataset(
        self,
        dataset: torch.utils.data.Dataset,
        dist_layer: DistributionLayer,
        dataloader_kwargs: Optional[dict] = None,
    ) -> None:
        """
        summarizes information about training data by logging gradient directions
        seen during training, and then using gram schmidt of these to form
        an orthonormal basis. directions not seen during training are
        taken to be irrelevant to data, and used for detecting generalization

        dataset - torch dataset of (input, target) pairs
        dist_layer: DistributionLayer mapping output of model to a distribution on y
        """
        # infer device from model:
        device = next(self.model.parameters()).device

        # loop through data, one sample at a time
        print("computing basis")
        if dataloader_kwargs is None:
            dataloader_kwargs = {
                "num_workers": 4,
            }
            if device.type != "cpu":
                dataloader_kwargs["pin_memory"] = True
        dataloader_kwargs.update({"batch_size": 1})

        dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

        sketch = self.sketch_class(
            N=self.n_weights, r=self.num_eigs, T=self.num_samples, device=device
        )

        n_data = len(dataloader)
        for i, (inputs, labels) in tqdm(enumerate(dataloader), total=n_data):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                z = self.model(inputs)
                if self.metric_threshold is not None:
                    metric = dist_layer.metric(z, labels).mean()
                    if metric > self.metric_threshold:
                        continue

                if self.use_empirical_fisher:
                    # contribution of this datapoint is
                    # C = J_l^T J_l, where J_l = d(-log p(y | x))/dw
                    pre_jac_factor = -dist_layer.validated_log_prob(z, labels)  # shape [1]
                else:
                    # contribution of this datapoint is
                    # C = J_f^T L L^T J
                    Lt_z = dist_layer.apply_sqrt_F(z).mean(dim=0)  # L^\T theta
                    # flatten
                    pre_jac_factor = Lt_z.view(-1)  # shape [prod(event_shape)]

                if self.use_random_proj:
                    pre_jac_factor = random_subslice(
                        pre_jac_factor, dim=0, k=self.offline_projection_dim, scale=True
                    )

            sqrt_C_T = self._get_weight_jacobian(
                pre_jac_factor, scaled_by_prior=True
            )  # shape ([T x N])

            sketch.low_rank_update(
                sqrt_C_T.t()
            )  # add C = sqrt_C sqrt_C^T to the sketch

        del sqrt_C_T  # free memory @TODO: sketch could take output tensors to populate directly
        eigs, eigvs = sketch.eigs()
        del sketch
        self.GGN_eigs.data = torch.clamp_min( eigs[-self.num_eigs:], min=torch.zeros(1)).to(device)
        self.GGN_basis.data = eigvs[:,-self.num_eigs:].to(device)
        self.GGN_sqrt_prior.data = copy(self.sqrt_prior)
        self.GGN_is_aligned = True

        self.configured.data = torch.ones(1, dtype=torch.bool)

    def local_kl_div(self, 
        inputs: torch.Tensor,
        dist_layer: DistributionLayer,
        use_prior: bool = False,
        n_eigs: Optional[int] = None,
        prior_multiplier: float = 1.,
    ) -> torch.Tensor:
        """
        Outputs E_{w \sim p(w | D)}[ KL(p(w), p(w^*)) ],
        the expected KL divergence between p(w) and p(w^*) when w is sampled from the posterior.

        Args:
            inputs (torch.Tensor): _description_
            use_prior (bool, optional): _description_. Defaults to False.
            n_eigs (Optional[int], optional): _description_. Defaults to None.
            prior_multiplier (float, optional): _description_. Defaults to 1..

        Returns:
            torch.Tensor: _description_
        """
        self.log_prior_scale.data += math.log(prior_multiplier)

        if n_eigs is None:
            n_eigs = self.num_eigs

        basis = self.GGN_basis[:, -n_eigs:]
        eigs = torch.clamp(self.GGN_eigs[-n_eigs:], min=0.0)

        scaling = torch.sqrt( eigs / (eigs + 1.0) )

        N = inputs.shape[0]  # batch size

        z_mean = self.model(inputs)  # batch of outputs
        Lt_z = dist_layer.apply_sqrt_F(z_mean)
        flat_Ltz = Lt_z.view(N, -1) # batch of flattened fisher scaled outputs        

        kl_divs = []
        for j in range(N):
            Lt_Jj = self._get_weight_jacobian(
                flat_Ltz[j, :], scaled_by_prior=True
            )
            
            pos_term = (Lt_Jj**2).sum()
            neg_term = ( ( (Lt_Jj @ basis) * scaling[None,:] )**2 ).sum()

            kl_divs.append( pos_term - neg_term )

        kl_divs = torch.stack(kl_divs)

        self.log_prior_scale.data -= math.log(prior_multiplier)

        return kl_divs

    def forward(
        self,
        inputs: torch.Tensor,
        use_prior: bool = False,
        n_eigs: Optional[int] = None,
        prior_multiplier: float = 1.,
        T: Optional[int] = None,
    ) -> Tuple[List[torch.distributions.Distribution], torch.Tensor]:
        """
        assumes inputs are of shape (N, input_dims...)
        where N is the batch dimension,
              input_dims... are the dimensions of a single input

        Returns
            z_mean : model(inputs)
            z_var : diagonal scod predictive variance for inputs
        """
        self.log_prior_scale.data += math.log(prior_multiplier)
        if T is None:
            T = self.online_projection_dim

        # skip computation if projection dim is 0
        if T is not None and T == 0:
            z_mean = self.model(inputs)
            z_var = torch.zeros_like(z_mean)
            return z_mean, z_var

        if n_eigs is None:
            n_eigs = self.num_eigs

        N = inputs.shape[0]  # batch size
        device = inputs.device # used to ensure compatibility of constructed tensors

        z_mean = self.model(inputs)  # batch of outputs
        flat_z = z_mean.view(N, -1)  # batch of flattened outputs
        flat_z_shape = flat_z.shape[-1]  # flattened output size

        # by default there is no projection matrix and proj_flat_z = flat_z
        P = None  # projection matrix
        proj_flat_z = flat_z
        if T is not None and T < flat_z.shape[-1]:
            # if projection dim is provided and less than original dim
            P = self._output_projection(flat_z_shape, T, device)
            proj_flat_z = flat_z @ P

        z_vars = []
        for j in range(N):
            J_proj_flat_zj = self._get_weight_jacobian(
                proj_flat_z[j, :], scaled_by_prior=True
            )
            proj_flat_zj_var = self._predictive_var(
                J_proj_flat_zj, n_eigs=n_eigs, prior_only=use_prior
            )
            if P is not None:
                chol_sig = torch.linalg.cholesky(proj_flat_zj_var)
                flat_zj_diag_var = torch.sum(
                    (P @ chol_sig) ** 2, dim=-1
                )  # ( P[:,None,:] @ ( (JJT - neg_term) @ P.T) )[:,0]
            else:
                flat_zj_diag_var = torch.diagonal(proj_flat_zj_var)

            zj_var = flat_zj_diag_var.view(z_mean[j, ...].shape)
            z_vars.append(zj_var)

        z_var = torch.stack(z_vars)

        self.log_prior_scale.data -= math.log(prior_multiplier)
        return z_mean, z_var


class OodDetector(nn.Module):
    def __init__(self, scod_model : SCOD, dist_layer : DistributionLayer, metric : str = "entropy"):
        super().__init__()
        self.scod_model = scod_model
        self.metric = metric
        self.dist_layer = dist_layer

    def _entropy_signal(self, x, orient_alt):
        z_mean, z_var = self.scod_model(x, orient_alt)
        dist = self.dist_layer.marginalize_gaussian(z_mean, z_var)
        return dist.entropy()

    def _local_kl_div(self, x):
        return self.scod_model.local_kl_div(x, self.dist_layer)

    def _trace_of_var(self, x):
        z_mean, z_var = self.scod_model(x)
        return z_var.view(z_var.shape[0], -1).sum(-1)

    def forward(self, x : torch.Tensor, orient_alt):
        if self.metric == "entropy":
            return self._entropy_signal(x, orient_alt)
        elif self.metric == "local_kl":
            return self._local_kl_div(x)
        elif self.metric == "var":
            return self._trace_of_var(x)