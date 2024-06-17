"""
prior_optim.py

Functions for tuning the prior scale of a scod model (either before or after training)
"""
import torch
from typing import Optional, Callable
from tqdm.autonotebook import tqdm
from .distributions import DistributionLayer
from .scod import SCOD
from .utils import gaussian_kl_div, gaussian_wasserstein_dist

def optimize_prior_scale_by_nll(
    scod_model : SCOD,
    dataset: torch.utils.data.Dataset,
    dist_layer: DistributionLayer,
    dataloader_kwargs: Optional[dict] = None,
    num_epochs: int = 2,
):
    """
    tunes prior variance scale (eps) via SGD to minimize
    validation nll on a given dataset
    """
    device = next(scod_model.model.parameters()).device

    scod_model.GGN_is_aligned = False
    if dataloader_kwargs is None:
        dataloader_kwargs = {
            "num_workers": 4,
            "batch_size": 20,
            "shuffle": True,
        }
        if device.type != "cpu":
            dataloader_kwargs["pin_memory"] = True

    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    dataset_size = len(dataset)
    optimizer = torch.optim.Adam(scod_model.hyperparameters, lr=1e-1)

    losses = []

    with tqdm(total=num_epochs, position=0) as pbar:
        pbar2 = tqdm(total=dataset_size, position=1)
        for epoch in range(num_epochs):
            pbar2.refresh()
            pbar2.reset(total=dataset_size)
            for inputs, labels in dataloader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                z_mean, z_var = scod_model.forward(inputs)
                dist = dist_layer.marginalize_gaussian(z_mean, z_var)
                loss = dist.log_prob(labels)

                loss.backward()
                optimizer.step()

                pbar2.set_postfix(
                    batch_loss=loss.item(), eps=scod_model.sqrt_prior.mean().item()
                )
                pbar2.update(inputs.shape[0])

                losses.append(loss.item())

            pbar.set_postfix(eps=scod_model.sqrt_prior.mean().item())
            pbar.update(1)

    return losses

def optimize_prior_scale_by_GP_kernel(
    scod_model : SCOD,
    dataset: torch.utils.data.Dataset,
    GP_kernel: Callable[[torch.Tensor], torch.Tensor],
    GP_mu: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    dataloader_kwargs: Optional[dict] = None,
    num_epochs: int = 20,
    grad_accumulation_steps: int = 5,
    dist_loss: str = "wass",
):
    """
    tunes prior variance scale (eps) via SGD to minimize
    difference between prior and a given GP
    GP_kernel should take in a batch of data and produce a gram matrix
    """
    device = next(scod_model.model.parameters()).device
    scod_model.GGN_is_aligned = False
    if dataloader_kwargs is None:
        dataloader_kwargs = {
            "num_workers": 4,
            "batch_size": 20,
            "shuffle": True,
        }
        if device.type != "cpu":
            dataloader_kwargs["pin_memory"] = True

    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    dataset_size = len(dataset)
    optimizer = torch.optim.Adam(scod_model.hyperparameters, lr=1e-1)

    losses = []
    min_eigs = []

    grad_counter = 0
    with tqdm(total=num_epochs, position=0) as pbar:
        for epoch in range(num_epochs):
            for inputs, labels in dataloader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                K_dnn = scod_model._kernel_matrix(inputs, prior_only=True)
                min_eig = torch.min(torch.abs(torch.linalg.eigvals(K_dnn)))

                K_gp = GP_kernel(inputs).to(device, non_blocking=True)
                err = None
                if GP_mu is not None:
                    mu_dnn = scod_model.model(inputs)
                    mu_gp = GP_mu(inputs)
                    err = (mu_dnn - mu_gp).view(-1)

                if dist_loss == "fwd_kl":
                    loss = gaussian_kl_div(K_gp, K_dnn, err)
                elif dist_loss == "rev_kl":
                    loss = gaussian_kl_div(K_dnn, K_gp, err)
                else:
                    loss = gaussian_wasserstein_dist(K_gp, K_dnn, err)

                loss = loss / grad_accumulation_steps

                loss.backward()

                grad_counter += 1

                if grad_counter % grad_accumulation_steps == 0:
                    # clip gradients
                    torch.nn.utils.clip_grad_norm_(scod_model.hyperparameters, 5.0)
                    # after grad_accumulation steps have passed, then take the gradient step
                    optimizer.step()
                    # zero the parameter gradients
                    optimizer.zero_grad()

                losses.append(loss.item())
                min_eigs.append(min_eig.item())

            pbar.set_postfix(eps=scod_model.sqrt_prior.mean().item())
            pbar.update(1)

    return losses, min_eigs