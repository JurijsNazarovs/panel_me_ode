import numpy as np
import torch
import torch.nn as nn

import lib.utils as utils
from lib.utils import get_device

from torch.distributions.normal import Normal
from torch.distributions import Independent


def gaussian_log_likelihood(mu_2d, data_2d, obsrv_std, indices=None):
    n_data_points = mu_2d.size()[-1]

    if n_data_points > 0:
        gaussian = Independent(
            Normal(loc=mu_2d, scale=obsrv_std.repeat(n_data_points)), 1)
        log_prob = gaussian.log_prob(data_2d)
        log_prob = log_prob / n_data_points
    else:
        log_prob = torch.zeros([1]).to(get_device(data_2d)).squeeze()
    return log_prob


def compute_masked_likelihood(mu, data, mask, obsrv_std=1.):
    loss = nn.GaussianNLLLoss(full=True, eps=1e-06, reduction='none')
    res = -loss(data, mu, 0 * data + obsrv_std**2)

    # gaussian = Independent(Normal(loc=mu, scale=0 * data + obsrv_std**2), 1)
    # res = gaussian.log_prob(data, reduction='none')
    res *= mask
    res = res.sum(axis=-2)  # , keepdim = True)  # sum over time
    res = res / mask.sum(axis=-2)  # , keepdim = True)
    res[torch.isnan(res)] = 0
    res = res.mean(axis=-1)
    res = res.transpose(0, 1)
    return res


def masked_gaussian_log_density(mu, data, obsrv_std, mask=None):
    n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()
    assert (data.size()[-1] == n_dims)

    # Shape after permutation: [n_traj, n_traj_samples, n_timepoints, n_dims]
    if mask is None:
        mu_flat = mu.reshape(n_traj_samples * n_traj, n_timepoints * n_dims)
        n_traj_samples, n_traj, n_timepoints, n_dims = data.size()
        data_flat = data.reshape(n_traj_samples * n_traj,
                                 n_timepoints * n_dims)

        res = gaussian_log_likelihood(mu_flat, data_flat, obsrv_std)
        res = res.reshape(n_traj_samples, n_traj).transpose(0, 1)
    else:
        res = compute_masked_likelihood(mu, data, mask, obsrv_std)
    return res


def compute_mse(mu, data, mask=None):
    n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()
    assert (data.size()[-1] == n_dims)

    # Shape after permutation: [n_traj, n_traj_samples, n_timepoints, n_dims]
    if mask is None:
        mu_flat = mu.reshape(n_traj_samples * n_traj, n_timepoints * n_dims)
        n_traj_samples, n_traj, n_timepoints, n_dims = data.size()
        data_flat = data.reshape(n_traj_samples * n_traj,
                                 n_timepoints * n_dims)
        res = nn.MSELoss()(mu_flat, data_flat)
    else:
        # We put sqrt(1/2) to compute only (mu-data)**2
        # then we need to add log(sqrt(pi))
        res = -compute_masked_likelihood(
            mu, data, mask, obsrv_std=np.sqrt(1 / 2))
        res += torch.Tensort(-np.log(np.pi**0.5)).to(res)

    return res
