import numpy as np
import sklearn as sk
import numpy as np
#import gc
import torch
import torch.nn as nn
from torch.nn.functional import relu

import lib.utils as utils
from lib.utils import get_device
from lib.encoder_decoder import *
from lib.likelihood_eval import *

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence, Independent

from torch import distributions

#import importlib
#import lib.base_models
#importlib.reload(lib.base_models)
#from lib.base_models import VAE_Baseline

import sys
sys.path.append("/home/nazarovs/projects/panel_mixed_sde/torchsde")
import torchsde
import math


def _stable_division(a, b, epsilon=1e-7):
    b = torch.where(b.abs().detach() > epsilon, b,
                    torch.full_like(b, fill_value=epsilon) * b.sign())
    return a / b


class LatentSDE(torchsde.SDEIto):
    def __init__(self,
                 input_dim,
                 latent_dim,
                 encoder_z0,
                 decoder,
                 diffeq_solver,
                 z0_prior,
                 device,
                 obsrv_std=None,
                 use_binary_classif=False,
                 use_poisson_proc=False,
                 linear_classifier=False,
                 classif_per_tp=False,
                 n_labels=1,
                 train_classif_w_reconstr=False,
                 pre_encoder=None,
                 adjoint=True,
                 drift=None):

        # From SDE block
        super(LatentSDE, self).__init__(noise_type="diagonal")
        self.sdeint_fn = torchsde.sdeint_adjoint if adjoint else torchsde.sdeint
        self.drift = drift

        theta = 1.0
        mu = 0.0
        sigma = obsrv_std
        self.obsrv_std = obsrv_std
        logvar = math.log(sigma**2 / (2. * theta))

        # Prior drift.
        self.register_buffer("theta", torch.tensor([[theta]]))
        self.register_buffer("mu", torch.tensor([[mu]]))
        self.register_buffer("sigma", torch.tensor([[sigma]]))

        # p(y0).
        self.register_buffer("py0_mean", torch.tensor([[mu]]))
        self.register_buffer("py0_logvar", torch.tensor([[logvar]]))

        # Approximate posterior drift: Takes in 2 positional encodings and the state.
        self.net = nn.Sequential(nn.Linear(3, 200), nn.Tanh(),
                                 nn.Linear(200, 200), nn.Tanh(),
                                 nn.Linear(200, 1))
        # Initialization trick from Glow.
        self.net[-1].weight.data.fill_(0.)
        self.net[-1].bias.data.fill_(0.)

        # q(y0).
        self.qy0_mean = nn.Parameter(torch.tensor([[mu]]), requires_grad=True)
        self.qy0_logvar = nn.Parameter(torch.tensor([[logvar]]),
                                       requires_grad=True)

        self.encoder_z0 = encoder_z0
        self.diffeq_solver = diffeq_solver
        self.pre_encoder = pre_encoder
        self.decoder = decoder

    # def f(self, t, y):  # Approximate posterior drift.
    #     if t.dim() == 0:
    #         t = torch.full_like(y, fill_value=t)
    #     # Positional encoding in transformers for time-inhomogeneous posterior.
    #     return self.drift(t, y)

    # def g(self, t, y):  # Shared diffusion.
    #     return self.sigma + 0 * y  #self.sigma.repeat(y.size(0), 1)

    # def h(self, t, y):  # Prior drift.
    #     return self.theta * (self.mu - y)

    # def f_aug(self, t, y):  # Drift for augmented dynamics with logqp term.
    #     #y = y[:, 0:1]

    #     #f, g, h = self.f(t, y), self.g(t, y), self.h(t, y)
    #     f = self.f(t, y)
    #     g = self.g(t, y)
    #     h = self.h(t, y)
    #     u = _stable_division(f - h, g)
    #     f_logqp = .5 * (u**2).sum(dim=1, keepdim=True)
    #     import pdb
    #     pdb.set_trace()
    #     f = f.sum(dim=1, keepdim=True)

    #     return torch.cat([f, f_logqp], dim=1)

    # def g_aug(self, t, y):
    #     # Diffusion for augmented dynamics with logqp term.
    #     #y = y[:, 0:1]
    #     g = self.g(t, y)
    #     g_logqp = torch.zeros_like(y)
    #     return torch.cat([g, g_logqp], dim=1)

    def f(self, t, y):  # Approximate posterior drift.
        if t.dim() == 0:
            t = torch.full_like(y, fill_value=t)
        # Positional encoding in transformers for time-inhomogeneous posterior.
        #return self.net(torch.cat((torch.sin(t), torch.cos(t), y), dim=-1))
        return self.drift(t, y)

    def g(self, t, y):  # Shared diffusion.
        return self.sigma.repeat(y.size(0), 1)

    def h(self, t, y):  # Prior drift.
        return self.theta * (self.mu - y)

    def f_aug(self, t, y):  # Drift for augmented dynamics with logqp term.
        y = y[:, 0:1]

        f = self.f(t, y)
        g = self.g(t, y)
        h = self.h(t, y)
        u = _stable_division(f - h, g)
        f_logqp = .5 * (u**2).sum(dim=1, keepdim=True)

        return torch.cat([f, f_logqp], dim=1)

    def g_aug(self, t, y):  # Diffusion for augmented dynamics with logqp term.
        y = y[:, 0:1]
        g = self.g(t, y)
        g_logqp = torch.zeros_like(y)
        return torch.cat([g, g_logqp], dim=1)

    @property
    def py0_std(self):
        return torch.exp(.5 * self.py0_logvar)

    @property
    def qy0_std(self):
        return torch.exp(.5 * self.qy0_logvar)

    def get_reconstruction(self,
                           time_steps_to_predict,
                           truth,
                           truth_time_steps,
                           mask=None,
                           n_z0=1,
                           run_backwards=True,
                           mode=None):

        if True:
            # isinstance(self.encoder_z0, Encoder_z0_ODE_RNN) or \
            # isinstance(self.encoder_z0, Encoder_z0_RNN):
            if self.pre_encoder is not None:
                truth = self.pre_encoder(truth)
                mask = torch.ones_like(truth)

            truth_w_mask = truth
            if mask is not None:
                truth_w_mask = torch.cat((truth, mask), -1)
            first_point_mu, first_point_std = self.encoder_z0(
                truth_w_mask, truth_time_steps, run_backwards=run_backwards)

            first_point_enc = utils.sample_standard_gaussian(
                first_point_mu, first_point_std)

        else:
            raise Exception("Unknown encoder type {}".format(
                type(self.encoder_z0).__name__))

        first_point_std = first_point_std.abs()
        assert (torch.sum(first_point_std < 0) == 0.)

        first_point_enc_aug = first_point_enc
        means_z0_aug = first_point_mu

        assert (not torch.isnan(time_steps_to_predict).any())
        assert (not torch.isnan(first_point_enc).any())
        assert (not torch.isnan(first_point_enc_aug).any())

        # Shape of sol_y [n_z0, n_samples, n_timepoints, n_latents]
        #sol_y = self.diffeq_solver(first_point_enc_aug, time_steps_to_predict)
        z0 = first_point_enc_aug[0]
        ts = time_steps_to_predict
        method = 'srk'
        dt = 1e-2
        rtol = 1e-3
        atol = 1e-3
        adaptive = True
        qy0 = distributions.Normal(loc=self.qy0_mean, scale=self.qy0_std)
        py0 = distributions.Normal(loc=self.py0_mean, scale=self.py0_std)
        logqp0 = distributions.kl_divergence(qy0, py0).sum(dim=1)  # KL(t=0)

        #aug_z0 = torch.cat([z0, torch.zeros(z0.shape[0], 1).to(z0)], dim=1)
        aug_z0 = torch.cat([z0, torch.zeros(z0.shape).to(z0)], dim=1)

        # fix issue with defining f and g based on latent space dimension
        aug_zs = self.sdeint_fn(sde=self,
                                y0=aug_z0,
                                ts=ts,
                                method=method,
                                dt=dt,
                                adaptive=adaptive,
                                rtol=rtol,
                                atol=atol,
                                names={
                                    'drift': 'f_aug',
                                    'diffusion': 'g_aug'
                                })

        sol_z, logqp_path = aug_zs[:, :, 0:1], aug_zs[-1, :, 1]
        sol_z = sol_z.permute(1, 0, 2)
        kl = (logqp0 + logqp_path).mean(dim=0)  # KL(t=0) + KL(path).

        pred_x = self.decoder(sol_z)

        all_extra_info = {
            "first_point": (first_point_mu, first_point_std, first_point_enc),
            "latent_traj": sol_z.detach(),
            "kl": kl
        }

        return pred_x, all_extra_info

    def get_gaussian_likelihood(self, truth, pred_y, mask=None):
        # pred_y shape [n_z0, n_traj, n_tp, n_dim]
        # truth shape  [n_traj, n_tp, n_dim]

        if mask is not None:
            mask = mask.repeat(pred_y.size(0), 1, 1, 1)

        # Compute likelihood of the data under the predictions

        log_density_data = masked_gaussian_log_density(
            pred_y, truth, obsrv_std=self.obsrv_std, mask=mask)
        log_density_data = log_density_data.permute(1, 0)

        # Compute the total density
        # Take mean over n_z0
        log_density = torch.mean(log_density_data, 0)

        # shape: [n_traj]
        return log_density

    def get_mse(self, truth, pred_y, mask=None):
        # pred_y shape [n_z0, n_traj, n_tp, n_dim]
        # truth shape  [n_traj, n_tp, n_dim]
        if mask is not None:
            mask = mask.repeat(pred_y.size(0), 1, 1, 1)

        # Compute likelihood of the data under the predictions
        log_density_data = compute_mse(pred_y, truth, mask=mask)
        # shape: [1]
        return torch.mean(log_density_data)

    def compute_all_losses(self, batch_dict, n_z0=1, kl_coef=1., **kwargs):
        # Condition on subsampled points
        # Make predictions for all the points
        pred_y, info = self.get_reconstruction(
            batch_dict["observed_tp"],  #batch_dict["tp_to_predict"],
            batch_dict["observed_data"],
            batch_dict["observed_tp"],
            mask=batch_dict["observed_mask"],
            n_z0=n_z0,
            mode=batch_dict["mode"])

        #print("get_reconstruction done -- computing likelihood")
        fp_mu, fp_std, fp_enc = info["first_point"]
        fp_std = fp_std.abs()
        fp_distr = Normal(fp_mu, fp_std)

        assert (torch.sum(fp_std < 0) == 0.)

        # Compute likelihood of all the points
        rec_likelihood = self.get_gaussian_likelihood(
            batch_dict["observed_data"],  #batch_dict["data_to_predict"],
            pred_y,
            mask=None)  #batch_dict["mask_predicted_data"])

        mse = self.get_mse(
            batch_dict["observed_data"],  #batch_dict["data_to_predict"],
            pred_y,
            mask=None)  #batch_dict["mask_predicted_data"])

        pois_log_likelihood = torch.Tensor([0.]).to(
            get_device(batch_dict["data_to_predict"]))

        ################################
        # Compute CE loss for binary classification on Physionet
        device = get_device(batch_dict["data_to_predict"])
        ce_loss = torch.Tensor([0.]).to(device)
        if (batch_dict["labels"] is not None) and self.use_binary_classif:

            if (batch_dict["labels"].size(-1) == 1) or (len(
                    batch_dict["labels"].size()) == 1):
                ce_loss = compute_binary_CE_loss(info["label_predictions"],
                                                 batch_dict["labels"])
            else:
                ce_loss = compute_multiclass_CE_loss(
                    info["label_predictions"],
                    batch_dict["labels"],
                    mask=batch_dict["mask_predicted_data"])

        # IWAE loss
        kl = info['kl']
        loss = -torch.logsumexp(rec_likelihood - kl_coef * kl, 0)
        if torch.isnan(loss):
            loss = -torch.mean(rec_likelihood - kl_coef * kl, 0)

        results = {}
        results["loss"] = torch.mean(loss)
        results["likelihood"] = torch.mean(rec_likelihood).detach()

        #results["mse_introp"] = torch.mean(mse).detach()
        mse = ((batch_dict["observed_data"] - pred_y)**2).mean()
        results["mse_introp"] = mse.detach()

        # Derive MSE extrap
        pred_y_extrap, _ = self.get_reconstruction(
            batch_dict["tp_to_predict"],
            batch_dict["observed_data"],
            batch_dict["observed_tp"],
            mask=batch_dict["observed_mask"],
            n_z0=n_z0,
            mode=batch_dict["mode"])
        mse_extrap = ((batch_dict["data_to_predict"] -
                       pred_y_extrap)**2).mean()
        results["mse_extrap"] = mse_extrap.detach()

        #

        results["pois_likelihood"] = torch.mean(pois_log_likelihood).detach()
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["kl_first_p"] = 0.  #torch.mean(kldiv_z0).detach()
        results["std_first_p"] = torch.mean(fp_std).detach()

        if batch_dict["labels"] is not None and self.use_binary_classif:
            results["label_predictions"] = info["label_predictions"].detach()

        return results

    def sample_traj_from_prior(self, time_steps_to_predict, n_z0=1):
        # input_dim = starting_point.size()[-1]
        # starting_point = starting_point.view(1,1,input_dim)

        # Sample z0 from prior
        starting_point_enc = self.z0_prior.sample([n_z0, 1, self.latent_dim
                                                   ]).squeeze(-1)

        starting_point_enc_aug = starting_point_enc

        sol_y = self.diffeq_solver.sample_traj_from_prior(
            starting_point_enc_aug, time_steps_to_predict, n_z0=3)

        return self.decoder(sol_y)

    def compute_preenc_dec_loss(self, batch_dict):

        lat_y = self.pre_encoder(batch_dict["observed_data"])
        pred_y = self.decoder(lat_y[None])
        #pred_y = self.decoder(lat_y)

        # Need to change it, too slow if mask is not None
        dist = nn.MSELoss()
        # loglikelihood = self.get_gaussian_likelihood(
        #     batch_dict["observed_data"], pred_y, mask=None).mean()
        loglikelihood = -dist(batch_dict["observed_data"], pred_y[0]).mean()

        # 5) loss
        loss = -loglikelihood

        # 6) Extra information
        results = {}
        results["loss"] = loss

        return results
