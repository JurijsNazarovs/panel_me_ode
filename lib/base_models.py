import numpy as np
import torch
import torch.nn as nn

import lib.utils as utils
from lib.encoder_decoder import *
from lib.likelihood_eval import *

from torch.distributions.normal import Normal
from torch.distributions import kl_divergence


class VAE_Baseline(nn.Module):
    def __init__(self,
                 input_dim,
                 latent_dim,
                 z0_prior,
                 device,
                 obsrv_std=0.01):

        super(VAE_Baseline, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device

        self.obsrv_std = torch.Tensor([obsrv_std]).to(device)
        self.z0_prior = z0_prior

        z0_dim = latent_dim

    def get_gaussian_likelihood(self, truth, pred_y, mask=None):
        # pred_y shape [n_traj_samples, n_traj(batch_size), n_tp, n_dim]
        # truth shape  [n_traj(batch_size), n_tp, n_dim]
        truth = torch.flatten(truth, 2)
        pred_y = torch.flatten(pred_y, 3)
        n_traj, n_tp, n_dim = truth.size()

        # Make truth consistent in size with n_traj_samples
        truth = truth.repeat(pred_y.size(0), 1, 1, 1)

        if mask is not None:
            mask = torch.flatten(mask, 2)
            mask = mask.repeat(pred_y.size(0), 1, 1, 1)

        # Compute likelihood of the data under the predictions
        log_density_data = masked_gaussian_log_density(
            pred_y, truth, obsrv_std=self.obsrv_std, mask=mask)
        log_density_data = log_density_data.permute(1, 0)

        # mean over batch, not n_traj_samples
        log_density = torch.mean(log_density_data, 1)
        # shape: [n_traj_samples]

        return log_density

    def get_mse(self, truth, pred_y, mask=None):
        # pred_y shape [n_z0, n_traj, n_tp, n_dim]
        # truth shape  [n_traj, n_tp, n_dim]
        truth = torch.flatten(truth, 2)
        pred_y = torch.flatten(pred_y, 3)

        n_traj, n_tp, n_dim = truth.size()

        # Compute likelihood of the data under the predictions
        truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)

        if mask is not None:
            mask = torch.flatten(mask, 2)
            mask = mask.repeat(pred_y.size(0), 1, 1, 1)

        # Compute likelihood of the data under the predictions
        log_density_data = compute_mse(pred_y, truth_repeated, mask=mask)
        # shape: [1]
        return torch.mean(log_density_data)

    def compute_preenc_dec_loss(self, batch_dict):
        # Functions is used to train encoder-decoder before
        # running main temporal block. Just to help encoder/decoder to converge
        # faster.

        if self.pre_encoder is None:
            raise ValueError("You try to compute preencoder, but it is None.")
        lat_y = self.pre_encoder(batch_dict["observed_data"])
        pred_y = self.decoder(lat_y[None])

        # In preencoder_decore loss we only orient on l2 (MSE)
        # to better train encoder.
        results = {}
        results["loss"] = nn.MSELoss()(batch_dict["observed_data"],
                                       pred_y[0]).mean()

        return results

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

        kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)

        if torch.isnan(kldiv_z0).any():
            print(fp_mu)
            print(fp_std)
            raise Exception("kldiv_z0 is Nan!")

        # Mean over number of latent dimensions
        # kldiv_z0 shape: [n_z0, n_traj, n_latent_dims]
        # if prior is a mixture of gaussians (KL is estimated)
        # kldiv_z0 shape: [1, n_traj, n_latent_dims]
        # if prior is a standard gaussian (KL is computed exactly)
        # shape after: [n_z0]
        kldiv_z0 = torch.mean(kldiv_z0, (1, 2))

        # Compute likelihood of all the points
        rec_likelihood = self.get_gaussian_likelihood(
            batch_dict["observed_data"],  #batch_dict["data_to_predict"],
            pred_y,
            mask=None)  #batch_dict["mask_predicted_data"])

        mse = self.get_mse(
            batch_dict["observed_data"],  #batch_dict["data_to_predict"],
            pred_y,
            mask=batch_dict["mask_predicted_data"])

        # VAE loss
        loss = -torch.logsumexp(rec_likelihood - kl_coef * kldiv_z0, 0)
        if torch.isnan(loss):
            loss = -torch.mean(rec_likelihood - kl_coef * kldiv_z0, 0)

        results = {}
        results["loss"] = torch.mean(loss)
        results["likelihood"] = torch.mean(rec_likelihood).detach()

        results["mse_introp"] = torch.mean(mse).detach()
        #mse = ((batch_dict["observed_data"] - pred_y)**2).mean()
        #results["mse_introp"] = mse.detach()

        if batch_dict['mode'] == "extrap":
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

        results["kl_first_p"] = torch.mean(kldiv_z0).detach()
        results["mu_first_p"] = torch.mean(fp_mu).item()  #detach()
        results["std_first_p"] = torch.mean(fp_std).detach()

        return results
