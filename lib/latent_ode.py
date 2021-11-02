import torch
import torch.nn as nn

import lib.utils as utils
from lib.utils import get_device
from lib.encoder_decoder import *
from lib.base_models import VAE_Baseline


class LatentODE(VAE_Baseline):
    def __init__(self,
                 input_dim,
                 latent_dim,
                 encoder_z0,
                 decoder,
                 diffeq_solver,
                 z0_prior,
                 device,
                 obsrv_std=None,
                 pre_encoder=None):

        super(LatentODE, self).__init__(input_dim=input_dim,
                                        latent_dim=latent_dim,
                                        z0_prior=z0_prior,
                                        device=device,
                                        obsrv_std=obsrv_std)

        self.encoder_z0 = encoder_z0
        self.diffeq_solver = diffeq_solver
        self.pre_encoder = pre_encoder
        self.decoder = decoder

    def get_reconstruction(self,
                           time_steps_to_predict,
                           truth,
                           truth_time_steps,
                           mask=None,
                           n_z0=1,
                           run_backwards=True,
                           mode=None):

        if isinstance(self.encoder_z0, Encoder_z0_ODE_RNN) or \
         isinstance(self.encoder_z0, Encoder_z0_RNN):
            if self.pre_encoder is not None:
                truth = self.pre_encoder(truth)
                mask = torch.ones_like(truth)

            truth_w_mask = truth
            if mask is not None:
                truth_w_mask = torch.cat((truth, mask), -1)
            first_point_mu, first_point_std = self.encoder_z0(
                truth_w_mask, truth_time_steps, run_backwards=run_backwards)

            means_z0 = first_point_mu.repeat(n_z0, 1, 1)
            sigma_z0 = first_point_std.repeat(n_z0, 1, 1)
            first_point_enc = utils.sample_standard_gaussian(
                means_z0, sigma_z0)

        else:
            raise Exception("Unknown encoder type {}".format(
                type(self.encoder_z0).__name__))

        first_point_std = first_point_std.abs()
        assert (torch.sum(first_point_std < 0) == 0.)

        first_point_enc_aug = first_point_enc
        means_z0_aug = means_z0

        assert (not torch.isnan(time_steps_to_predict).any())
        assert (not torch.isnan(first_point_enc).any())
        assert (not torch.isnan(first_point_enc_aug).any())

        # Shape of sol_y [n_z0, n_samples, n_timepoints, n_latents]
        sol_y = self.diffeq_solver(first_point_enc_aug, time_steps_to_predict)
        pred_x = self.decoder(sol_y)

        all_extra_info = {
            "first_point": (first_point_mu, first_point_std, first_point_enc),
            "latent_traj": sol_y.detach()
        }

        return pred_x, all_extra_info
