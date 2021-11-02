import torch
import torch.nn as nn
import lib.utils as utils
from lib.utils import get_device

from lib.encoder_decoder import *
from lib.base_models import VAE_Baseline
from torch.distributions.normal import Normal


class ME(nn.Module):
    def __init__(self, dim):
        super(ME, self).__init__()
        self.fix = nn.Parameter(torch.rand(dim), requires_grad=True)
        # SD is diagonal, otheriwse need (dim, dim) and use MVN for sampling
        self.rand_rho = nn.Parameter(-torch.rand(dim) - 3, requires_grad=True)
        self.rand_eff = None
        self.me = None

    def forward(self, x):
        # Performs random projection
        x = x.reshape(x.shape[:-1] + (-1, 1, self.me.shape[-2]))
        ans = torch.matmul(x, self.me).flatten(2)
        return ans

    def update(self, n_samples, n_z0=1, n_w=1):
        # Should be updated every batch, not single forward pass
        # we sample as diagonal variance, i.e. parameters are independent,
        # otheriwse we need to use multivariate
        distr = Normal(torch.zeros_like(self.fix),
                       torch.ones_like(self.rand_sd))
        self.rand_eff = distr.sample((n_z0 * n_w, n_samples))
        self.me = self.fix + self.rand_eff * self.rand_sd

    @property
    def rand_sd(self):
        return torch.nn.functional.softplus(self.rand_rho) + 1e-8

    def print(self, string=False):
        s = "--------------------\n"
        s += 'fix_eff: %s\n' % self.fix.cpu().data.numpy()
        s += 'rand_rho: %s\n' % self.rand_rho.cpu().data.numpy()
        s += 'rand_sd: %s\n' % self.rand_sd.cpu().data.numpy()
        s += "--------------------"
        if string:
            return (s)
        else:
            print(s)

    def __repr__(self):
        s = ('fix_eff: [%s]' % ','.join([str(i) for i in self.fix.shape]),
             'rand_rho: [%s]' % ','.join([str(i)
                                          for i in self.rand_rho.shape]))
        return '\n'.join(s)


class LatentMEODE(VAE_Baseline):
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

        super(LatentMEODE, self).__init__(input_dim=input_dim,
                                          latent_dim=latent_dim,
                                          z0_prior=z0_prior,
                                          device=device,
                                          obsrv_std=obsrv_std)

        self.me = diffeq_solver.ode_func.gradient_net[-1]  #reference
        self.encoder_z0 = encoder_z0
        self.diffeq_solver = diffeq_solver
        self.decoder = decoder
        self.pre_encoder = pre_encoder

    def get_reconstruction(
            self,
            time_steps_to_predict,
            truth,  #observed
            truth_time_steps,  #observed
            mask=None,
            n_z0=1,
            n_w=1,
            run_backwards=True,
            mode=None,  #not used => remove
            z0_info=None,
            z0=None):

        # Input:
        # z0_info = (mu, std, z0)
        # z0 is just z0. If info is provided, then recovered from there
        if z0_info is not None:
            assert (len(z0_info) == 3)
            z0_mu, z0_std, z0 = z0_info
        elif z0 is None:
            # Only for odernn encoder of z0
            if self.pre_encoder is not None:
                truth = self.pre_encoder(truth)
                mask = torch.ones_like(truth)

            truth_w_mask = truth
            if mask is not None:
                truth_w_mask = torch.cat((truth, mask), -1)

            z0_mu, z0_rho = self.encoder_z0(truth_w_mask,
                                            truth_time_steps,
                                            run_backwards=run_backwards)
            import pdb
            pdb.set_trace()

            z0_std = torch.nn.functional.softplus(z0_rho) + 1e-8

            z0_mu_ = z0_mu.repeat(n_z0, 1, 1)
            z0_std_ = z0_std.repeat(n_z0, 1, 1)
            z0 = utils.sample_standard_gaussian(z0_mu_, z0_std_)
            z0 = z0.repeat(n_w, 1, 1)

        else:
            # z0 is given => no need to sample
            # Necessary to regenerate best trajectory from ME
            z0_mu = None
            z0_std = None

            if hasattr(self.pre_encoder, 'update'):
                self.pre_encoder.update(truth)  #dry run to update weights

        assert (not torch.isnan(time_steps_to_predict).any())
        assert (not torch.isnan(z0).any())

        # Shape of sol_z [n_z0, n_samples, n_timepoints, n_latents]
        #print("Memory leak here -- OOM")

        sol_z = self.diffeq_solver(z0, time_steps_to_predict)  #trajectory
        pred_x = self.decoder(sol_z)

        all_extra_info = {
            "first_point": (z0_mu, z0_std, z0),
            "latent_traj": sol_z.detach()
        }

        # if self.use_binary_classif:
        #     if self.classif_per_tp:
        #         all_extra_info["label_predictions"] = self.classifier(sol_z)
        #     else:
        #         all_extra_info["label_predictions"] = self.classifier(
        #             z0).squeeze(-1)

        return pred_x, all_extra_info

    # def sample_traj_from_prior(self, time_steps_to_predict, n_z0=1):
    #     # input_dim = starting_point.size()[-1]
    #     # starting_point = starting_point.view(1,1,input_dim)

    #     # Sample z0 from prior
    #     z0 = self.z0_prior.sample([n_z0, 1, self.latent_dim]).squeeze(-1)
    #     sol_z = self.diffeq_solver.sample_traj_from_prior(
    #         z0, time_steps_to_predict, n_z0=3)

    #     return self.decoder(sol_z)

    def compute_all_losses(self,
                           batch_dict,
                           n_z0=1,
                           n_w=1,
                           kl_coef=1.,
                           z0_info=None,
                           z0=None,
                           summary=False):
        # Summary if make a summary of different statistics except loss

        # Condition on subsampled points
        # Make predictions for all the points
        pred_y, info = self.get_reconstruction(
            batch_dict["observed_tp"],
            batch_dict["observed_data"],
            batch_dict["observed_tp"],
            mask=batch_dict["observed_mask"],
            n_z0=n_z0,
            n_w=n_w,
            mode=batch_dict["mode"],
            z0_info=z0_info,
            z0=z0)
        z0_mu, z0_std, z0 = info["first_point"]
        z0_std = z0_std.abs()
        assert (torch.sum(z0_std < 0) == 0.)

        # 1) Get best trajectories as indicator function
        best_pred_y, selected_z0, selected_me = self.get_best_traj(
            pred_y, batch_dict['observed_data'], z0)

        # 2) Compute kl
        # z0
        q_z = Normal(z0_mu, z0_std).log_prob

        # prior - should replace with model relation or so
        z0_prior_mu = torch.zeros_like(z0_mu)
        z0_prior_std = torch.ones_like(z0_std)
        q_prior_z = Normal(z0_prior_mu, z0_prior_std).log_prob

        me_prior_mu = torch.zeros_like(self.me.fix)
        me_prior_std = torch.ones_like(self.me.fix)
        q_prior_me = Normal(me_prior_mu, me_prior_std).log_prob

        # ME
        q_me = Normal(self.me.fix, self.me.rand_sd).log_prob

        kl = (q_z(selected_z0) - q_prior_z(selected_z0)).mean() +\
              (q_me(selected_me) - q_prior_me(selected_me)).mean()

        if torch.isnan(kl):
            print(z0_mu)
            print(z0_std)
            # z0_std[torch.isnan(q_z(selected_z0))] happens to be zero
            # Need to redo it with rho
            # nan happened here: q_z(selected_z0)
            # torch.isnan(q_z(selected_z0)).any()
            import pdb
            pdb.set_trace()
            print("Debug KL")
            raise Exception("kl is Nan!")

        # 3) Compute likelihood
        loglikelihood = self.get_gaussian_likelihood(
            batch_dict["observed_data"],
            best_pred_y,
            mask=batch_dict["observed_mask"]).mean()

        # 4) loss
        loss = kl * kl_coef - loglikelihood
        #loss = -torch.logsumexp(rec_likelihood - kl_coef * kldiv_z0, 0)

        # 5) Extra information
        results = {}
        if summary:
            results["loss"] = loss.item()  #detach()
        else:
            results["loss"] = loss

        results["likelihood"] = loglikelihood.item()  #detach()
        results["kl_first_p"] = kl.item()  #detach()
        results["mu_first_p"] = torch.mean(z0_mu).item()  #detach()
        results["std_first_p"] = torch.mean(z0_std).item()  #detach()

        #results['best_ind'] = best_ind
        results['selected_z0'] = selected_z0.detach()
        results['selected_me'] = selected_me.detach()

        if summary:
            # Compute interpolation MSE
            mse = ((batch_dict["observed_data"] - best_pred_y)**2).mean()
            results["mse_introp"] = mse.detach()

            if batch_dict['mode'] == "extrap":
                #compute MSE for extrapolated data
                self.me.me = selected_me
                # Remark**: in encoders which use pool, unpool, we have to
                # update indices of pool and size, based on last observed element,
                # to compute decoder. Thus, even if we have z0, we need to
                # provide last observed point for proper reconstruction.
                # Another way would be to combine observed time + predict time
                # and use all observed data, but we need mse of just last step.

                best_reconstruction, _ = self.get_reconstruction(
                    batch_dict["tp_to_predict"],
                    batch_dict["observed_data"][:, -1:],  #Remark**
                    None,  #batch_dict["observed_tp"],
                    mask=batch_dict["observed_mask"],
                    n_z0=1,
                    n_w=1,
                    z0=selected_z0)
                mse = self.get_mse(
                    batch_dict["data_to_predict"],
                    best_reconstruction,
                    mask=None  #batch_dict["mask_predicted_data"]
                )

                # can compute MSE per point as:
                # mse_per_point = ((batch_dict["data_to_predict"] -\
                #                   best_reconstruction[0])**2).sum(-1).mean(0)

                results["mse_extrap"] = torch.mean(mse).detach()

        return results

    def get_best_reconstruction(self,
                                batch_dict,
                                n_z0=1,
                                n_w=1,
                                is_numpy=False,
                                return_z_me=False,
                                z0_info=None,
                                z0=None):

        pred_y, info = self.get_reconstruction(
            batch_dict["observed_tp"],
            batch_dict["observed_data"],
            batch_dict["observed_tp"],
            mask=batch_dict["observed_mask"],
            n_z0=n_z0,
            n_w=n_w,
            mode=batch_dict["mode"],
            z0_info=z0_info,
            z0=z0)
        # z0_mu, z0_std, z0 = info["first_point"]
        _, _, z0 = info['first_point']
        best_reconstruction, selected_z0, selected_me = self.get_best_traj(
            pred_y, batch_dict['observed_data'], z0)

        if is_numpy:
            best_reconstruction = best_reconstruction.cpu().numpy()
            selected_z0 = selected_z0.cpu().numpy()
            selected_me = selected_me.cpu().numpy()

        if return_z_me:
            return best_reconstruction, selected_z0, selected_me
        else:
            return best_reconstruction

    def get_best_traj(self, pred_y, y, z0, return_ind=False):
        # 1) select index based on smallest distance
        # pred_y, batch_dict['observed_data']
        # n_z*n_w, batch, time, dimension
        y = y.repeat((pred_y.size(0), ) + (1, ) * (len(pred_y.shape) - 1))

        #correlation = vcorrcoef(y.detach(), pred_y.detach(), ax=2)
        # Normalizae MSE in simulations for every example
        # to be in range [0, 1]
        mse = ((pred_y -
                y)**2).sum(axis=[i for i in range(2, len(pred_y.shape))])
        mse = (mse - mse.min(dim=0)[0])/\
            (mse.max(dim=0)[0] - mse.min(dim=0)[0])

        dist = -mse  #+ correlation
        best_ind = dist.max(dim=0)[1]  # select trajectory based on ind

        # 2) Make an indecies-mask for gather
        # find best reconstructed values
        best_ind_y = best_ind.reshape(1, -1)
        for _ in range(len(pred_y.shape) - 2):
            best_ind_y = best_ind_y.unsqueeze(2)
        best_ind_y = best_ind_y.repeat((1, 1) + pred_y.shape[2:])

        # find z0 and me, which lead to best reconstruction
        # Old
        #best_ind_z0_me = best_ind.reshape(1, -1).unsqueeze(2)
        #best_ind_z0_me = best_ind_z0_me.repeat(1, 1, self.latent_dim)

        # New
        best_ind_z0_me = best_ind.reshape(1, -1).unsqueeze(2)
        best_ind_z0 = best_ind_z0_me.repeat(1, 1, self.latent_dim)
        best_ind_me = best_ind_z0_me.unsqueeze(3).unsqueeze(4).repeat(
            1, 1, self.latent_dim, self.me.me.shape[-2], 1)

        if return_ind:
            #return best_ind, best_ind_y, best_ind_z0_me
            return best_ind, best_ind_y, best_ind_z0, best_ind_me
        else:
            # 3) Select best initial point z0 and ME
            best_pred_y = pred_y.gather(0, best_ind_y)
            # Old
            # best_z0 = z0.gather(0, best_ind_z0_me)
            # best_me = self.me.me.gather(0, best_ind_z0_me)

            # New
            best_z0 = z0.gather(0, best_ind_z0)
            best_me = self.me.me.gather(0, best_ind_me)

            return best_pred_y, best_z0, best_me


def vcorrcoef(x, y, ax=1):
    # Function is written that ax is time axis
    # Final correlation is aggregated on all axis after ax
    shape = list(x.shape)
    shape[ax] = 1

    xm = x.mean(axis=ax).reshape(shape)
    ym = y.mean(axis=ax).reshape(shape)

    r_num = ((x - xm) * (y - ym)).sum(axis=ax)
    r_den = torch.sqrt(((x - xm)**2).sum(axis=ax) * ((y - ym)**2).sum(axis=ax))
    corr = r_num / r_den
    corr = corr.sum(axis=[i for i in range(ax, len(corr.shape))])
    return corr
