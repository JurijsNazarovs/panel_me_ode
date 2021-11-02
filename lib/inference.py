import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt
from PIL import Image

import os

import numpy as np
import torch

import lib.utils as utils
import matplotlib.gridspec as gridspec
from lib.utils import get_device
import pickle

#from matplotlib.lines import Line2D
#from scipy.stats import kde
#import subprocess
#import importlib
#importlib.reload(utils)
# from lib.encoder_decoder import *
# from lib.rnn_baselines import *
#from lib.ode_rnn import *
#import torch.nn.functional as functional
#from torch.distributions.normal import Normal
#from lib.likelihood_eval import masked_gaussian_log_density

# try:
#     import umap
# except:
#     print("Couldn't import umap")

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
LARGE_SIZE = 22

alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
#alphas = [0.1 + i for i in alphas]
percentiles = [0.999, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

color = "blue"
# From https://colorbrewer2.org/.
if color == "blue":
    #sample_colors = ('#8c96c6', '#8c6bb1', '#810f7c')
    sample_colors = ['k', 'r', 'b', 'g']
    fill_color = '#9ebcda'
    mean_color = 'r'  #'#4d004b'
    num_samples = len(sample_colors)
else:
    sample_colors = ('#fc4e2a', '#e31a1c', '#bd0026')
    fill_color = '#fd8d3c'
    mean_color = '#800026'
    num_samples = len(sample_colors)


def init_fonts(main_font_size=LARGE_SIZE):
    plt.rc('font', size=main_font_size)  # controls default text sizes
    plt.rc('axes', titlesize=main_font_size)  # fontsize of the axes title
    plt.rc('axes',
           labelsize=main_font_size - 2)  # fontsize of the x and y labels
    plt.rc('xtick',
           labelsize=main_font_size - 2)  # fontsize of the tick labels
    plt.rc('ytick',
           labelsize=main_font_size - 2)  # fontsize of the tick labels
    plt.rc('legend', fontsize=main_font_size - 2)  # legend fontsize
    plt.rc('figure', titlesize=main_font_size)  # fontsize of the figure title


def inference(
    model,
    data_loader,
    n_tests=1,  #can be a batch of data_loader
    n_obs_show=1,  #how many samples per batch we take
    n_z0=1,
    n_w=1,
    is_save_np=False,
    is_plot=False,
    visualizer=None,
    path_base="experiment/epoch_",
    is1d=False,
    save_separate=False,  #save not combined real and pred plots also
    get_best=True,  #find best trajectory
    extrap_extra_steps=None,
    #plot_args={}
):
    for test_iter in range(n_tests):
        data_dict = utils.get_next_batch(data_loader)

        if n_obs_show > 0:
            n_obs_show = min(n_obs_show, len(data_dict["data_to_predict"]))
            samples_show = np.random.choice(len(data_dict["data_to_predict"]),
                                            n_obs_show,
                                            replace=False)
            data_dict = utils.select_samples_from_batch(data_dict,
                                                        samples=samples_show)

        obs_data, obs_ts, reconstructions, rec_ts,\
            best_reconstruction, best_traj, best_me, extrap_line, *extra_terms = \
                get_data_time(model, data_dict, n_z0, n_w, get_best=get_best, is1d=is1d,
                              extrap_extra_steps=extrap_extra_steps)

        if is_save_np:
            #save numpy for next analysis
            out_path = "results/%s_%02d.pickle" % (path_base, test_iter)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            with open(out_path, 'wb') as f:
                pickle.dump(
                    [
                        obs_data,
                        reconstructions,
                        best_reconstruction,
                        best_traj,  #in latent space
                        best_me.cpu().numpy(),
                        *extra_terms
                    ],
                    f)

        if is_plot:
            plot_args_ = {
                'title': '',
                'linestyle': '-',
                'marker': 'x',
                'linewidth': 1,
                'n_samples_show': 0,  #-1 is all
                'show_percentiles': True,
                'show_sd': False,
                'show_obs': True,
                'show_arrows': False,  #not working
                'hide_ticks': True,
            }  #| plot_args #combine 2 dictionaries

            if is1d:  #batch, time, dim => 1d data
                dim_to_show = 0  #usefull for hopper dataset with several dim
                eps = 1 / 4
                # max_y = max(obs_data[:, :, dim_to_show].max(),
                #             reconstructions[:, :, :, dim_to_show].max()) + eps
                # min_y = min(obs_data[:, :, dim_to_show].min(),
                #             reconstructions[:, :, :, dim_to_show].min()) - eps

                # max_y = obs_data[:, :, dim_to_show].max() + eps
                # min_y = obs_data[:, :, dim_to_show].min() - eps

                max_y = 4  #6
                min_y = 1

                plot_1d(reconstructions[:, :, :, dim_to_show],
                        rec_ts,
                        obs_data[:, :, dim_to_show],
                        obs_ts,
                        best_traj=best_reconstruction,
                        ylims=[min_y, max_y],
                        extrap_line=extrap_line,
                        plot_path="plots/%s_%02d_best.pdf" %
                        (path_base, test_iter),
                        show_mean=True,
                        **plot_args_)
                plot_1d(reconstructions[:, :, :, dim_to_show],
                        rec_ts,
                        obs_data[:, :, dim_to_show],
                        obs_ts,
                        ylims=[min_y, max_y],
                        extrap_line=extrap_line,
                        plot_path="plots/%s_%02d.pdf" % (path_base, test_iter),
                        show_mean=False,
                        **plot_args_)
            else:
                plot_not1d(obs_data,
                           best_reconstruction,
                           visualizer,
                           data_dict,
                           path_base,
                           test_iter,
                           save_separate=save_separate)


def get_data_time(model,
                  data_dict,
                  n_z0=1,
                  n_w=1,
                  n_t_vis=100,
                  get_best=True,
                  is1d=False,
                  extrap_extra_steps=10):
    # n_t_vis - number of time steps used for visualization of
    # simulated trajectories for 1d plot
    predict_data = data_dict["data_to_predict"]
    predict_time_steps = data_dict["tp_to_predict"]

    observed_data = data_dict["observed_data"]
    observed_time_steps = data_dict["observed_tp"]
    observed_mask = data_dict["observed_mask"]
    device = get_device(predict_time_steps)

    extrap_line = None
    if data_dict['mode'] == "extrap":
        if is1d:
            rec_ts = utils.linspace_vector(observed_time_steps[0],
                                           predict_time_steps[-1],
                                           2 * n_t_vis).to(device)
            extrap_line = (observed_time_steps[-1] + predict_time_steps[0]) / 2
        else:
            rec_ts = torch.cat([observed_time_steps,
                                predict_time_steps]).to(device)
    else:
        if is1d:
            rec_ts = utils.linspace_vector(observed_time_steps[0],
                                           observed_time_steps[-1],
                                           n_t_vis).to(device)
        else:
            rec_ts = observed_time_steps

    # Sample trajectories
    with torch.no_grad():
        model.me.update(len(observed_data), n_z0, n_w)
        reconstructions, traj_info = model.get_reconstruction(
            rec_ts,
            observed_data,
            observed_time_steps,
            mask=observed_mask,
            n_z0=n_z0,
            n_w=n_w)
        best_reconstruction = None
        best_me = None
        best_traj = None

        if get_best:
            # 1) chose best z0 and me on observed data
            # 2) use these z0 and me to generate trajecttory on much more t
            #    to get smooth trajectory

            # 1)
            z0 = traj_info['first_point'][2]
            _, best_z0, best_me =\
                model.get_best_reconstruction(
                data_dict,
                n_z0=n_z0,
                n_w=n_w,
                return_z_me=True,
                z0=z0)

            # 2)
            model.me.me = best_me
            best_reconstruction, info = model.get_reconstruction(
                rec_ts,  #obs_ts + predict_ts
                observed_data,  #to reconstruct encoder
                None,  #observed_time_steps,
                mask=None,  #observed_mask,
                n_z0=1,
                n_w=1,
                z0=best_z0)
            best_reconstruction = best_reconstruction[0]
            best_traj = info['latent_traj'][0]

            if extrap_extra_steps is not None:
                # 3) compute extra steps of extrapolation when we do not have
                # data for to compute error
                #extra_ts = utils.linspace_vector(
                #    rec_ts[-1] + 1, rec_ts[-1] + extrap_extra_steps,
                #    extrap_extra_steps).to(device)

                # Have to do this way because otherwise 3d reconstruction using
                # encoder fails with pool-unpool indices.
                extra_ts = utils.linspace_vector(
                    rec_ts[0], rec_ts[-1] + extrap_extra_steps,
                    extrap_extra_steps + len(rec_ts)).to(device)

                best_reconstruction_extrap, info_extrap = model.get_reconstruction(
                    extra_ts,
                    observed_data,  #to reconstruct encoder
                    None,  #observed_time_steps,
                    mask=None,  #observed_mask,
                    n_z0=1,
                    n_w=1,
                    z0=best_z0)
                best_reconstruction_extrap = best_reconstruction_extrap[0]
                best_traj_extrap = info_extrap['latent_traj'][0]

    if data_dict['mode'] == "extrap":
        obs_data = torch.cat([observed_data, predict_data],
                             axis=1).cpu().numpy()
        obs_ts = torch.cat([observed_time_steps,
                            predict_time_steps]).cpu().numpy()
    else:
        obs_data = observed_data.cpu().numpy()
        obs_ts = observed_time_steps.cpu().numpy()

    result = (obs_data, obs_ts, reconstructions.cpu().numpy(),
              rec_ts.cpu().numpy(), best_reconstruction.cpu().numpy(),
              best_traj.cpu().numpy(), best_me, extrap_line)

    if extrap_extra_steps is not None:
        result += (best_reconstruction_extrap.cpu().numpy(),
                   best_traj_extrap.cpu().numpy())

    return result


def plot_1d(
        traj,
        ts_vis,
        obs_data,
        ts,
        title="",
        ylims=None,
        add_to_plot=False,
        mask=None,
        color=None,
        linestyle='-',
        linewidth=3,
        marker='x',
        show_percentiles=True,
        show_mean=False,
        show_sd=True,
        n_samples_show=0,  #trajectories to draw for each observ
        show_obs=True,
        show_arrows=False,
        hide_ticks=False,
        extrap_line=None,
        plot_path=None,
        best_traj=None):
    # traj are [n_z*n_w, batch, ...]

    fig, ax = plt.subplots(frameon=False)

    if show_percentiles:
        traj_ = traj.reshape(-1, len(ts_vis))
        traj_ = np.sort(traj_, axis=0)
        for alpha, percentile in zip(alphas, percentiles):
            idx = int((1 - percentile) / 2. * traj_.shape[0])
            traj_bot, traj_top = traj_[idx], traj_[-idx]
            ax.fill_between(ts_vis,
                            traj_bot,
                            traj_top,
                            alpha=alpha,
                            color=fill_color)

    if show_sd:
        traj_ = traj.reshape(-1, len(ts_vis))
        traj_mean = traj_.mean(axis=0)
        traj_sd = traj_.std(axis=0)
        mean_minus_sd = (traj_mean - traj_sd)
        mean_plus_sd = (traj_mean + traj_sd)

        ax.fill_between(
            ts_vis,
            mean_minus_sd,
            mean_plus_sd,
            alpha=0.3,  #alpha,
            color=fill_color)
    if show_mean:
        traj_ = traj.reshape(-1, len(ts_vis))
        ax.plot(ts_vis, traj_.mean(axis=0), color=mean_color)

    if n_samples_show == -1:
        n_samples_show = traj.shape[0]
    if n_samples_show > 0:
        n_samples_show = min(n_samples_show, traj.shape[0])
        # traj shape is: simulations, batch, time
        for i in range(traj.shape[1]):  #for a data from batch
            for j in range(n_samples_show):  #for all simulations for this data
                ax.plot(ts_vis,
                        traj[j, i, :],
                        color=sample_colors[i % len(sample_colors)],
                        linewidth=linewidth)
    if best_traj is not None:
        for i in range(best_traj.shape[0]):
            if best_traj.shape[0] == 1:
                color = 'g'
                linestyle = '-'
            else:
                if n_samples_show < 2:
                    color = sample_colors[i % len(sample_colors)]
                else:
                    color = 'g'
                linestyle = (0, (5, 5))  #loosely dashed
            ax.plot(ts_vis,
                    best_traj[i],
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth + 1)

    # plot markers for observed data
    if show_obs:
        for i in range(obs_data.shape[0]):
            if len(obs_data) == 1:
                color = "k"
            else:
                color = sample_colors[i % len(sample_colors)]

            ax.scatter(ts,
                       obs_data[i],
                       marker=marker,
                       zorder=3,
                       color=color,
                       s=105)

    # if show_arrows:  #no
    #     num, dt = 12, 0.12
    #     t, y = torch.meshgrid([
    #         torch.linspace(0.2, 1.8, num).to(device),
    #         torch.linspace(-1.5, 1.5, num).to(device)
    #     ])
    #     t, y = t.reshape(-1, 1), y.reshape(-1, 1)
    #     fty = model.f(t=t, y=y).reshape(num, num)
    #     dt = torch.zeros(num, num).fill_(dt).to(device)
    #     dy = fty * dt
    #     dt_, dy_, t_, y_ = dt.cpu().numpy(), dy.cpu().numpy(), t.cpu().numpy(
    #     ), y.cpu().numpy()
    #     plt.quiver(t_,
    #                y_,
    #                dt_,
    #                dy_,
    #                alpha=0.3,
    #                edgecolors='k',
    #                width=0.0035,
    #                scale=50)

    if hide_ticks:
        plt.xticks([], [])
        plt.yticks([], [])

    if ylims is not None:
        plt.ylim(ylims)

    if extrap_line is not None:
        plt.axvline(x=extrap_line, color='k', linestyle='--', linewidth=3)

    ax.set_title(title)
    plt.xlabel('$t$')
    plt.ylabel('$Y_t$')
    plt.tight_layout()

    if plot_path is not None:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        plt.close()


def plot_not1d(obs_data,
               best_reconstruction,
               visualizer,
               data_dict,
               path_base="plots/",
               test_iter=0,
               save_separate=False,
               img_h=80,
               img_w=80):
    for sample in range(obs_data.shape[0]):
        plot_path = "plots/%s_%02d_%d.png" %\
            (path_base, test_iter, sample)
        if save_separate:
            plot_path_real = "_real.".join(plot_path.split('.'))
            plot_path_pred = "_pred.".join(plot_path.split('.'))
        else:
            plot_path_real, plot_path_pred = None, None

        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        real_plot = visualizer(obs_data[sample],
                               concatenate=True,
                               plot_path=plot_path_real,
                               img_w=img_w,
                               img_h=img_h)
        pred_plot = visualizer(best_reconstruction[sample],
                               concatenate=True,
                               plot_path=plot_path_pred,
                               img_w=img_w,
                               img_h=img_h)

        if not (real_plot is None or pred_plot is None):
            comb_plot = np.concatenate([real_plot, pred_plot], axis=0)

            if data_dict['mode'] == 'extrap':
                # fill the gap with color between
                # observed and extrapolated time
                new_shape = list(comb_plot.shape)
                gap = img_w // 4
                new_shape[1] += gap
                new_plot = np.zeros(new_shape, dtype=comb_plot.dtype)

                before_gap = len(data_dict["observed_tp"]) * img_w
                new_plot[:, :before_gap] = comb_plot[:, :before_gap]
                new_plot[:, (before_gap + gap):] = comb_plot[:, before_gap:]

                comb_plot = new_plot

            im = Image.fromarray(comb_plot)
            im.save(plot_path)
