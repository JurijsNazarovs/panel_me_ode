import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt
import importlib

import time
import datetime
import argparse
import numpy as np
import pandas as pd
from random import SystemRandom
from sklearn import model_selection

import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.optim as optim

import lib.utils as utils

#from lib.rnn_baselines import *
#from lib.ode_rnn import *

from lib.create_latent_ode_model import create_LatentODE_model
from lib.ode_func import ODEFunc
from lib.diffeq_solver import DiffeqSolver
from lib.utils import summary

#import lib.parse_datasets
#importlib.reload(lib.parse_datasets)
from lib.parse_datasets import parse_datasets

#import lib.create_latent_me_ode_model
#importlib.reload(lib.create_latent_me_ode_model)
from lib.create_latent_me_ode_model import create_LatentMEODE_model

#import lib.create_latent_sde_model
#importlib.reload(lib.create_latent_sde_model)
from lib.create_latent_sde_model import create_LatentSDE_model

#import lib.inference
#importlib.reload(lib.inference)
from lib.inference import inference

import torch
import time

import config
importlib.reload(config)
from config import get_arguments


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("output.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


sys.stdout = Logger()


def train(args, data_obj, model, optimizer, kl_coef, n_batches, epoch):
    model.train()

    start_time = time.time()
    for itr in range(n_batches):
        print("Batch: %d" % itr, end='\r')
        batch_dict = utils.get_next_batch(data_obj["train_dataloader"])

        optimizer.zero_grad()
        if args.model == 'meode':
            model.me.update(len(batch_dict['observed_data']), args.n_z0,
                            args.n_w)

        if epoch <= args.epochs_encoder_only:
            # In case we want to train encoder/decoder only
            # without temporal information
            train_res = model.compute_preenc_dec_loss(batch_dict)
        else:
            train_res = model.compute_all_losses(
                batch_dict,
                n_z0=args.n_z0,  #n_z0
                n_w=args.n_w,  #only for latent ME ode
                kl_coef=kl_coef)

        print("Loss: %.6f" % train_res["loss"])
        train_res["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

        # Save every batch
        torch.save(
            {
                'args': args,
                'state_dict': model.state_dict(),
                'epoch': epoch,
            }, ckpt_path + "_batch")

    end_time = time.time()
    print("Epoch time: %f" % ((end_time - start_time)))

    # Save every epochs
    torch.save(
        {
            'args': args,
            'state_dict': model.state_dict(),
            'epoch': epoch,
        }, ckpt_path)


def test(args,
         data_obj,
         model,
         logger,
         epoch,
         kl_coef=1,
         compute_summary=True,
         best_loss=None):
    model.eval()
    if compute_summary:
        test_res = summary(model,
                           data_obj["test_dataloader"],
                           args,
                           n_batches=min(data_obj["n_test_batches"], 10**8),
                           experimentID=experimentID,
                           device=device,
                           n_z0=args.n_z0,
                           n_w=args.n_w,
                           kl_coef=kl_coef)

        message = 'Epoch %04d, Loss %.4f,Likelihood %.4f,KL %.4f,MSE_introp %.4f' %\
            (epoch, test_res["loss"],
             test_res["likelihood"], test_res["kl_first_p"],
             test_res['mse_introp'])
        if args.extrap:
            message += ',MSE_extrap %.4f' % test_res['mse_extrap']
        message += '\nz0_mu %.4f,z0_std %.4f' %\
            (test_res['mu_first_p'], test_res['std_first_p'])

        logger.info("Experiment " + str(experimentID))
        logger.info(message)
        logger.info("KL_coef %.4f" % kl_coef)
        if args.dataset == 'toy' and args.model == 'meode':
            logger.info(model.me.print(string=True))

        if best_loss is not None:
            if test_res["loss"] <= best_loss:
                best_loss = test_res['loss']
                logger.info("New best loss: %f" % best_loss)
                torch.save(
                    {
                        'args': args,
                        'state_dict': model.state_dict(),
                        'epoch': epoch,
                        'best_loss': best_loss,
                    }, ckpt_path + '_best')
            else:
                logger.info("Best loss still the same: %f" % best_loss)

    # Make testing inference: numpy, plots, ...
    if args.model == 'meode':
        which = "test"  #test train to save ME for future analysis
        if args.test_only:
            n_tests = 50  #data_obj['n_%s_batches' % which]
            n_obs_show = 4  #0=args.batch_size
        else:
            n_tests = 1
            n_obs_show = 1

        inference(
            model,
            data_obj["%s_dataloader" % which],
            n_tests=n_tests,
            n_obs_show=n_obs_show,  #0=args.batch_size
            n_z0=args.n_z0,
            n_w=args.n_w,
            is_save_np=args.test_only,
            is_plot=True,
            visualizer=data_obj['data_gen'].visualize,
            path_base="%s_%s/%03d" % (experimentID, which, epoch),
            is1d=(args.dataset in ["toy"]),  #, "tadpole"]),
            save_separate=args.dataset == "adni",
            extrap_extra_steps=5)

        if args.dataset in ["toy"]:
            inference(
                model,
                data_obj["%s_dataloader" % which],
                n_tests=n_tests,
                n_obs_show=1,  #0=args.batch_size
                n_z0=args.n_z0,
                n_w=args.n_w,
                is_save_np=args.test_only,
                is_plot=True,
                visualizer=data_obj['data_gen'].visualize,
                path_base="%s_%s/1sample_%03d" % (experimentID, which, epoch),
                is1d=True,
                save_separate=args.dataset == "adni",
                extrap_extra_steps=5)

    return best_loss  #either None or value


################################################################################
parser = get_arguments()
args = parser.parse_args()
if torch.cuda.is_available():
    device = torch.device('cuda')
    if args.device:
        device = "%s:%d" % (device, args.device)
else:
    device = torch.device('cpu')

print("Device:", device)
os.makedirs(args.save, exist_ok=True)

#args.extrap = False
if args.dataset == "toy":
    args.latents = 1

if __name__ == '__main__':
    #args.latent_me_ode = True
    #args.latent_ode = True
    #args.latent_sde = True
    #args.test_only = True
    #args.extrap=True

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    ## Establishing experiments ID and checkpoint path
    experimentID = args.experimentID
    if experimentID is None:
        experimentID = int(SystemRandom().random() * 100000)
    ckpt_path = os.path.join(args.save,
                             "experiment_" + str(experimentID) + '.ckpt')
    utils.makedirs("results/")

    ## Creating a string with the command to write in the log file later
    input_command = " ".join(sys.argv)

    ## Generate/load Data
    print("Sampling dataset of %d training examples" % args.n_samples)
    data_obj = parse_datasets(args, device)
    input_dim = data_obj["input_dim"]
    batch_dict = utils.get_next_batch(data_obj["train_dataloader"])

    ## Create the model
    obsrv_std = torch.Tensor([1e-3]).to(device)
    z0_prior = torch.distributions.Normal(
        torch.Tensor([0.0]).to(device),
        torch.Tensor([1.]).to(device))

    utils.printline()
    print("Constructing model: %s" % args.model)
    utils.printline()
    if args.model == 'meode':
        print("Latent Mixed Effect ODE model is selected")
        model = create_LatentMEODE_model(args, input_dim, z0_prior, obsrv_std,
                                         device).to(device)
    elif args.model == 'ode':
        print("Latent ODE model is selected")
        model = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std,
                                       device).to(device)
    # elif args.model == 'sde':
    #     print("Latent SDE model is selected")
    #     model = create_LatentSDE_model(args, input_dim, z0_prior, obsrv_std,
    #                                    device).to(device)
    else:
        raise Exception("Temporal model not specified")

    ##################################################################
    # Load checkpoint and evaluate the model
    if args.load:
        # In case we load model to contrinue from last epoch
        if args.best:
            ckpt_path_load = ckpt_path + "_best"
        else:
            ckpt_path_load = ckpt_path
        epoch_st, best_loss = utils.get_ckpt_model(ckpt_path_load, model,
                                                   device)
        epoch_st += 1
        print("Current best loss: %.8f" % best_loss)
    else:
        epoch_st, best_loss = 1, np.infty
    ##################################################################

    # Main part
    log_path = "logs/" + str(experimentID) + ".log"
    if not os.path.exists("logs/"):
        utils.makedirs("logs/")
    logger = utils.get_logger(logpath=log_path,
                              filepath=os.path.abspath("main.py"))
    logger.info(input_command)

    optimizer = optim.Adamax(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=.999)
    kl_scheduler = utils.LinearScheduler(iters=args.epochs_kl_anneal, maxval=1)

    if args.test_only:
        with torch.no_grad():
            test(args, data_obj, model, logger, epoch=0, compute_summary=True)
    else:
        for epoch in range(epoch_st):
            # in case we load model and need to update schedulers to
            # appropriate state
            if epoch < args.epochs_until_kl_inc:
                kl_coef = 0.
            else:
                kl_scheduler.step()  #update kl
                scheduler.step()  #update lr

        for epoch in range(epoch_st, args.n_epochs + 1):
            print('Epoch %04d' % epoch)
            if epoch < args.epochs_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = kl_scheduler.val
                kl_scheduler.step()  #update kl
                scheduler.step()  #update lr
                print("kl:", kl_coef)
                print("lr:", scheduler.get_last_lr())

            if not args.test_only:
                train(args,
                      data_obj,
                      model,
                      optimizer,
                      kl_coef,
                      n_batches=data_obj["n_train_batches"],
                      epoch=epoch)

            # Do testing and report summary
            if epoch % args.n_epochs_to_viz == 0 and\
               epoch >= args.n_epochs_start_viz and\
               epoch > args.epochs_until_kl_inc:
                with torch.no_grad():
                    best_loss = test(args,
                                     data_obj,
                                     model,
                                     logger,
                                     epoch,
                                     kl_coef,
                                     compute_summary=True,
                                     best_loss=best_loss)
