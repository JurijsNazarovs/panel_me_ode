import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import relu

import lib.utils as utils

from lib.encoder_decoder import *
from lib.adni.adni_encoder_decoder import *
from lib.diffeq_solver import DiffeqSolver  #replace with sde
from torch.distributions.normal import Normal
from lib.ode_func import ODEFunc

import importlib
import lib.latent_sde
importlib.reload(lib.latent_sde)
from lib.latent_sde import LatentSDE


################################################################################
def create_LatentSDE_model(args,
                           input_dim,
                           z0_prior,
                           obsrv_std,
                           device,
                           n_labels=1):

    # 0) Decide if we do pre_encoder transformation
    gen_data_dim = input_dim
    if args.dataset in ["toy", "hopper", "tadpole"]:
        pre_encoder = None
        enc_input_dim = int(input_dim) * 2  # we concatenate the mask
    else:
        pre_enc_dim = args.latents  #args.pre_enc_dim
        if args.dataset == "rotmnist":
            pre_encoder = Encoder2d(input_dim, pre_enc_dim)
        else:
            pre_encoder = Encoder3d(input_dim, pre_enc_dim)
        enc_input_dim = pre_enc_dim * 2  # we concatenate the mask

    if args.dataset == "toy":
        # Because in this example \Gamma(z) = 1, to evaluate
        # how good our sampling is to fit random and fixed effect
        ode_func_net = nn.Sequential(torch.nn.Identity())
    else:
        ode_func_net = utils.create_net(args.latents,
                                        args.latents,
                                        n_layers=args.gen_layers,
                                        n_units=args.units,
                                        nonlinear=nn.Tanh)

    gen_ode_func = ODEFunc(ode_func_net=ode_func_net, device=device).to(device)

    diffeq_solver = DiffeqSolver(  #gen_data_dim,
        gen_ode_func,
        'rk4',  #rk4 is fixed timegrid method to avoid dynamic allocation of memory
        #'euler',  #dopri5',
        args.latents,
        odeint_rtol=1e-3,
        odeint_atol=1e-4,
        device=device)

    # 2) Define encoder
    z0_dim = args.latents
    z0_diffeq_solver = None
    n_rec_dims = args.rec_dims
    if args.z0_encoder == "odernn":
        ode_func_net = utils.create_net(n_rec_dims,
                                        n_rec_dims,
                                        n_layers=args.rec_layers,
                                        n_units=args.units,
                                        nonlinear=nn.Tanh)

        # rec_ode_func defines the gradient computation for ode,
        # based on ode_funct_net
        rec_ode_func = ODEFunc(  #input_dim=enc_input_dim,
            #latent_dim=n_rec_dims,
            ode_func_net=ode_func_net,
            device=device).to(device)
        z0_diffeq_solver = DiffeqSolver(  #enc_input_dim,
            rec_ode_func,
            "euler",
            args.latents,
            odeint_rtol=1e-3,
            odeint_atol=1e-4,
            device=device)

        encoder_z0 = Encoder_z0_ODE_RNN(
            n_rec_dims,  #output
            enc_input_dim,  #input
            z0_diffeq_solver,
            z0_dim=z0_dim,
            n_gru_units=args.gru_units,
            device=device).to(device)
    elif args.z0_encoder == "rnn":
        encoder_z0 = Encoder_z0_RNN(
            z0_dim,  #output
            enc_input_dim,  #input
            lstm_output_size=n_rec_dims,
            device=device).to(device)
    else:
        raise Exception("Unknown encoder for Latent ODE model: " +
                        args.z0_encoder)

    # Define decoder
    if args.dataset == "toy":
        decoder = torch.nn.Identity().to(device)
    else:
        if args.dataset in ["hopper", "tadpole"]:
            decoder = Decoder(args.latents, gen_data_dim).to(device)
        elif args.dataset == "rotmnist":
            decoder = Decoder2d(args.latents, gen_data_dim).to(device)
        elif args.dataset == "adni":
            decoder = Decoder3d(pre_encoder, args.latents,
                                gen_data_dim).to(device)
        else:
            raise Exception("Unknown decoder for dataset: " + args.dataset)

    # Build model
    model = LatentSDE(input_dim=gen_data_dim,
                      latent_dim=args.latents,
                      encoder_z0=encoder_z0,
                      decoder=decoder,
                      diffeq_solver=diffeq_solver,
                      z0_prior=z0_prior,
                      device=device,
                      obsrv_std=obsrv_std,
                      pre_encoder=pre_encoder,
                      adjoint=args.sde_adjoint,
                      drift=gen_ode_func).to(device)

    return model
