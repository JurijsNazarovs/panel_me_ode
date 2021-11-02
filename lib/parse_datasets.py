import os
import numpy as np

import torch
import torch.nn as nn
import importlib

import lib.utils as utils
from torch.utils.data import DataLoader

import sys
sys.path.append("../")  # to access data_generators

from data_generators.me_ode_1d import MEODE1d
from data_generators.mujoco_physics import HopperPhysics
from data_generators.rotating_mnist import RotatingMnist
from data_generators.adni import Adni
from data_generators.tadpole import Tadpole


def parse_datasets(args, device):
    def basic_collate_fn(batch,
                         time_steps,
                         args=args,
                         device=device,
                         data_type="train"):

        if args.dataset == "adni":
            batch = batch[0]
        else:
            batch = torch.stack(batch)

        data_dict = {"data": batch, "time_steps": time_steps}
        data_dict = utils.split_and_subsample_batch(data_dict,
                                                    args,
                                                    data_type=data_type)
        return data_dict

    dataset_name = args.dataset
    data_gen = None

    #n_total_tp = args.n_t + args.extrap * 2
    #print("n_total_tp:", n_total_tp)

    if dataset_name == "toy":
        data_gen = MEODE1d(root="data",
                           download=False,
                           n_samples=args.n_samples,
                           n_t=args.n_t,
                           min_t=args.min_t,
                           max_t=args.max_t,
                           y0_mean=1.3,
                           y0_std=0.01,
                           fix_eff=0.3,
                           rand_eff_std=0.1,
                           device=device,
                           name="ME_ODE_1d")

    elif dataset_name == "hopper":
        # MuJoCo dataset
        data_gen = HopperPhysics(root='data',
                                 n_samples=args.n_samples,
                                 n_t=args.n_t,
                                 n_same_initial=1,
                                 n_angles=args.n_angles,
                                 fix_eff=0.3,
                                 rand_eff_std=0.1,
                                 device=device,
                                 steps_to_skip=20,
                                 name="ME_Hopper")  # Generate data
    elif dataset_name == "rotmnist":
        # Rotating MNIST
        data_gen = RotatingMnist(
            root='data',
            n_samples=args.n_samples,
            n_t=args.n_t,
            n_same_initial=4,
            n_angles=args.n_angles,
            frame_size=28,
            device=device,
            specific_digit=None if args.mnist_digit < 0 else args.mnist_digit,
            n_styles=10,
            mnist_data_path=None,
            mnist_labels_path=None,
            name="ME_Rotating_MNIST")  # Generate data
    elif dataset_name == "tadpole":
        # Brain Images TADPOLE
        data_gen = Tadpole(datapath='data/TADPOLE/cdrsb/',
                           device=device,
                           name="TADPOLE")  # Generate data
    elif dataset_name == "adni":
        # Brain Images ADNI
        adni_path = 'data/ADNI/MRI3_Seqs'
        batch_size = args.batch_size
        train_gen = Adni(datapath=adni_path,
                         device=device,
                         name="ADNI",
                         batch_size=batch_size,
                         is_train=True)

        test_gen = Adni(datapath=adni_path,
                        device=device,
                        name="ADNI",
                        batch_size=batch_size,
                        is_train=False)
    else:
        raise Exception("Unknown dataset: %s" % dataset_name)

    if dataset_name == "adni":
        # if data_gen.n > args.n_samples:
        #     print("Loaded data set has %d samples" % data_gen.n)
        #     print("First %d samples will be chosen" % args.n_samples)
        #     dataset = dataset[:args.n_samples]

        time_steps = train_gen.t
        train_dataloader = DataLoader(
            train_gen,
            shuffle=False,
            collate_fn=lambda batch: basic_collate_fn(
                batch, time_steps, data_type="train"))

        test_dataloader = DataLoader(test_gen,
                                     shuffle=False,
                                     collate_fn=lambda batch: basic_collate_fn(
                                         batch, time_steps, data_type="test"))

        data_objects = {
            "data_gen": train_gen,
            "train_dataloader": utils.inf_generator(train_dataloader),
            "test_dataloader": utils.inf_generator(test_dataloader),
            "input_dim": 1,  # batch, time, dim/channel, ...
            "n_train_batches": train_gen.n_train_batches,
            "n_test_batches": train_gen.n_test_batches,
        }
    else:
        dataset = data_gen.data
        if len(dataset) > args.n_samples:
            print("Loaded data set has %d samples" % len(dataset))
            print("First %d samples will be chosen" % args.n_samples)
            dataset = dataset[:args.n_samples]

        time_steps = data_gen.t
        train_y, test_y = utils.split_train_test(dataset, train_fraq=0.8)

        batch_size = min(args.batch_size, args.n_samples)
        train_dataloader = DataLoader(
            train_y,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: basic_collate_fn(
                batch, time_steps, data_type="train"))
        test_dataloader = DataLoader(
            test_y,
            batch_size=batch_size,  # args.n_samples,
            shuffle=False,
            collate_fn=lambda batch: basic_collate_fn(
                batch, time_steps, data_type="test"))

        data_objects = {
            "data_gen": data_gen,
            "train_dataloader": utils.inf_generator(train_dataloader),
            "test_dataloader": utils.inf_generator(test_dataloader),
            "input_dim": dataset.shape[2],  # batch, time, dim/channel, ...
            "n_train_batches": len(train_dataloader),
            "n_test_batches": len(test_dataloader)
        }

    return data_objects
