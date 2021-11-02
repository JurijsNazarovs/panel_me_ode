import argparse


def get_arguments():
    parser = argparse.ArgumentParser('Panel-ME ODE')
    parser.add_argument('--save',
                        type=str,
                        default='experiments/',
                        help="Path for save checkpoints")
    parser.add_argument('--load',
                        action='store_true',
                        default=None,
                        help="Whether to load model")
    parser.add_argument('--best',
                        action='store_true',
                        default=None,
                        help="To load the best model")
    parser.add_argument('-r',
                        '--random_seed',
                        type=int,
                        default=1989,
                        help="Random seed for reproducibility")
    parser.add_argument('--n_epochs_start_viz',
                        type=int,
                        default=0,
                        help="When to start vizualization")
    parser.add_argument('--n_epochs_to_viz',
                        type=int,
                        default=1,
                        help="Vizualize every N epochs")

    # [Model] training
    parser.add_argument('--n_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        help="Starting learning rate.")
    parser.add_argument('--model',
                        type=str,
                        default='meode',
                        choices=['meode', 'ode'],
                        help="Temporal model")
    parser.add_argument(
        '--z0_encoder',
        type=str,
        default='odernn',
        choices=['odernn', 'rnn'],
        help="Type of encoder to get z0 (initial point of ODE)")
    parser.add_argument('--latents',
                        type=int,
                        default=13,
                        help="Size of the initial latent state (z0)")

    # 1. Args define the derivative network (ode net) for recognition z0
    parser.add_argument('--rec-dims',
                        type=int,
                        default=20,
                        help="Dimensionality of the recognition model, used "
                        "to generate z0. In odernn: this the output of "
                        "gradnet; in rnn: this is output of GRUBlock")
    parser.add_argument('--rec-layers',
                        type=int,
                        default=1,
                        help="Number of layers in ODE func in recognition ODE")
    parser.add_argument(
        '-g',
        '--gru-units',
        type=int,
        default=100,
        help="Number of units per layer in each of GRU update networks "
        "in odernn")
    # 2. Args define the derivative network (ode net) for generating zt
    parser.add_argument('--gen-layers',
                        type=int,
                        default=1,
                        help="Number of layers in ODE func in generative ODE")
    # 3. For both, recognition of z0 and generation of zt
    parser.add_argument('-u',
                        '--units',
                        type=int,
                        default=100,
                        help="Number of units per layer in ODE func for both: "
                        "recognition of z0 and generation of zt.")

    # Mixed Effect related
    parser.add_argument('--me_dim',
                        type=int,
                        default=1,
                        help="Size of mixed effect projection")
    parser.add_argument('--n_z0',
                        type=int,
                        default=1,
                        help="Number of sampled initial points z0")
    parser.add_argument('--n_w',
                        type=int,
                        default=2,
                        help="Number of sampled w per z0")

    ## [Data] related
    parser.add_argument(
        '--dataset',
        type=str,
        default='toy',
        help="Dataset to load. Available: toy, hopper, rotmnist, adni")
    parser.add_argument('--n_samples',
                        type=int,
                        default=10**2,
                        help="Size of the dataset, used to generate data")
    parser.add_argument('--n_t',
                        type=int,
                        default=20,
                        help="Number of sampled time-points from the process")
    parser.add_argument(
        '--extrap',
        action='store_true',  #replace with store_true
        help="Extrapolate analysis to min_t and max_t")
    parser.add_argument(
        '--extrap_percent',
        type=float,
        default=0.3,
        help="Percentage of time points (from the right) to extrapolate model")

    ## [Data] Toy options
    parser.add_argument('--min_t',
                        type=float,
                        default=0.,
                        help="Toy data: t is in [args.min_t, args.max_t]")
    parser.add_argument('--max_t',
                        type=float,
                        default=3.,
                        help="Toy data: t is in [args.min_t, args.max_t]")
    ## [Data] Rotating MNIST options
    parser.add_argument(
        '--n_angles',
        type=int,
        default=4,
        help="Rotating MNIST/MuJoCo data: number of angles to rotate")
    parser.add_argument('--mnist_digit',
                        type=int,
                        default=3,
                        help="Rotating MNIST, specific digit")

    parser.add_argument('--device', type=int, default=1, help='Cuda device')
    parser.add_argument('--experimentID',
                        type=str,
                        default="0",
                        help='Experiment ID')
    parser.add_argument('--epochs_kl_anneal',
                        type=int,
                        default=1,
                        help='Number of iterations for linear KL schedule.')
    parser.add_argument('--epochs_until_kl_inc',
                        type=int,
                        default=1,
                        help='Number of epochs to burn in.')
    parser.add_argument('--epochs_encoder_only',
                        type=int,
                        default=-1,
                        help='Number of iterations for encdoer/decoder only. '
                        'Useful for 2d/3d datasets.')
    parser.add_argument('--test_only',
                        action='store_true',
                        help='Whether only to test, no training')

    ## Currently not used but necessary to define
    parser.add_argument(
        '-s',
        '--sample-tp',
        type=float,
        default=None,
        help="Number of time points to sub-sample."
        "If > 1, subsample exact number of points. If the number is in [0,1], "
        "take a percentage of available points per time series. "
        "If None, do not subsample"
    )  #for every trajectory different time-points are subsampled
    parser.add_argument(
        '-c',
        '--cut-tp',
        type=int,
        default=None,
        help="Cut out the section of the timeline of the specified length "
        "(in number of points). "
        "Used for periodic function demo.")

    return parser
