import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Pytorch implementation of GAN models.")

    parser.add_argument('--model', type=str, default='DCGAN', choices=['GAN', 'DCGAN', 'WGAN-CP', 'WGAN-GP'])
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'cifar', 'stl10'],
                            help='The name of dataset')
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--iters', type=int, default=40000, help='The number of iterations for generator in WGAN model.')
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--eval_freq', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    return check_args(parser.parse_args())


# Checking arguments
def check_args(args):
    # --epoch
    try:
        assert args.epochs >= 1
    except:
        print('Number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('Batch size must be larger than or equal to one')

    args.channels = 3
    return args
