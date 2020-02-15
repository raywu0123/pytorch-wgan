from yoctol_argparse import YoctolArgumentParser
from yoctol_argparse.actions import IdKwargs
from yoctol_argparse.types import path, int_in_range

from models import MODEL_HUB


def parse_args():
    parser = YoctolArgumentParser(description="Pytorch implementation of GAN models.")

    parser.add_argument(
        '--model',
        action=IdKwargs,
        id_choices=MODEL_HUB.keys(),
        split_token=',',
        default=IdKwargs.IdKwargsPair('WGAN-PR', dict(lambda_term=10., method='max')),
    )
    parser.add_argument(
        '--dataroot',
        type=path,
        default='./datasets',
        help='path to dataset',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='mnist',
        choices=['mnist', 'fashion-mnist', 'cifar', 'stl10'],
        help='The name of dataset',
    )
    parser.add_argument(
        '--epochs',
        type=int_in_range(minval=1),
        default=100,
        help='The number of epochs to run',
    )
    parser.add_argument(
        '--iters',
        type=int_in_range(minval=1),
        default=40000,
        help='The number of iterations for generator in WGAN model.',
    )
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--eval_freq', type=int, default=500)
    parser.add_argument(
        '--batch_size',
        type=int_in_range(minval=1),
        default=64,
        help='The size of batch',
    )
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--wandb', action='store_true')

    args = parser.parse_args()
    args.channels = 3
    return args

