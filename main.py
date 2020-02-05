import os
os.environ['PYTHONHASHSEED'] = str(0)
import random
random.seed(0)

import numpy as np
np.random.seed(0)
import torch
torch.random.manual_seed(0)

from dotenv import load_dotenv
load_dotenv('.env')
import wandb

from utils.config import parse_args
from utils.data_loader import get_data_loader
from models.gan import GAN
from models.dcgan import DCGAN_MODEL
from models.wgan_clipping import WGAN_CP
from models.wgan_gradient_penalty import WGAN_GP


def main(args):
    model = None
    if args.model == 'GAN':
        model = GAN(args)
    elif args.model == 'DCGAN':
        model = DCGAN_MODEL(args)
    elif args.model == 'WGAN-CP':
        model = WGAN_CP(args)
    elif args.model == 'WGAN-GP':
        model = WGAN_GP(args)
    else:
        print("Model type non-existing. Try again.")
        exit(-1)

    train_loader, test_loader = get_data_loader(args)

    wandb.init(config=args)
    model.train(train_loader)


if __name__ == '__main__':
    args = parse_args()
    main(args)
