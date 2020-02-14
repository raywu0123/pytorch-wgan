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
from models import MODEL_HUB


def main(args):
    model = MODEL_HUB[args.model](args)
    train_loader, test_loader = get_data_loader(args)
    if args.wandb:
        wandb.init(config=args)
    model.train(train_loader)


if __name__ == '__main__':
    args = parse_args()
    main(args)
