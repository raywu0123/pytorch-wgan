import torch
from torch.utils.data import DataLoader

from .inception_score import InceptionScoreEvaluator


def get_prior(batch_size: int, dim: int, device: torch.device):
    return torch.randn(batch_size, dim, 1, 1).to(device)


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    return device


def infinite_batch_generator(data_loader: DataLoader):
    while True:
        for data in data_loader:
            yield data
