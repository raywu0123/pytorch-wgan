import torch
from torch import nn


def pairwise_euclidean_square_distance(a: torch.Tensor, b: torch.Tensor):
    a_square = (a ** 2).sum(dim=-1).view(len(a), 1)
    b_square = (b ** 2).sum(dim=-1).view(1, len(b))
    square_dist = a_square + b_square - 2 * a @ b.T
    return square_dist.clamp_min(0.)


def total_gradient_norm(params):
    total_norm = 0
    for p in params:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


class EMA(nn.Module):

    def __init__(self, decay_rate):
        super(EMA, self).__init__()
        self.decay_rate = decay_rate
        self.decay_rate_power = 1
        self.average = nn.Parameter(torch.tensor(1.), requires_grad=False)

    def forward(self, x):
        self.decay_rate_power *= self.decay_rate
        self.average.data = self.decay_rate * self.average + (1 - self.decay_rate) * x
        return self.average / (1 - self.decay_rate_power)
