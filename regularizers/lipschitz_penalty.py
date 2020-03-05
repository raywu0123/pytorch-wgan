import torch
from torch import nn

from .base import RegularizerBase
from .utils import pairwise_euclidean_square_distance
from utils import epsilon


class LipschitzPenaltyRegularizer(RegularizerBase):

    def __init__(self, D: nn.Module, **kwargs):
        super().__init__(D, **kwargs)
        assert isinstance(self.type, str)
        assert self.lambd >= 0

    def __str__(self):
        return 'lipschitz_penalty'

    def __call__(self, real_images, real_scores, fake_images, fake_scores, **kwargs):
        real_images = real_images.view(len(real_images), -1)  # (N, D)
        real_scores = real_scores.view(len(real_scores), -1)  # (N, 1)
        fake_images = fake_images.view(len(fake_images), -1)  # (N, D)
        fake_scores = fake_scores.view(len(fake_scores), -1)  # (N, 1)

        pairwise_image_square_dist = pairwise_euclidean_square_distance(
            real_images, fake_images,
        )  # (N, N)
        pairwise_score_square_dist = pairwise_euclidean_square_distance(
            real_scores, fake_scores,
        )  # (N, N)
        lipschitz_square = (
                pairwise_score_square_dist / (pairwise_image_square_dist + epsilon)
        )
        if self.type == 'mean':
            lipschitz_square = lipschitz_square.mean()
        elif self.type == 'max':
            lipschitz_square = lipschitz_square.max()
        else:
            raise ValueError('Invalid method')
        lipschitz_penalty = (torch.clamp_min(lipschitz_square, 1.) - 1.) * self.lambd
        lipschitz_penalty.backward(retain_graph=True)
        return lipschitz_penalty.item()
