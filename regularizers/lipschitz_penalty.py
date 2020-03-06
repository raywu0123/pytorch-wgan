from itertools import combinations

import torch
from torch import nn

from .base import RegularizerBase
from .utils import pairwise_euclidean_square_distance, InterpolationCalculator
from utils import epsilon


class LipschitzPenaltyRegularizer(RegularizerBase):

    def __init__(self, D: nn.Module, **kwargs):
        super().__init__(D, **kwargs)
        assert isinstance(self.type, str)
        assert self.lambd >= 0

        self.interpolation_calculator = InterpolationCalculator(self.inter, D)

    def __str__(self):
        return 'lipschitz_penalty'

    @staticmethod
    def _lipschitz_square(images_1, scores_1, images_2, scores_2):
        images_1 = images_1.view(len(images_1), -1)  # (N, D)
        scores_1 = scores_1.view(len(scores_1), -1)  # (N, 1)
        images_2 = images_2.view(len(images_2), -1)  # (N, D)
        scores_2 = scores_2.view(len(scores_2), -1)  # (N, 1)
        pairwise_image_square_dist = pairwise_euclidean_square_distance(
            images_1, images_2,
        )  # (N, N)
        pairwise_score_square_dist = pairwise_euclidean_square_distance(
            scores_1, scores_2,
        )  # (N, N)

        lipschitz_square = (
            pairwise_score_square_dist / (pairwise_image_square_dist + epsilon)
        )
        return lipschitz_square

    def __call__(self, real_images, real_scores, fake_images, fake_scores, **kwargs):
        inters = [
            self.interpolation_calculator(
                real_images=real_images, real_scores=real_scores,
                fake_images=fake_images, fake_scores=fake_scores,
            )
            for _ in range(self.n_inters)
        ]
        all_samples = [*inters, (real_images, real_scores), (fake_images, fake_scores)]
        lipschitz_square_list = [
            self._lipschitz_square(
                images_1, scores_1, images_2, scores_2
            )
            for (images_1, scores_1), (images_2, scores_2) in combinations(all_samples, 2)
        ]
        lipschitz_square = torch.cat(lipschitz_square_list)

        if self.type == 'mean':
            lipschitz_square = lipschitz_square.mean()
        elif self.type == 'max':
            lipschitz_square = lipschitz_square.max()
        else:
            raise ValueError('Invalid method')
        lipschitz_penalty = (torch.clamp_min(lipschitz_square, 1.) - 1.) * self.lambd
        lipschitz_penalty.backward(retain_graph=True)
        return lipschitz_penalty.item()
