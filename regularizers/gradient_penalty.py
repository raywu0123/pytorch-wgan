import torch
from torch import nn
from torch import autograd

from .base import RegularizerBase
from .utils import InterpolationCalculator


class GradientPenaltyRegularizer(RegularizerBase):

    def __init__(self, D: nn.Module, **kwargs):
        super().__init__(D, **kwargs)
        assert isinstance(self.type, str)
        assert self.center >= 0
        assert self.lambd >= 0
        self.interpolation_calculators = [
            InterpolationCalculator(pos, D)
            for pos in InterpolationCalculator.keys() if pos in self.type
        ]

    def __str__(self):
        return 'gradient_penalty'

    def __call__(self, real_images, real_scores, fake_images, fake_scores, **kwargs):
        penalties = []
        for interpolation_calculator in self.interpolation_calculators:
            samples, scores = interpolation_calculator(
                real_images=real_images, real_scores=real_scores,
                fake_images=fake_images, fake_scores=fake_scores,
            )
            gradients = autograd.grad(
                outputs=scores,
                inputs=samples,
                grad_outputs=torch.ones(scores.size(), device=fake_scores.device),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            gradients = gradients.view(gradients.size(0), -1)
            penalty = ((gradients.norm(2, dim=1) - self.center) ** 2).mean() * self.lambd
            penalties.append(penalty)

        penalty_mean = sum(penalties) / len(penalties)
        penalty_mean.backward(retain_graph=True)
        return penalty_mean.item()
