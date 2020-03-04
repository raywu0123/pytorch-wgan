from torch import nn
from .base import RegularizerBase


class WeightClippingRegularizer(RegularizerBase):

    def __init__(self, D: nn.Module, **kwargs):
        super().__init__(D, **kwargs)
        assert self.clip_val > 0

    def __str__(self):
        return 'weight_clipping'

    def __call__(self, **kwargs):
        d_params = self.discriminator.parameters()
        for p in d_params:
            p.data.clamp_min(-self.clip_val, self.clip_val)

        return None
