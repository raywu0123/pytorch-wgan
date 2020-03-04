from torch import nn

from .weight_clipping import WeightClippingRegularizer
from .gradient_penalty import GradientPenaltyRegularizer
from .lipschitz_penalty import LipschitzPenaltyRegularizer


class RegularizerFactory:

    builder_fns = {
        'clipping': WeightClippingRegularizer,
        'gp': GradientPenaltyRegularizer,
        'lp': LipschitzPenaltyRegularizer,
    }

    def create_regularizers(self, D: nn.Module, regularizers: list):
        regularizers = [
            self.builder_fns[key](D, **value)
            for key, value in regularizers
        ]
        return regularizers
