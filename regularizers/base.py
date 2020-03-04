from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn


class RegularizerBase(ABC):

    def __init__(self, D: nn.Module, **kwargs):
        self.discriminator = D
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def __call__(self, **kwargs) -> Tuple[str, torch.Tensor]:
        pass
