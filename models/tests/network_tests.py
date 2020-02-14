import torch

from models.wgan_gradient_penalty import Discriminator


def test_output_shape():
    inp = torch.ones([7, 3, 32, 32])
    D = Discriminator(channels=3)
    out = D(inp)
    assert out.shape == torch.Size([7, 1, 1, 1])
