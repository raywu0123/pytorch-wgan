from .wgan_pairwise_reg import WGAN_PairwiseReg
from .wgan_pairwise_constraint import WGAN_PairwiseConstraint
from .dcgan import DCGAN_MODEL
from .gan import GAN

MODEL_HUB = {
    'GAN': GAN,
    'DCGAN': DCGAN_MODEL,
    'WGAN-PR': WGAN_PairwiseReg,
    'WGAN-PC': WGAN_PairwiseConstraint,
}
