from .wgan_pairwise_reg import WGAN_PairwiseReg
from .wgan_pairwise_constraint import WGAN_PairwiseConstraint
from .wgan_gradient_penalty import WGAN_GP
from .wgan_clipping import WGAN_CP
from .dcgan import DCGAN_MODEL
from .gan import GAN

MODEL_HUB = {
    'GAN': GAN,
    'DCGAN': DCGAN_MODEL,
    'WGAN-CP': WGAN_CP,
    'WGAN-GP': WGAN_GP,
    'WGAN-PR': WGAN_PairwiseReg,
    'WGAN-PC': WGAN_PairwiseConstraint,
}
