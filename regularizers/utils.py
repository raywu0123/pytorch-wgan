import torch
from utils import epsilon


def naive_interpolation(real_images, fake_images, real_scores=None, fake_scores=None):
    alpha = torch.empty([len(real_images), 1, 1, 1]).uniform_().to(real_images.device)
    alpha = alpha.expand(*real_images.shape)
    interpolations = alpha * real_images + (1 - alpha) * fake_images.detach()
    interpolations = interpolations.clone().detach().requires_grad_(True)
    return interpolations


def topk_interpolation(real_images, real_scores, fake_images, fake_scores):
    N = len(real_images)
    real_fake_lipschitz_square = lipschitz_square(
        real_images, real_scores,
        fake_images, fake_scores,
    ).view(-1)  # (N ** 2,)
    _, topk_indices = real_fake_lipschitz_square.topk(k=N)
    topk_real_indicies, topk_fake_indices = topk_indices // N, topk_indices % N
    topk_real_images, topk_fake_images = real_images[topk_real_indicies], fake_images[topk_fake_indices]
    return naive_interpolation(real_images=topk_real_images, fake_images=topk_fake_images)


def lipschitz_square(real_images, real_scores, fake_images, fake_scores):
    real_images = real_images.view(len(real_images), -1)  # (M, D)
    real_scores = real_scores.view(len(real_scores), -1)  # (M, 1)
    fake_images = fake_images.view(len(fake_images), -1)  # (N, D)
    fake_scores = fake_scores.view(len(fake_scores), -1)  # (N, 1)

    pairwise_image_square_dist = pairwise_euclidean_square_distance(
        real_images, fake_images,
    )  # (M, N)
    pairwise_score_square_dist = pairwise_euclidean_square_distance(
        real_scores, fake_scores,
    )  # (M, N)
    lipschitz_square = (
        pairwise_score_square_dist / (pairwise_image_square_dist + epsilon)
    )
    return lipschitz_square


def pairwise_euclidean_square_distance(a: torch.Tensor, b: torch.Tensor):
    a_square = (a ** 2).sum(dim=-1).view(len(a), 1)
    b_square = (b ** 2).sum(dim=-1).view(1, len(b))
    square_dist = a_square + b_square - 2 * a @ b.T
    return square_dist.clamp_min(0.)


class InterpolationCalculator:

    options = {
        'real': lambda real_images, real_scores, **kwargs: (real_images, real_scores),
        'fake': lambda fake_images, fake_scores, **kwargs: (fake_images, fake_scores),
        'naive_inter': naive_interpolation,
        'topk_inter': topk_interpolation,
    }

    def __init__(self, option, D):
        self.option = option
        self.D = D

    @classmethod
    def keys(cls):
        return cls.options.keys()

    def __call__(self, **kwargs):
        ret = self.options[self.option](**kwargs)
        if isinstance(ret, tuple):
            return ret
        else:
            return ret, self.D(ret)
