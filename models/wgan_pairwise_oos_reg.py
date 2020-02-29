from functools import partial
import os
import json

import numpy as np
import torch
import torch.optim as optim
from torch import autograd
import wandb

from utils import (
    InceptionScoreEvaluator,
    infinite_batch_generator,
    get_device,
    get_prior,
    epsilon,
)
from .utils import pairwise_euclidean_square_distance
from .wgan_gradient_penalty import Generator, Discriminator


class WGAN_PairwiseOutOfSupportReg:

    def __init__(self, args, lambda_term: int, interpolation: str, penalty: str):
        self.C = args.channels
        self.batch_size = args.batch_size
        self.wandb = args.wandb
        self.eval_freq = args.eval_freq
        self.log_freq = args.log_freq
        self.device = get_device()

        print("WGAN_PairwiseOutOfSupportReg init model.")
        self.G = Generator(args.channels).to(self.device)
        self.D = Discriminator(args.channels).to(self.device)

        # WGAN values from paper
        self.learning_rate = 1e-4
        self.b1 = 0.5
        self.b2 = 0.999

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

        self.iters = args.iters
        self.lambda_term = lambda_term
        self.interpolation = interpolation
        self.penalty = penalty

        self.critic_iter = 5
        self.IS_evaluator = InceptionScoreEvaluator(
            generator=self.generate_img,
            device=self.device,
        )
        self.get_prior = partial(
            get_prior,
            batch_size=self.batch_size, dim=100, device=self.device,
        )

    def train(self, train_loader):
        data_generator = infinite_batch_generator(train_loader)
        one = torch.tensor(1).float().to(self.device)
        mone = one * -1

        for n_iters in range(1, self.iters + 1, 1):
            D_logs = self.train_discriminator(data_generator, one, mone)
            G_logs = self.train_generator(mone)

            if n_iters % self.eval_freq == 0:
                inception_score = self.IS_evaluator.get_score()
                logs = {'IS-mean': inception_score[0], 'IS-std': inception_score[1]}
                print(f"iters {n_iters}: {json.dumps(logs)}")
                if self.wandb:
                    wandb.log(logs, step=n_iters)

            if n_iters % self.log_freq == 0:
                logs = {**D_logs, **G_logs}
                print(f"iters {n_iters}: {json.dumps(logs)}")
                if self.wandb:
                    samples = self.generate_img(num=10)
                    samples = np.asarray(samples).transpose([0, 2, 3, 1])
                    wandb.log({
                        **logs,
                        'fake_samples': wandb.Image(np.concatenate(samples)),
                    }, step=n_iters)
                # self.save_model()

    def train_discriminator(self, data_generator, one, mone):
        for d_iter, (images, _) in zip(range(self.critic_iter), data_generator):
            self.d_optimizer.zero_grad()
            images = images.to(self.device)
            d_score_real = self.D(images)
            d_score_real_mean = d_score_real.mean()
            d_score_real_mean.backward(mone, retain_graph=True)

            z = self.get_prior()
            fake_images = self.G(z)
            d_score_fake = self.D(fake_images)
            d_score_fake_mean = d_score_fake.mean()
            d_score_fake_mean.backward(one, retain_graph=True)

            # Train with gradient penalty
            lipschitz_penalty = self.calculate_lipschitz_penalty(
                images, d_score_real, fake_images, d_score_fake,
            )
            lipschitz_penalty.backward()
            self.d_optimizer.step()

        return {
            'd_score_real': d_score_real_mean.item(),
            'd_score_fake': d_score_fake_mean.item(),
            'd_loss': d_score_fake_mean.item() - d_score_real_mean.item(),
            'lipschitz_penalty': lipschitz_penalty.item(),
        }

    def train_generator(self, mone):
        self.g_optimizer.zero_grad()
        z = self.get_prior()
        fake_images = self.G(z)
        g_loss = self.D(fake_images)
        g_loss = g_loss.mean()
        g_loss.backward(mone)
        self.g_optimizer.step()
        return {'g_loss': -g_loss.item()}

    @staticmethod
    def _lipschitz_square(real_images, real_scores, fake_images, fake_scores):
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
        return lipschitz_square

    def calculate_lipschitz_penalty(self, real_images, real_scores, fake_images, fake_scores):
        N = len(real_images)
        if self.interpolation == 'topk':
            real_fake_lipschitz_square = self._lipschitz_square(
                real_images, real_scores,
                fake_images, fake_scores,
            ).view(-1)  # (N ** 2,)
            _, topk_indices = real_fake_lipschitz_square.topk(k=N)
            topk_real_indicies, topk_fake_indices = topk_indices // N, topk_indices % N
            topk_real_images, topk_fake_images = real_images[topk_real_indicies], fake_images[topk_fake_indices]
        elif self.interpolation == 'naive':
            topk_real_images, topk_fake_images = real_images, fake_images
        else:
            raise ValueError('Unsupported interpolation method.')

        alphas = torch.empty([N, 1, 1, 1]).uniform_().to(self.device)
        interpolations = topk_real_images * alphas + topk_fake_images * (1 - alphas)
        interpolations = interpolations.clone().detach().requires_grad_(True)
        interpolation_scores = self.D(interpolations)

        if self.penalty == 'gradient':
            gradients = autograd.grad(
                outputs=interpolation_scores,
                inputs=interpolations,
                grad_outputs=torch.ones(interpolation_scores.size(), device=self.device),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            gradients = gradients.view(gradients.size(0), -1)
            penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        elif self.penalty == 'slope':
            inter_real_lipschitz_square = self._lipschitz_square(
                interpolations, interpolation_scores,
                real_images, real_scores,
            )  # (k, N)
            inter_fake_lipschitz_square = self._lipschitz_square(
                interpolations, interpolation_scores,
                fake_images.detach(), fake_scores,
            )  # (k, N)
            penalty = torch.cat([
                inter_real_lipschitz_square, inter_fake_lipschitz_square
            ]).mean()
        else:
            raise ValueError('Unsupported penalty method.')
        return penalty * self.lambda_term

    def generate_img(self, num: int):
        with torch.no_grad():
            z = get_prior(batch_size=num, dim=100, device=self.device)
            samples = self.G(z).data.cpu().numpy()
            generated_images = []
            for sample in samples:
                generated_images.append(sample.reshape(self.C, 32, 32))
        return generated_images

    def save_model(self):
        torch.save(self.G.state_dict(), './generator.pkl')
        torch.save(self.D.state_dict(), './discriminator.pkl')
        print('Models save to ./generator.pkl & ./discriminator.pkl ')

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))
