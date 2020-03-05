from functools import partial
import os
import json

import numpy as np
import torch
import torch.optim as optim
import wandb

from utils import (
    InceptionScoreEvaluator,
    infinite_batch_generator,
    get_device,
    get_prior,
    epsilon,
)
from .utils import EMA, pairwise_euclidean_square_distance
from .networks import Discriminator, Generator


class WGAN_PairwiseConstraint:

    def __init__(self, args, decay_rate: float, method: str, warmup_iters: int):
        self.C = args.channels
        self.batch_size = args.batch_size
        self.wandb = args.wandb
        self.eval_freq = args.eval_freq
        self.log_freq = args.log_freq
        self.device = get_device()

        print("WGAN_PairwiseConstraint init model.")
        self.G = Generator(args.channels).to(self.device)
        self.D = Discriminator(args.channels).to(self.device)
        self.lipschitz_ema = EMA(decay_rate=decay_rate).to(self.device)

        # WGAN values from paper
        self.learning_rate = 1e-4
        self.b1 = 0.5
        self.b2 = 0.999

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

        self.iters = args.iters
        self.method = method
        self.warmup_iters = warmup_iters

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
            if n_iters < self.warmup_iters:
                denom = 1.
            else:
                denom = self.lipschitz_ema.average + epsilon
            D_logs = self.train_discriminator(data_generator, one, mone, denom)
            G_logs = self.train_generator(mone, denom)

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

    def train_discriminator(self, data_generator, one, mone, denom):
        for d_iter, (images, _) in zip(range(self.critic_iter), data_generator):
            self.d_optimizer.zero_grad()
            images = images.to(self.device)
            d_score_real = self.D(images)
            d_hat_score_real_mean = d_score_real.mean() / denom
            d_hat_score_real_mean.backward(mone)

            z = self.get_prior()
            fake_images = self.G(z)
            fake_images.requires_grad_(True)
            d_score_fake = self.D(fake_images)
            d_hat_score_fake_mean = d_score_fake.mean() / denom
            d_hat_score_fake_mean.backward(one)
            self.d_optimizer.step()

            lipschitz = self.calculate_lipschitz(
                images, d_score_real,
                fake_images, d_score_fake,
            )
            self.lipschitz_ema(lipschitz)

        return {
            'd_score_real': d_hat_score_real_mean.item(),
            'd_score_fake': d_hat_score_fake_mean.item(),
            'd_loss': d_hat_score_fake_mean.item() - d_hat_score_real_mean.item(),
            'lipschitz': self.lipschitz_ema.average.item(),
        }

    def train_generator(self, mone, denom):
        self.g_optimizer.zero_grad()
        z = self.get_prior()
        fake_images = self.G(z)
        g_loss = self.D(fake_images)
        g_loss = g_loss.mean() / denom
        g_loss.backward(mone)
        self.g_optimizer.step()
        return {'g_loss': -g_loss.item()}

    def calculate_lipschitz(self, real_images, real_scores, fake_images, fake_scores):
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
        if self.method == 'mean':
            lipschitz_square = lipschitz_square.mean()
        elif self.method == 'max':
            lipschitz_square = lipschitz_square.max()
        else:
            raise ValueError('Invalid method')
        lipschitz = torch.sqrt(lipschitz_square).detach()
        return lipschitz

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
