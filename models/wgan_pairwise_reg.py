from functools import partial
import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from utils import (
    InceptionScoreEvaluator,
    infinite_batch_generator,
    get_device,
    get_prior,
    epsilon,
)


class Generator(torch.nn.Module):

    def __init__(self, channels):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=4, stride=2, padding=1))
        # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


class Discriminator(torch.nn.Module):

    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True))
        # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024 * 4 * 4)


class WGAN_PairwiseReg:

    def __init__(self, args):
        self.C = args.channels
        self.wandb = args.wandb
        self.eval_freq = args.eval_freq
        self.log_freq = args.log_freq
        self.device = get_device()

        print("WGAN_PairwiseReg init model.")
        self.G = Generator(args.channels).to(self.device)
        self.D = Discriminator(args.channels).to(self.device)

        # WGAN values from paper
        self.learning_rate = 1e-4
        self.b1 = 0.5
        self.b2 = 0.999
        self.batch_size = 64

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

        self.iters = args.iters

        self.critic_iter = 5
        self.lambda_term = 60
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
            fake_images.requires_grad_(True)
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

    def calculate_lipschitz_penalty(self, real_images, real_scores, fake_images, fake_scores):
        real_images = real_images.view(len(real_images), -1)  # (N, D)
        real_scores = real_scores.view(len(real_scores), -1)  # (N, 1)
        fake_images = fake_images.view(len(fake_images), -1)  # (N, D)
        fake_scores = fake_scores.view(len(fake_scores), -1)  # (N, 1)

        pairwise_image_square_dist = self.pairwise_euclidean_square_distance(
            real_images, fake_images,
        )  # (N, N)
        pairwise_score_square_dist = self.pairwise_euclidean_square_distance(
            real_scores, fake_scores,
        )  # (N, N)
        lipschitz_square = (
            pairwise_score_square_dist / (pairwise_image_square_dist + epsilon)
        )
        return lipschitz_square.mean() * self.lambda_term

    @staticmethod
    def pairwise_euclidean_square_distance(a: torch.Tensor, b: torch.Tensor):
        a_square = (a ** 2).sum(dim=-1).view(len(a), 1)
        b_square = (b ** 2).sum(dim=-1).view(1, len(b))
        square_dist = a_square + b_square - 2 * a @ b.T
        return square_dist

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
