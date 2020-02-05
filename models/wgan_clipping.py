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
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=1024),
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
        return x.view(-1, 1024*4*4)


class WGAN_CP:

    def __init__(self, args):
        print("WGAN_CP init model.")
        self.C = args.channels
        self.wandb = args.wandb
        self.device = get_device()
        self.G = Generator(args.channels).to(self.device)
        self.D = Discriminator(args.channels).to(self.device)

        # WGAN values from paper
        self.learning_rate = 0.00005

        self.batch_size = 64
        self.weight_clipping_limit = 0.01

        # WGAN with gradient clipping uses RMSprop instead of ADAM
        self.d_optimizer = optim.RMSprop(self.D.parameters(), lr=self.learning_rate)
        self.g_optimizer = optim.RMSprop(self.G.parameters(), lr=self.learning_rate)

        self.iters = args.iters
        self.critic_iter = 5

        self.log_freq = args.log_freq
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

            if n_iters % self.log_freq == 0:
                inception_score = self.IS_evaluator.get_score()
                logs = {**D_logs, **G_logs, 'IS-mean': inception_score[0], 'IS-std': inception_score[1]}
                print(f"iters {n_iters}: {json.dumps(logs)}")
                if self.wandb:
                    samples = self.generate_img(num=self.batch_size)
                    samples = np.asarray(samples).transpose([0, 2, 3, 1])
                    wandb.log({
                        **logs,
                        **{
                            f'fake_samples_{i}': wandb.Image(sample)
                            for i, sample in enumerate(samples)
                        },
                    }, step=n_iters)
                # self.save_model()

    def train_discriminator(self, data_generator, one, mone):
        for d_iter, (images, _) in zip(range(self.critic_iter), data_generator):
            # Clamp parameters to a range [-c, c], c=self.weight_clipping_limit
            for p in self.D.parameters():
                p.data.clamp_(-self.weight_clipping_limit, self.weight_clipping_limit)

            self.D.zero_grad()
            images = images.to(self.device)

            z = self.get_prior()
            d_loss_real = self.D(images)
            d_loss_real = d_loss_real.mean()
            d_loss_real.backward(mone)

            # Train with fake images
            fake_images = self.G(z)
            d_loss_fake = self.D(fake_images)
            d_loss_fake = d_loss_fake.mean()
            d_loss_fake.backward(one)
            self.d_optimizer.step()

        return {
            'd_loss_real': d_loss_real.item(),
            'd_loss_fake': d_loss_fake.item(),
            'd_loss': d_loss_real.item() - d_loss_fake.item(),
        }

    def train_generator(self, mone):
        self.G.zero_grad()
        z = self.get_prior()
        fake_images = self.G(z)
        g_loss = self.D(fake_images)
        g_loss = g_loss.mean().mean(0).view(1)
        g_loss.backward(mone)
        self.g_optimizer.step()
        return {'g_loss': g_loss.item()}

    def generate_img(self, num: int):
        z = self.get_prior()
        samples = self.G(z).data.cpu().numpy()[:num]
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
