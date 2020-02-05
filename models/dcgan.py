from functools import partial
import os
import json

import numpy as np
import torch
import torch.nn as nn
import wandb

from utils import (
    InceptionScoreEvaluator,
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
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True))
            # outptut of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0),
            # Output 1
            nn.Sigmoid())

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384 features
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)


class DCGAN_MODEL:

    def __init__(self, args):
        print("DCGAN model initialization.")
        self.device = get_device()
        self.wandb = args.wandb
        self.log_freq = args.log_freq
        self.C = args.channels
        self.G = Generator(args.channels).to(self.device)
        self.D = Discriminator(args.channels).to(self.device)

        # binary cross entropy loss and optimizer
        self.loss = nn.BCELoss()

        # Using lower learning rate than suggested by (ADAM authors)
        # lr=0.0002  and Beta_1 = 0.5 instead od 0.9 works better [Radford2015]
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.get_prior = partial(
            get_prior,
            batch_size=self.batch_size, dim=100, device=self.device,
        )
        self.IS_evaluator = InceptionScoreEvaluator(
            generator=self.generate_img,
            device=self.device,
        )

    def train(self, train_loader):
        generator_iter = 0
        for epoch in range(self.epochs):
            for i, (images, _) in enumerate(train_loader):
                # Check if round number of batches
                if i == train_loader.dataset.__len__() // self.batch_size:
                    break

                images = images.to(self.device)
                real_labels = torch.ones(self.batch_size).to(self.device)
                fake_labels = torch.zeros(self.batch_size).to(self.device)

                # Train discriminator
                # Compute BCE_Loss using real images
                outputs = self.D(images)
                d_loss_real = self.loss(outputs, real_labels)

                # Compute BCE Loss using fake images
                z = self.get_prior()
                fake_images = self.G(z)
                outputs = self.D(fake_images)
                d_loss_fake = self.loss(outputs, fake_labels)

                # Optimize discriminator
                d_loss = d_loss_real + d_loss_fake
                self.D.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Train generator
                # Compute loss with fake images
                z = self.get_prior()
                fake_images = self.G(z)
                outputs = self.D(fake_images)
                g_loss = self.loss(outputs, real_labels)

                # Optimize generator
                self.D.zero_grad()
                self.G.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()
                generator_iter += 1

                if generator_iter % self.log_freq == 0:
                    inception_score = self.IS_evaluator.get_score()
                    logs = {
                        'd_loss_real': d_loss_real.item(),
                        'd_loss_fake': d_loss_fake.item(),
                        'd_loss': d_loss.item(),
                        'g_loss': g_loss.item(),
                        'IS-mean': inception_score[0],
                        'IS-std': inception_score[1],
                    }
                    print(f"iters {generator_iter}: {json.dumps(logs)}")
                    if self.wandb:
                        samples = self.generate_img(num=self.batch_size)
                        samples = np.asarray(samples).transpose([0, 2, 3, 1])
                        wandb.log({
                            **logs,
                            **{
                                f'fake_samples_{i}': wandb.Image(sample)
                                for i, sample in enumerate(samples)
                            },
                        }, step=generator_iter)
                    # self.save_model()

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
