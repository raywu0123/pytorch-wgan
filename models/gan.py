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


class GAN(object):

    def __init__(self, args):
        self.device = get_device()
        self.wandb = args.wandb
        self.log_freq = args.log_freq
        self.C = args.channels

        # Generator architecture
        self.G = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Tanh(),
        ).to(self.device)

        # Discriminator architecture
        self.D = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        ).to(self.device)

        # Binary cross entropy loss and optimizer
        self.loss = nn.BCELoss()
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0002, weight_decay=0.00001)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=0.0002, weight_decay=0.00001)

        # Set the logger
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
        for epoch in range(self.epochs + 1):
            for i, (images, _) in enumerate(train_loader):
                # Check if round number of batches
                if i == train_loader.dataset.__len__() // self.batch_size:
                    break

                # Flatten image 1,32x32 to 1024
                images = images.view(self.batch_size, -1)
                z = self.get_prior()

                real_labels = torch.ones(self.batch_size).to(self.device)
                fake_labels = torch.zeros(self.batch_size).to(self.device)

                # Train discriminator
                # compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
                # [Training discriminator = Maximizing discriminator being correct]
                outputs = self.D(images)
                d_loss_real = self.loss(outputs, real_labels)

                # Compute BCELoss using fake images
                fake_images = self.G(z)
                outputs = self.D(fake_images)
                d_loss_fake = self.loss(outputs, fake_labels)

                # Optimize discriminator
                d_loss = d_loss_real + d_loss_fake
                self.D.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Train generator
                z = self.get_prior()
                fake_images = self.G(z)
                outputs = self.D(fake_images)

                # We train G to maximize log(D(G(z))[maximize likelihood of discriminator being wrong] instead of
                # minimizing log(1-D(G(z)))[minizing likelihood of discriminator being correct]
                # From paper  [https://arxiv.org/pdf/1406.2661.pdf]
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