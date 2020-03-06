from functools import partial
import os
import json
from typing import List

import numpy as np
import torch
from torch import nn
import torch.optim as optim
import wandb

from utils import (
    InceptionScoreEvaluator,
    get_device,
    get_prior,
)
from .utils import total_gradient_norm
from .networks import Generator, Discriminator
from regularizers.base import RegularizerBase


class WGAN_PairwiseReg:

    def __init__(self, args):
        self.C = args.channels
        self.batch_size = args.batch_size
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

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

        self.iters = args.iters

        self.critic_iter = 5
        self.IS_evaluator = InceptionScoreEvaluator(
            generator=self.generate_img,
            device=self.device,
        )
        self.get_prior = partial(
            get_prior,
            batch_size=self.batch_size, dim=100, device=self.device,
        )

    def train(self, data_generator, regularizers: List[RegularizerBase]):
        one = torch.tensor(1).float().to(self.device)
        mone = one * -1

        for n_iters in range(1, self.iters + 1, 1):
            D_logs = self.train_discriminator(data_generator, regularizers, one, mone)
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

    def train_discriminator(self, data_generator, regularizers: List[RegularizerBase], one, mone):
        for d_iter, (images, _) in zip(range(self.critic_iter), data_generator):
            self.d_optimizer.zero_grad()
            images = images.to(self.device)
            images.requires_grad_(True)
            d_score_real = self.D(images)
            d_score_real_mean = d_score_real.mean()
            d_score_real_mean.backward(mone, retain_graph=True)

            z = self.get_prior()
            z.requires_grad = True
            fake_images = self.G(z)
            d_score_fake = self.D(fake_images)
            d_score_fake_mean = d_score_fake.mean()
            fake_images.retain_grad()
            d_score_fake_mean.backward(one, retain_graph=True)
            fake_image_gradient_norm = fake_images.grad.norm(2).item()
            penalty_terms = {
                str(regularizer):
                regularizer(
                    real_images=images, real_scores=d_score_real,
                    fake_images=fake_images, fake_scores=d_score_fake,
                )
                for regularizer in regularizers
            }

            nn.utils.clip_grad_norm_(self.D.parameters(), 20.)
            self.d_optimizer.step()

        penalty_terms = {
            key: val for key, val in penalty_terms.items() if val is not None
        }
        total_param_gradient_norm = total_gradient_norm(self.D.parameters())
        return {
            'd_score_real': d_score_real_mean.item(),
            'd_score_fake': d_score_fake_mean.item(),
            'd_loss': d_score_fake_mean.item() - d_score_real_mean.item(),
            'd_total_param_gradient_norm': total_param_gradient_norm,
            'd_fake_image_gradient_norm': fake_image_gradient_norm,
            'd_fake_image_gradient_norm_with_penalty': fake_images.grad.norm(2).item(),
            **penalty_terms,
        }

    def train_generator(self, mone):
        self.g_optimizer.zero_grad()
        z = self.get_prior()
        fake_images = self.G(z)
        g_loss = self.D(fake_images)
        g_loss = g_loss.mean()
        fake_images.retain_grad()
        g_loss.backward(mone)
        self.g_optimizer.step()
        total_param_gradient_norm = total_gradient_norm(self.G.parameters())
        return {
            'g_loss': -g_loss.item(),
            'g_total_param_gradient_norm': total_param_gradient_norm,
            'g_fake_image_gradient_norm': fake_images.grad.norm(2).item(),
        }

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
