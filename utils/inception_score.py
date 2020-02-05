import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm


class InceptionScoreEvaluator:

    def __init__(
            self,
            generator,
            device: torch.device,
            generator_iters: int = 125,
            generator_batch_size: int = 400,
            inception_batch_size: int = 32,
            resize: bool = True,
            splits: int = 10,
    ):
        self.generator = generator
        self.device = device
        self.generator_iters = generator_iters
        self.generator_batch_size = generator_batch_size
        self.inception_batch_size = inception_batch_size
        self.resize = resize
        self.splits = splits

        assert generator_batch_size > 0

        self.inception_model = None
        self.up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True).to(self.device)

    def get_score(self):
        sample_list = []
        for i in range(self.generator_iters):
            sample_list += self.generator(num=self.generator_batch_size)
        score = self.get_score_of_array(sample_list)
        return score

    def get_pred(self, x):
        with torch.no_grad():
            if self.resize:
                x = self.up(x)
            if self.inception_model is None:
                self.inception_model = self.load_inception_model()
            x = self.inception_model(x)
        return F.softmax(x, dim=-1).data.cpu().numpy()

    def load_inception_model(self):
        print('Loading Inception Model...')
        model = inception_v3(pretrained=True, transform_input=False).to(self.device)
        model.eval()
        print('Finished.')
        return model

    def get_score_of_array(self, imgs: np.array):
        """
            Computes the inception score of the generated images imgs
            imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
            cuda -- whether or not to run on GPU
            batch_size -- batch size for feeding into Inception v3
            splits -- number of splits
        """
        N = len(imgs)
        assert N > self.inception_batch_size
        dataloader = torch.utils.data.DataLoader(imgs, batch_size=self.inception_batch_size)

        # Get predictions
        preds = np.zeros((N, 1000))
        for i, batch in tqdm(enumerate(dataloader, 0), total=N):
            batch = batch.to(self.device)
            batch_size_i = batch.size()[0]
            preds[i * self.inception_batch_size: i * self.inception_batch_size + batch_size_i] = self.get_pred(batch)

        # Now compute the mean kl-div
        split_scores = []
        for k in range(self.splits):
            part = preds[k * (N // self.splits): (k+1) * (N // self.splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))
        return np.mean(split_scores), np.std(split_scores)
