import torch.nn as nn
import torch
from Hyperparameters import Hyperparameters as hp


class TacotronLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, mels, mels_hat, mags, mags_hat):
        mel_loss = torch.mean(torch.abs(mels - mels_hat))
        mag_loss = torch.abs(mags - mags_hat)
        mag_loss = 0.5 * torch.mean(mag_loss) + 0.5 * torch.mean(mag_loss[:, :, :hp.n_priority_freq])
        return mel_loss, mag_loss
