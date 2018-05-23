from utils import *
from Data import text_normalize
from Hyperparameters import Hyperparameters as hp
import torch
from scipy.io.wavfile import write
from Network import *

import sys
import os
# import cv2


device = torch.device(hp.device)


def synthesis(log_number, epoch):
    log_dir = hp.log_dir.format(log_number)

    text = 'da4 jia1 hao3 wo3 lai2 zi4 zhe4 jiang1 da4 xue2 zhuan1 ye4 fang1 xiang4 shi4 ji4 suan4 ji1 ke1 xue2'
    text = text_normalize(text) + 'E'
    text = [hp.char2idx[c] for c in text]
    text = torch.Tensor(text).type(torch.LongTensor).unsqueeze(0).to(device)
    GO = torch.zeros(1, 1, hp.n_mels).to(device)

    model = Tacotron().to(device)

    model_path = log_dir + '/state/epoch{}.pt'.format(epoch)
    # model_path = '../../log/train9/state/epoch1600.pt'
    model.load_state_dict(torch.load(model_path))

    model.eval()

    mel_hat, mag_hat, attn = model(text, GO)

    mag_hat = mag_hat.squeeze().detach().cpu().numpy()
    attn = attn.squeeze().detach().cpu().numpy()

    plt.imshow(attn.T, cmap='hot', interpolation='nearest')
    plt.xlabel('Decoder Steps')
    plt.ylabel('Encoder Steps')
    fig_path = os.path.join(log_dir, 'test_wav/epoch{}.jpg'.format(epoch))
    plt.savefig(fig_path, format='png')

    wav_hat = spectrogram2wav(mag_hat)
    wav_path = os.path.join(log_dir, 'test_wav/epoch{}.wav'.format(epoch))
    # write('../../log/train9/test_wav/{}.wav'.format(i), hp.sr, wav)
    write(wav_path, hp.sr, wav_hat)
    print('synthesis ' + wav_path)


if __name__ == '__main__':
    argv = sys.argv
    log_number = int(argv[1])
    epoch = int(argv[2])
    print('start synthesis')
    synthesis(log_number, epoch)
    print('Done')
