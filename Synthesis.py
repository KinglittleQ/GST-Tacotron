from utils import *
from Data import get_eval_data
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

    model = Tacotron().to(device)
    model_path = log_dir + '/state/epoch{}.pt'.format(epoch)
    # model_path = '../../log/train9/state/epoch1600.pt'
    model.load_state_dict(torch.load(model_path))

    model.eval()

    ref_wavs = ['ref_wav/nannan.wav', 'ref_wav/xiaofeng.wav', 'ref_wav/donaldduck.wav']
    speakers = ['nannan', 'xiaofeng', 'donaldduck']
    for ref_wav, speaker in zip(ref_wavs, speakers):
        text, GO, ref_mels = get_eval_data(hp.eval_text, ref_wav)
        text = text.to(device)
        GO = GO.to(device)
        ref_mels = ref_mels.to(device)

        mel_hat, mag_hat, attn = model(text, GO, ref_mels)
        mag_hat = mag_hat.squeeze().detach().cpu().numpy()
        attn = attn.squeeze().detach().cpu().numpy()

        plt.imshow(attn.T, cmap='hot', interpolation='nearest')
        plt.xlabel('Decoder Steps')
        plt.ylabel('Encoder Steps')
        fig_path = os.path.join(log_dir, 'test_wav/epoch{}-{}.png'.format(epoch, speaker))
        plt.savefig(fig_path, format='png')

        wav_hat = spectrogram2wav(mag_hat)
        wav_path = os.path.join(log_dir, 'test_wav/epoch{}-{}.wav'.format(epoch, speaker))
        write(wav_path, hp.sr, wav_hat)
        print('synthesis ' + wav_path)


if __name__ == '__main__':
    argv = sys.argv
    log_number = int(argv[1])
    epoch = int(argv[2])
    print('start synthesis')
    synthesis(log_number, epoch)
    print('Done')
