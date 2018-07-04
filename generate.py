from utils import *
from Data import get_eval_data
from Hyperparameters import Hyperparameters as hp
import torch
from scipy.io.wavfile import write
from Network import *

from pypinyin import lazy_pinyin, Style

device = torch.device('cpu')


def synthesis(model, eval_text):
    eval_text = _pinyin(eval_text)

    model.eval()

    # ref_wavs = [
    #     'ref_wav/nannan.wav', 'ref_wav/xiaofeng.wav', 'ref_wav/donaldduck.wav'
    # ]
    ref_wavs = [
        'ref_wav/nannan.wav',
        'ref_wav/xiaofeng.wav',
        'ref_wav/donaldduck.wav'
    ]
    speakers = ['nannan', 'xiaofeng', 'donaldduck']

    wavs = {}

    for ref_wav, speaker in zip(ref_wavs, speakers):
        text, GO, ref_mels = get_eval_data(eval_text, ref_wav)
        text = text.to(device)
        GO = GO.to(device)
        ref_mels = ref_mels.to(device)

        mel_hat, mag_hat, attn = model(text, GO, ref_mels)
        mag_hat = mag_hat.squeeze().detach().cpu().numpy()
        attn = attn.squeeze().detach().cpu().numpy()

        wav_hat = spectrogram2wav(mag_hat)
        wavs[speaker] = wav_hat

    return wavs


def load_model(checkpoint_path):
    model = Tacotron().to(device)
    model.load_state_dict(
        torch.load(
            checkpoint_path, map_location=lambda storage, location: storage))
    return model


def _pinyin(s):
    symbols = '0123456789abcdefghijklmnopqrstuvwxyz '
    s = lazy_pinyin(s, style=Style.TONE2)
    yin = []
    for token in s:
        if token != ' ':
            a = ''
            for c in token:
                if c in symbols:
                    a += c
            yin.append(a)
    a = ''
    s = ' '.join(yin)
    for i in range(len(s)):
        if s[i] == ' ' and i < len(s) - 1 and s[i + 1] == ' ':
            continue
        a += s[i]
    return a


if __name__ == '__main__':
    text = '''毛主席是中国的红太阳'''
    model = load_model('checkpoint/epoch100.pt')
    wavs = synthesis(model, text)
    for k in wavs:
        wav = wavs[k]
        write('samples/{}.wav'.format(k), hp.sr, wav)
