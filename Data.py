from Hyperparameters import Hyperparameters as hp
from torch.utils.data import Dataset, DataLoader
import torch

from utils import *

import os
import unicodedata
import re


class SpeechDataset(Dataset):
    '''
    text: [T_x]
    mel: [T_y/r, n_mels*r]
    mag: [T_y, 1+n_fft/2]
    '''

    def __init__(self, r=slice(0, None)):
        print('Start loading data')
        # fpaths, texts = get_data(hp.data, r)  # thchs30
        # fpaths, texts = get_keda_data(hp.data, r)  # keda api
        # fpaths, texts = get_thchs30_data(hp.data, r)
        fpaths, texts = get_blizzard_data(hp.data, r)
        print('Finish loading data')
        self.fpaths = fpaths
        self.texts = texts

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, idx):
        _, mel, mag = load_spectrograms(self.fpaths[idx])
        mel = torch.from_numpy(mel)
        mag = torch.from_numpy(mag)
        GO_mel = torch.zeros(1, mel.size(1))  # GO frame
        mel = torch.cat([GO_mel, mel], dim=0)
        text = self.texts[idx]
        return {'text': text, 'mel': mel, 'mag': mag}


def collate_fn(batch):
    '''
    texts: [N, max_T_x]
    mels:  [N, max_T_y/r, n_mels*r]
    mags:  [N, max_T_y, 1+n_fft/2]
    '''

    texts = [d['text'] for d in batch]
    mels = [d['mel'] for d in batch]
    mags = [d['mag'] for d in batch]

    texts = pad_sequence(texts)
    mels = pad_sequence(mels)
    mags = pad_sequence(mags)

    return {'text': texts, 'mel': mels, 'mag': mags}


def text_normalize(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                   if unicodedata.category(char) != 'Mn')  # Strip accents

    text = text.lower()
    text = re.sub("[^{}]".format(hp.vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text


def pad_sequence(sequences):
    '''
    pad sequence to same length (max length)
    ------------------
    input:
        sequences --- a list of tensor with variable length
        out --- a tensor with max length
    '''
    lengths = [data.size(0) for data in sequences]
    batch_size = len(sequences)
    max_len = max(lengths)
    trailing_dims = sequences[0].size()[1:]
    out_dims = (batch_size, max_len) + trailing_dims
    dtype = sequences[0].data.type()
    out = torch.zeros(*out_dims).type(dtype)
    for i, data in enumerate(sequences):
        out[i, :lengths[i]] = data

    return out


def get_keda_data(dataset_dir, r):
    wav_paths = []
    texts = []

    wav_dirs = ['nannan', 'xiaofeng', 'donaldduck']
    csv_paths = ['transcript-nannan.csv', 'transcript-xiaofeng.csv', 'transcript-donaldduck.csv']
    for wav_dir, csv_path in zip(wav_dirs, csv_paths):
        csv = open(os.path.join(dataset_dir, csv_path), 'r')
        for line in csv.readlines():
            items = line.strip().split('|')
            wav_paths.append(os.path.join(dataset_dir, wav_dir, items[0] + '.wav'))
            text = text_normalize(items[1]) + 'E'
            text = [hp.char2idx[c] for c in text]
            text = torch.Tensor(text).type(torch.LongTensor)
            texts.append(text)
        csv.close()

    for wav in wav_paths[-20:]:
        print(wav)

    return wav_paths[r], texts[r]


def get_thchs30_data(dataset_dir, r):
    wav_paths = []
    text_paths = []

    data_dir = os.path.join(dataset_dir, 'data')
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        fname, ext = os.path.splitext(file_path)
        if ext == '.wav' and fname[-7:] != '_cutoff':
            wav_paths.append(fname + '_cutoff' + ext)
            text_paths.append(file_path + '.trn')

    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    dev_dir = os.path.join(dataset_dir, 'dev')

    for d in [train_dir, test_dir, dev_dir]:
        for file in os.listdir(d):
            file_path = os.path.join(d, file)
            fname, ext = os.path.splitext(file_path)
            if ext == '.wav' and fname[-7:] != '_cutoff':
                text_path = os.path.join(data_dir, file + '.trn')
                wav_paths.append(fname + '_cutoff' + ext)
                text_paths.append(text_path)

    for wav, txt in zip(wav_paths[-20:], text_paths[-20:]):
        print(wav, txt)

    texts = []
    for file in text_paths[r]:
        f = open(file, 'r', encoding='utf-8')
        text = f.readlines()[1].strip()
        text = text_normalize(text) + 'E'
        text = [hp.char2idx[c] for c in text]
        text = torch.Tensor(text).type(torch.LongTensor)
        texts.append(text)

    print(wav_paths[r][0], text_paths[r][0])

    return wav_paths[r], texts


def get_aishell_data(data_dir, r):
    path = os.path.join(data_dir, 'transcript.txt')
    data_dir = os.path.join(data_dir, 'wav', 'train')
    wav_paths = []
    texts = []
    with open(path, 'r') as f:
        for line in f.readlines():
            items = line.strip().split('|')
            wav_paths.append(os.path.join(data_dir, items[0] + '.wav'))
            text = items[1]
            text = text_normalize(text) + 'E'
            text = [hp.char2idx[c] for c in text]
            text = torch.Tensor(text).type(torch.LongTensor)
            texts.append(text)

    for wav, txt in zip(wav_paths[-20:], texts[-20:]):
        print(wav, txt)

    return wav_paths[r], texts[r]


def get_LJ_data(data_dir, r):
    path = os.path.join(data_dir, 'transcript.csv')
    data_dir = os.path.join(data_dir, 'wavs')
    wav_paths = []
    texts = []
    with open(path, 'r') as f:
        for line in f.readlines():
            items = line.strip().split('|')
            wav_paths.append(os.path.join(data_dir, items[0] + '.wav'))
            text = items[1]
            text = text_normalize(text) + 'E'
            text = [hp.char2idx[c] for c in text]
            text = torch.Tensor(text).type(torch.LongTensor)
            texts.append(text)

    for wav in wav_paths[-20:]:
        print(wav)

    return wav_paths[r], texts[r]

def get_blizzard_data(data_dir, r):
    file_list = './filelists/bliz13_audio_text_train_filelist.txt'

    texts = []
    wav_paths = []
    with open(file_list, 'r') as f:
        for line in f.readlines():
            wav_path, text = line.strip().split('|')
            wav_paths.append(os.path.join(data_dir, wav_path))

            text = text_normalize(text) + 'E'
            text = [hp.char2idx[c] for c in text]
            text = torch.Tensor(text).type(torch.LongTensor)
            texts.append(text)

    for wav in wav_paths[-20:]:
        print(wav)

    return wav_paths[r], texts[r]


def get_eval_data(text, wav_path):
    '''
    get data for eval
    --------------
    input:
        text --- pinyin format sequence
    output:
        text --- [1, T_x]
        mel ---  [1, 1, n_mels]
    '''
    text = text_normalize(text) + 'E'
    text = [hp.char2idx[c] for c in text]
    text = torch.Tensor(text).type(torch.LongTensor)  # [T_x]
    text = text.unsqueeze(0)  # [1, T_x]
    mel = torch.zeros(1, 1, hp.n_mels)  # GO frame [1, 1, n_mels]

    _, ref_mels, _ = load_spectrograms(wav_path)
    ref_mels = torch.from_numpy(ref_mels).unsqueeze(0)

    return text, mel, ref_mels


if __name__ == '__main__':
    dataset = LJDataset()
    loader = DataLoader(dataset=dataset, batch_size=8, collate_fn=collate_fn)

    for batch in loader:
        print(batch['text'][0])
        print(batch['mel'].size())
        print(batch['mag'].size())
        break
