from Network import Tacotron
from Data import SpeechDataset, collate_fn, get_eval_data

from Hyperparameters import Hyperparameters as hp
from Loss import TacotronLoss
from utils import spectrogram2wav

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel

from scipy.io.wavfile import write
from time import time
import matplotlib.pyplot as plt
import os
import sys
# import cv2


device = torch.device(hp.device)


def train(log_dir, dataset_size, start_epoch=0):
    # log directory
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(os.path.join(log_dir, 'state')):
        os.mkdir(os.path.join(log_dir, 'state'))
    if not os.path.exists(os.path.join(log_dir, 'wav')):
        os.mkdir(os.path.join(log_dir, 'wav'))
    if not os.path.exists(os.path.join(log_dir, 'state_opt')):
        os.mkdir(os.path.join(log_dir, 'state_opt'))
    if not os.path.exists(os.path.join(log_dir, 'attn')):
        os.mkdir(os.path.join(log_dir, 'attn'))
    if not os.path.exists(os.path.join(log_dir, 'test_wav')):
        os.mkdir(os.path.join(log_dir, 'test_wav'))

    f = open(os.path.join(log_dir, 'log{}.txt'.format(start_epoch)), 'w')

    msg = 'use {}'.format(hp.device)
    print(msg)
    f.write(msg + '\n')

    # load model
    model = Tacotron().cuda()#.to(device)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    if start_epoch != 0:
        model_path = os.path.join(log_dir, 'state', 'epoch{}.pt'.format(start_epoch))
        model.load_state_dict(torch.load(model_path))
        msg = 'Load model of' + model_path
    else:
        msg = 'New model'
    print(msg)
    f.write(msg + '\n')

    # load optimizer
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    if start_epoch != 0:
        opt_path = os.path.join(log_dir, 'state_opt', 'epoch{}.pt'.format(start_epoch))
        optimizer.load_state_dict(torch.load(opt_path))
        msg = 'Load optimizer of' + opt_path
    else:
        msg = 'New optimizer'
    print(msg)
    f.write(msg + '\n')

    # print('lr = {}'.format(hp.lr))

    model = model.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

    criterion = TacotronLoss()  # Loss

    # load data
    if dataset_size is None:
        train_dataset = SpeechDataset(r=slice(hp.eval_size, None))
    else:
        train_dataset = SpeechDataset(r=slice(hp.eval_size, hp.eval_size + dataset_size))

    train_loader = DataLoader(dataset=train_dataset, batch_size=hp.batch_size, collate_fn=collate_fn, num_workers=8, shuffle=True)

    num_train_data = len(train_dataset)
    total_step = hp.num_epochs * num_train_data // hp.batch_size
    start_step = start_epoch * num_train_data // hp.batch_size
    step = 0
    global_step = step + start_step
    prev = beg = int(time())

    for epoch in range(start_epoch + 1, hp.num_epochs):

        model.train(True)
        for i, batch in enumerate(train_loader):
            step += 1
            global_step += 1

            texts = batch['text'].to(device)
            mels = batch['mel'].to(device)
            mags = batch['mag'].to(device)

            optimizer.zero_grad()

            mels_input = mels[:, :-1, :]  # shift
            mels_input = mels_input[:, :, -hp.n_mels:]  # get last frame
            ref_mels = mels[:, 1:, :]

            mels_hat, mags_hat, _ = model(texts, mels_input, ref_mels)

            mel_loss, mag_loss = criterion(mels[:, 1:, :], mels_hat, mags, mags_hat)
            loss = mel_loss + mag_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)  # clip gradients
            optimizer.step()
            # scheduler.step()

            if global_step in hp.lr_step:
                optimizer = set_lr(optimizer, global_step, f)

            if (i + 1) % hp.log_per_batch == 0:
                now = int(time())
                use_time = now - prev
                # total_time = hp.num_epoch * (now - beg) * num_train_data // (hp.batch_size * (i + 1) + epoch * num_train_data)
                total_time = total_step * (now - beg) // step
                left_time = total_time - (now - beg)
                left_time_h = left_time // 3600
                left_time_m = left_time // 60 % 60
                msg = 'step: {}/{}, epoch: {}, batch {}, loss: {:.3f}, mel_loss: {:.3f}, mag_loss: {:.3f}, use_time: {}s, left_time: {}h {}m'
                msg = msg.format(global_step, total_step, epoch, i + 1, loss.item(), mel_loss.item(), mag_loss.item(), use_time, left_time_h, left_time_m)

                f.write(msg + '\n')
                print(msg)

                prev = now

        # save model, optimizer and evaluate
        if epoch % hp.save_per_epoch == 0 and epoch != 0:
            torch.save(model.state_dict(), os.path.join(log_dir, 'state/epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(), os.path.join(log_dir, 'state_opt/epoch{}.pt'.format(epoch)))
            msg = 'save model, optimizer in epoch{}'.format(epoch)
            f.write(msg + '\n')
            print(msg)

            model.eval()

            #for file in os.listdir(hp.ref_wav):
            wavfile = hp.ref_wav
            name, _ = os.path.splitext(hp.ref_wav.split('/')[-1])

            text, mel, ref_mels = get_eval_data(hp.eval_text, wavfile)
            text = text.to(device)
            mel = mel.to(device)
            ref_mels = ref_mels.to(device)

            mel_hat, mag_hat, attn = model(text, mel, ref_mels)

            mag_hat = mag_hat.squeeze().detach().cpu().numpy()
            attn = attn.squeeze().detach().cpu().numpy()
            plt.imshow(attn.T, cmap='hot', interpolation='nearest')
            plt.xlabel('Decoder Steps')
            plt.ylabel('Encoder Steps')
            fig_path = os.path.join(log_dir, 'attn/epoch{}-{}.png'.format(epoch, name))
            plt.savefig(fig_path, format='png')

            wav = spectrogram2wav(mag_hat)
            write(os.path.join(log_dir, 'wav/epoch{}-{}.wav'.format(epoch, name)), hp.sr, wav)

            msg = 'synthesis eval wav in epoch{} model'.format(epoch)
            print(msg)
            f.write(msg)

    msg = 'Training Finish !!!!'
    f.write(msg + '\n')
    print(msg)

    f.close()


def set_lr(optimizer, step, f):
    if step == 500000:
        msg = 'set lr = 0.0005'
        f.write(msg)
        print(msg)
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005

    elif step == 1000000:
        msg = 'set lr = 0.0003'
        f.write(msg)
        print(msg)
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0003

    elif step == 2000000:
        msg = 'set lr = 0.0001'
        f.write(msg)
        print(msg)
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    return optimizer


if __name__ == '__main__':
    argv = sys.argv
    log_number = int(argv[1])
    start_epoch = int(argv[3])
    if argv[2].lower() != 'all':
        dataset_size = int(argv[2])
    else:
        dataset_size = None
    train(hp.log_dir.format(log_number), dataset_size, start_epoch)
