import torch


class Hyperparameters():

    data = '../../../data/data_aishell'

    max_Ty = max_iter = 200

    # gpu = 2
    device = 'cuda:3'

    lr = 0.001
    batch_size = 32  # !!!
    num_epochs = 10000  # !!!
    eval_size = 1
    save_per_epoch = 1
    log_per_batch = 10
    log_dir = '../../log/train{}'

    model_path = None
    optimizer_path = None

    eval_text = '''er2 dui4 lou2 shi4 cheng2 jiao1 yi4 zhi4 zuo4 yong4 zui4 da4 de xian4 gou4'''

    lr_step = [500000, 1000000, 2000000]

    vocab = "PE abcdefghijklmnopqrstuvwxyz1234'.?"  # P: Padding E: End of Sentence
    char2idx = {char: idx for idx, char in enumerate(vocab)}

    E = 256
    K = 16
    decoder_K = 8
    embedded_size = E
    dropout_p = 0.5
    num_banks = 15
    num_highways = 4

    sr = 16000  # Sample rate.
    n_fft = 2048  # fft points (samples)
    frame_shift = 0.0125  # seconds
    frame_length = 0.05  # seconds
    hop_length = int(sr * frame_shift)  # samples.
    win_length = int(sr * frame_length)  # samples.
    n_mels = 80  # Number of Mel banks to generate
    power = 1.2  # Exponent for amplifying the predicted magnitude
    n_iter = 50  # Number of inversion iterations
    preemphasis = .97  # or None
    max_db = 100
    ref_db = 20

    n_priority_freq = int(3000 / (sr * 0.5) * (n_fft / 2))

    r = 5

    use_gpu = torch.cuda.is_available()


if __name__ == '__main__':
    print(Hyperparameters.char2idx['E'])
