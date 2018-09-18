import torch


class Hyperparameters():

    data = '/d/blizzard/lessac_cathy5/wavn'
    #data = '../../../data/data_thchs30'

    max_Ty = max_iter = 200

    # gpu = 2
    device = 'cuda:0'
    # device = 'cpu'

    lr = 0.001
    batch_size = 16   # !!!
    num_epochs = 100  # !!!
    eval_size = 1
    save_per_epoch = 1
    log_per_batch = 20
    log_dir = './log/train{}'

    model_path = None
    optimizer_path = None

    # eval_text = '''er2 dui4 lou2 shi4 cheng2 jiao1 yi4 zhi4 zuo4 yong4 zui4 da4 de5 xian4 gou4'''
    #eval_text = '''chua1n pu3 zo3ng to3ng shuo1 ta1 ce2ng ji1ng xia4ng me3i guo2 re2n mi2n che2ng nuo4 jia1ng yo3u yi1 ge4 me3i ha3o de she4ng da4n li3 wu4  sui2 zhe zhe4 yi1 jia3n shui4 fa3 a4n to1ng guo4  ta1 ye3 dui4 xia4n le zhe4 yi1 che2ng nuo4'''
    eval_text = 'it took me a long time to develop a brain . now that i have it i\'m not going to be silent !'
    ref_wav = '/d/blizzard/lessac_cathy5/wavn/PP_309_093.wav'

    lr_step = [500000, 1000000, 2000000]

    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?"  # english
    #vocab = "PE abcdefghijklmnopqrstuvwxyz12345.?"  # chinese
    char2idx = {char: idx for idx, char in enumerate(vocab)}

    E = 256

    # reference encoder
    ref_enc_filters = [32, 32, 64, 64, 128, 128]
    ref_enc_size = [3, 3]
    ref_enc_strides = [2, 2]
    ref_enc_pad = [1, 1]
    ref_enc_gru_size = E // 2

    # style token layer
    token_num = 10
    # token_emb_size = 256
    num_heads = 8
    # multihead_attn_num_unit = 256
    # style_att_type = 'mlp_attention'
    # attn_normalize = True

    K = 16
    decoder_K = 8
    embedded_size = E
    dropout_p = 0.5
    num_banks = 15
    num_highways = 4

    # sr = 22050  # Sample rate.
    sr = 16000  # keda, thchs30, aishell
    n_fft = 1024  # fft points (samples) - ALE changed this from 2048
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
