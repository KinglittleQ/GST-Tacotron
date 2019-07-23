import torch
import torch.nn as nn
from Modules import *
from GST import GST
from Hyperparameters import Hyperparameters as hp


class Tacotron(nn.Module):
    '''
    input:
        texts: [N, T_x]
        mels: [N, T_y/r, n_mels*r]
    output:
        mels --- [N, T_y/r, n_mels*r]
        mags --- [N, T_y, 1+n_fft//2]
        attn_weights --- [N, T_y/r, T_x]
    '''

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(hp.vocab), hp.E)
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.gst = GST()

    def forward(self, texts, mels, ref_mels):
        embedded = self.embedding(texts)  # [N, T_x, E]
        memory, encoder_hidden = self.encoder(embedded)  # [N, T_x, E]

        style_embed = self.gst(ref_mels)  # [N, 256]
        style_embed = style_embed.expand_as(memory)
        memory = memory + style_embed

        mels_hat, mags_hat, attn_weights = self.decoder(mels, memory)

        return mels_hat, mags_hat, attn_weights


class Encoder(nn.Module):
    '''
    input:
        inputs: [N, T_x, E]
    output:
        outputs: [N, T_x, E]
        hidden: [2, N, E//2]
    '''

    def __init__(self):
        super().__init__()
        self.prenet = PreNet(in_features=hp.E)  # [N, T, E//2]

        self.conv1d_bank = Conv1dBank(K=hp.K, in_channels=hp.E // 2, out_channels=hp.E // 2)  # [N, T, E//2 * K]

        self.conv1d_1 = Conv1d(in_channels=hp.K * hp.E // 2, out_channels=hp.E // 2, kernel_size=3)  # [N, T, E//2]
        self.conv1d_2 = Conv1d(in_channels=hp.E // 2, out_channels=hp.E // 2, kernel_size=3)  # [N, T, E//2]
        self.bn1 = BatchNorm1d(num_features=hp.E // 2)
        self.bn2 = BatchNorm1d(num_features=hp.E // 2)

        self.highways = nn.ModuleList()
        for i in range(hp.num_highways):
            self.highways.append(Highway(in_features=hp.E // 2, out_features=hp.E // 2))

        self.gru = nn.GRU(input_size=hp.E // 2, hidden_size=hp.E // 2, num_layers=2, bidirectional=True, batch_first=True)

    def forward(self, inputs, prev_hidden=None):
        # prenet
        inputs = self.prenet(inputs)  # [N, T, E//2]

        # CBHG
        # conv1d bank
        outputs = self.conv1d_bank(inputs)  # [N, T, E//2 * K]
        outputs = max_pool1d(outputs, kernel_size=2)  # [N, T, E//2 * K]

        # conv1d projections
        outputs = self.conv1d_1(outputs)  # [N, T, E//2]
        outputs = self.bn1(outputs)
        outputs = nn.functional.relu(outputs)  # [N, T, E//2]
        outputs = self.conv1d_2(outputs)  # [N, T, E//2]
        outputs = self.bn2(outputs)

        outputs = outputs + inputs  # residual connect

        # highway
        for layer in self.highways:
            outputs = layer(outputs)
            # outputs = nn.functional.relu(outputs)  # [N, T, E//2]

        # outputs = torch.transpose(outputs, 0, 1)  # [T, N, E//2]

        self.gru.flatten_parameters()
        outputs, hidden = self.gru(outputs, prev_hidden)  # outputs [N, T, E]

        return outputs, hidden


class Decoder(nn.Module):
    '''
    input:
        inputs --- [N, T_y/r, n_mels * r]
        memory --- [N, T_x, E]
    output:
        mels   --- [N, T_y/r, n_mels*r]
        mags --- [N, T_y, 1+n_fft//2]
        attn_weights --- [N, T_y/r, T_x]
    '''

    def __init__(self):
        super().__init__()
        self.prenet = PreNet(hp.n_mels)
        self.attn_rnn = AttentionRNN()
        self.attn_projection = nn.Linear(in_features=2 * hp.E, out_features=hp.E)
        self.gru1 = nn.GRU(input_size=hp.E, hidden_size=hp.E, batch_first=True, bidirectional=False)
        self.gru2 = nn.GRU(input_size=hp.E, hidden_size=hp.E, batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(in_features=hp.E, out_features=hp.n_mels * hp.r)
        self.cbhg = DecoderCBHG()  # Deng
        self.fc2 = nn.Linear(in_features=hp.E, out_features=1 + hp.n_fft // 2)  # Deng

    def forward(self, inputs, memory):
        if self.training:
            # prenet
            outputs = self.prenet(inputs)  # [N, T_y/r, E//2]

            attn_weights, outputs, attn_hidden = self.attn_rnn(outputs, memory)

            attn_apply = torch.bmm(attn_weights, memory)  # [N, T_y/r, E]
            attn_project = self.attn_projection(torch.cat([attn_apply, outputs], dim=2))  # [N, T_y/r, E]

            # GRU1
            self.gru1.flatten_parameters()
            outputs1, gru1_hidden = self.gru1(attn_project)  # outputs1--[N, T_y/r, E]  gru1_hidden--[1, N, E]
            gru_outputs1 = outputs1 + attn_project  # [N, T_y/r, E]
            # GRU2
            self.gru2.flatten_parameters()
            outputs2, gru2_hidden = self.gru2(gru_outputs1)  # outputs2--[N, T_y/r, E]  gru2_hidden--[1, N, E]
            gru_outputs2 = outputs2 + gru_outputs1

            # generate log melspectrogram
            mels = self.fc1(gru_outputs2)  # [N, T_y/r, n_mels*r]

            # CBHG
            out, cbhg_hidden = self.cbhg(mels)  # out -- [N, T_y, E]

            # generate linear spectrogram
            mags = self.fc2(out)  # out -- [N, T_y, 1+n_fft//2]

            return mels, mags, attn_weights

        else:
            # inputs = Go_frame  [1, 1, n_mels*r]
            attn_hidden = None
            gru1_hidden = None
            gru2_hidden = None

            mels = []
            mags = []
            attn_weights = []
            for i in range(hp.max_Ty):
                inputs = self.prenet(inputs)
                attn_weight, outputs, attn_hidden = self.attn_rnn(inputs, memory, attn_hidden)
                attn_weights.append(attn_weight)  # attn_weight: [1, 1, T_x]
                attn_apply = torch.bmm(attn_weight, memory)  # [1, 1, E]
                attn_project = self.attn_projection(torch.cat([attn_apply, outputs], dim=-1))  # [1, 1, E]

                # GRU1
                self.gru1.flatten_parameters()
                outputs1, gru1_hidden = self.gru1(attn_project, gru1_hidden)  # outputs1--[1, 1, E]  gru1_hidden--[1, 1, E]
                outputs1 = outputs1 + attn_project  # [1, T_y/r, E]
                # GRU2
                self.gru2.flatten_parameters()
                outputs2, gru2_hidden = self.gru2(outputs1, gru2_hidden)  # outputs2--[1, T_y/r, E]  gru2_hidden--[1, 1, E]
                outputs2 = outputs2 + outputs1

                # generate log melspectrogram
                mel = self.fc1(outputs2)  # [1, 1, n_mels*r]
                inputs = mel[:, :, -hp.n_mels:]  # get last frame
                mels.append(mel)

            mels = torch.cat(mels, dim=1)  # [1, max_iter, n_mels*r]
            attn_weights = torch.cat(attn_weights, dim=1)  # [1, T, T_x]

            out, cbhg_hidden = self.cbhg(mels)
            mags = self.fc2(out)

            return mels, mags, attn_weights


class DecoderCBHG(nn.Module):
    '''
    input:
        inputs: [N, T/r, n_mels * r]
    output:
        outputs: [N, T, E]
        hidden: [2, N, E//2]
    '''

    def __init__(self):
        super().__init__()

        self.conv1d_bank = Conv1dBank(K=hp.decoder_K, in_channels=hp.n_mels, out_channels=hp.E // 2)

        self.conv1d_1 = Conv1d(in_channels=hp.decoder_K * hp.E // 2, out_channels=hp.E, kernel_size=3)
        self.bn1 = BatchNorm1d(hp.E)
        self.conv1d_2 = Conv1d(in_channels=hp.E, out_channels=hp.n_mels, kernel_size=3)
        self.bn2 = BatchNorm1d(hp.n_mels)

        self.highways = nn.ModuleList()
        for i in range(hp.num_highways):
            self.highways.append(Highway(in_features=hp.n_mels, out_features=hp.n_mels))

        self.gru = nn.GRU(input_size=hp.n_mels, hidden_size=hp.E // 2, num_layers=2, bidirectional=True, batch_first=True)

    def forward(self, inputs, prev_hidden=None):
        inputs = inputs.view(inputs.size(0), -1, hp.n_mels)  # [N, T, n_mels]

        # conv1d bank
        outputs = self.conv1d_bank(inputs)  # [N, T, E//2 * K]
        outputs = max_pool1d(outputs, kernel_size=2)

        # conv1d projections
        outputs = self.conv1d_1(outputs)  # [N, T, E]
        outputs = self.bn1(outputs)
        outputs = nn.functional.relu(outputs)
        outputs = self.conv1d_2(outputs)  # [N, T, n_mels]
        outputs = self.bn2(outputs)

        outputs = outputs + inputs  # residual connect  [N, T, n_mels]

        # highway net
        for layer in self.highways:
            outputs = layer(outputs)  # [N, T, n_mels]

        # bidirection gru
        self.gru.flatten_parameters()
        outputs, hidden = self.gru(outputs, prev_hidden)  # outputs: [N, T, E]

        return outputs, hidden
