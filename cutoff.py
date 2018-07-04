import wave
import numpy as np


def cutoff(input_wav, output_wav):
    '''
    input_wav --- input wav file path
    output_wav --- output wav file path
    '''

    # read input wave file and get parameters.
    with wave.open(input_wav, 'r') as fw:
        params = fw.getparams()
        # print(params)
        nchannels, sampwidth, framerate, nframes = params[:4]

        strData = fw.readframes(nframes)
        waveData = np.fromstring(strData, dtype=np.int16)

        max_v = np.max(abs(waveData))
        for i in range(waveData.shape[0]):
            if abs(waveData[i]) > 0.08 * max_v:
                break

        for j in range(waveData.shape[0] - 1, 0, -1):
            if abs(waveData[j]) > 0.08 * max_v:
                break

    # write new wav file
    with wave.open(output_wav, 'w') as fw:
        params = list(params)
        params[3] = nframes - i - (waveData.shape[0] - 1 - j)
        fw.setparams(params)
        fw.writeframes(strData[2 * i:2 * (j + 1)])


if __name__ == '__main__':
    cutoff('eval.wav', 'eval_cutoff.wav')
