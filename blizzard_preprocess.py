import sys
import numpy as np
from os.path import join


def main(blizzard_path, val_ratio=0.05):

    with open(join(blizzard_path, 'prompts.gui')) as prompts_file:
        lines = [l[:-1] for l in prompts_file]

    wav_paths = [
        join(blizzard_path, 'wavn', fname + '.wav')
        for fname in lines[::3]
    ]

    # Clean up the transcripts
    transcripts = lines[1::3]
    for i in range(len(transcripts)):
        t = transcripts[i]
        t = t.replace('@ ', '')
        t = t.replace('# ', '')
        t = t.replace('| ', '')
        t = t.lower()
        transcripts[i] = t

    # randomize
    zips = np.array(list(zip(wav_paths, transcripts)))
    np.random.shuffle(zips)
    wav_paths, transcripts = zips.T

    # split
    split_idx = int(round(val_ratio * len(wav_paths)))
    wav_paths_val = wav_paths[:split_idx]
    wav_paths_train = wav_paths[split_idx:]
    transcripts_val = transcripts[:split_idx]
    transcripts_train = transcripts[split_idx:]

    train_and_val = [
        ('bliz13_audio_text_train_filelist.txt', wav_paths_train, transcripts_train),
        ('bliz13_audio_text_val_filelist.txt', wav_paths_val, transcripts_val)
    ]
    for outfile, wavs, scripts in train_and_val:
        with open(join('filelists', outfile), 'w') as f:
            for wav, text in zip(wavs, scripts):
                f.write('|'.join([
                    wav,
                    text,
                ])+'\n')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('provide path to blizzard dir as first (and only) '
            'argument (the directory given should contain a "prompts.gui" file)')
        exit(1)
    blizpath = sys.argv[1]
    main(blizpath)
