# GST-Tacotron-Pytorch

A  PyTorch implementation of  [Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis](https://arxiv.org/abs/1803.09017)

![model](pic/model.png)

## Update
Add blizzard dataset support.

## Requirements

``` shell
pip3 install -r requirements.txt
```

## File structure

- `Hyperparameters.py` --- contain all hyperparameters
- `Network.py` --- encoder\decoder
- `Modules.py` --- some modules for tacotron
- `Loss.py` --- calculate loss
- `Data.py` --- load dataset
- `utils.py` --- some util function for loading and saving data
- `Synthesis.py` --- generate wav file

## How to train
- Download multispeaker dataset
- preprocess your data and write your `get_XX_data` function in `Data.py`
- Adjust hyperparameters  in `Hyperparameters.py`
- make a directory named `log` in the parent of parent directory of Tacotron code

```
--- log
|    |
|    --- log[log_number]
|
--- code
     |
     --- Tacotron
             |
             --- train.py
             |
             --- Network.py
             |
           ......
```

- run train.py

``` shell
python3 train.py [log_number] [dataset_size] [start_epoch]

[log_number]: the log directory number
[dataset_size]: int or all
[start_epoch]: which epoch start to train (0 if start from scratch )

for example:
python3 train.py 0 all 0
```

## How to generate wav

run`generate.py`, modify the `text`in `generate.py` before running

> only support Chinese

