#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/12/7
"""
waveglow_inference
"""
from pathlib import Path
import logging
import argparse
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__name__).stem)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input_path", default='../data/samples/biaobei', type=str)
    parser.add_argument('-w', '--waveglow_path', default='../models/waveglow/waveglow_256channels_v4.pt',
                        type=str,
                        help='Path to waveglow decoder checkpoint with model')
    parser.add_argument('-o', "--output_path", default='../results/waveglow', type=str)
    parser.add_argument("-c", "--config_path", default='waveglow/config.json', type=str)
    parser.add_argument("--cuda", type=str, default='0', help='Set CUDA_VISIBLE_DEVICES')
    parser.add_argument("--save_model_path", type=str, default='', help='Save model for torch load')
    args = parser.parse_args()
    return args


args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

import json
from scipy.io import wavfile
import shutil

import torch
import librosa
from tqdm import tqdm

# from mellotron.layers import TacotronSTFT
from waveglow.mel2samp import MAX_WAV_VALUE, Mel2Samp, load_wav_to_torch


# 要把glow所在目录包含进来，否则导致glow缺失报错。

def main(input_path, waveglow_path, config_path, output_path, save_model_path):
    waveglow = torch.load(waveglow_path)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cuda().eval()
    if save_model_path:
        torch.save(waveglow, save_model_path)
    # waveglow = torch.load('../waveglow_v5_model.pt', map_location='cuda')
    with open(config_path) as f:
        data = f.read()
    data_config = json.loads(data)["data_config"]
    mel2samp = Mel2Samp(**data_config)

    if os.path.isfile(input_path) and input_path.endswith('txt'):
        audio_path_lst = [w.strip() for w in open(args.input, encoding='utf8')]
    elif os.path.isdir(input_path):
        audio_path_lst = [w for w in Path(input_path).glob('**/*') if w.is_file() and w.name.endswith(('mp3', 'wav'))]
    else:
        audio_path_lst = [input_path]

    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    for audio_path in tqdm(audio_path_lst, 'waveglow', ncols=100):
        audio_path = Path(audio_path)
        outpath = output_dir.joinpath(audio_path.name)
        name_cnt = 2
        while outpath.is_file():
            outpath = output_dir.joinpath(f'{audio_path.stem}-{name_cnt}{audio_path.suffix}')
            name_cnt += 1

        shutil.copyfile(audio_path, outpath)

        # 用mellotron的模块等价的方法生成频谱
        # audio_norm, sr = librosa.load(str(audio_path), sr=None)
        # audio_norm = torch.from_numpy(audio_norm).unsqueeze(0)
        # stft = TacotronSTFT(mel_fmax=8000.0)
        # audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        # mel = stft.mel_spectrogram(audio_norm)
        # mel = torch.autograd.Variable(mel.cuda())

        audio, sr = load_wav_to_torch(audio_path)
        mel = mel2samp.get_mel(audio)
        mel = torch.autograd.Variable(mel.cuda())
        mel = torch.unsqueeze(mel, 0)

        with torch.no_grad():
            audio = waveglow.infer(mel, sigma=1.0)
            audio = audio * MAX_WAV_VALUE
        audio = audio.squeeze()
        audio = audio.cpu().numpy()
        audio = audio.astype('int16')
        outpath = output_dir.joinpath(f'{outpath.name}.waveglow.wav')
        wavfile.write(outpath, data_config['sampling_rate'], audio)


if __name__ == "__main__":
    args = parse_args()
    main(input_path=args.input_path,
         waveglow_path=args.waveglow_path,
         output_path=args.output_path,
         config_path=args.config_path,
         save_model_path=args.save_model_path)
