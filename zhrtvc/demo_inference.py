#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/12/8
"""
demo_inference

整个batch的文本等控制数据的准备。
合成器推理单位batch的文本。
声码器推理单位batch的频谱。
如果模型没有load，则自动load。
保存日志和数据。
"""
from pathlib import Path
import logging
import argparse
import os
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__name__).stem)


def parse_args():
    parser = argparse.ArgumentParser(description='声音编码器、语音合成器和声码器推理')
    parser.add_argument('--mellotron_path', type=str, default=r"../models/mellotron/samples/mellotron-samples-000000.pt",  #'../models/mellotron/mellotron_samples_model.pt'
                        help='Mellotron model file path')
    parser.add_argument('--melgan_path', type=str, default='', help='MelGAN model file path')
    parser.add_argument('--waveglow_path', type=str, default='../models/waveglow/waveglow_v5_model.pt',
                        help='WaveGlow model file path')
    parser.add_argument('--device', type=str, default='', help='Use device to inference')
    parser.add_argument('--sampling_rate', type=int, default=22050, help='Input file path or text')
    parser.add_argument('--input', type=str, default='这里有很多金矿。\tbiaobei', help='Input file path or text')
    parser.add_argument('--output', type=str, default='../results/demo_inference', help='Output file path or dir')
    parser.add_argument("--cuda", type=str, default='0', help='Set CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    return args


args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

import re
import json
import shutil
import collections as clt
import functools
import multiprocessing as mp
import traceback
import tempfile

import time

import numpy as np
import pydub
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import phkit
import aukit
import unidecode
import yaml

from waveglow import inference as waveglow
from melgan import inference as melgan
from mellotron import inference as mellotron
from utils.argutils import locals2dict

_device = 'cuda' if torch.cuda.is_available() else 'cpu'
filename_formatter_re = re.compile(r'[\s\\/:*?"<>|\']+')


def process_one(kwargs: dict):
    try:
        kwargs['code'] = 'success'
        return kwargs
    except Exception as e:
        traceback.print_exc()
        kwargs['code'] = f'{e}'
        return kwargs


def run_process(n_proc=1, **kwargs):
    kwargs_lst = []
    for kw in tqdm(kwargs, 'kwargs', ncols=100):
        kwargs_lst.append(kw)

    if n_proc <= 1:
        with tempfile.TemporaryFile('w+t', encoding='utf8') as fout:
            for kw in tqdm(kwargs_lst, 'process-{}'.format(n_proc), ncols=100):
                outs = process_one(kw)
                for out in outs:
                    fout.write(f'{json.dumps(out, ensure_ascii=False)}\n')
    else:
        func = functools.partial(process_one)
        job = mp.Pool(n_proc).imap(func, kwargs_lst)
        with tempfile.TemporaryFile('w+t', encoding='utf8') as fout:
            for outs in tqdm(job, 'process-{}'.format(n_proc), ncols=100, total=len(kwargs_lst)):
                for out in outs:
                    fout.write(f'{json.dumps(out, ensure_ascii=False)}\n')


def plot_mel_alignment_gate_audio(mel, alignment, gate, audio, figsize=(16, 16)):
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    axes = axes.flatten()
    axes[0].imshow(mel, aspect='auto', origin='bottom', interpolation='none')
    axes[1].imshow(alignment, aspect='auto', origin='bottom', interpolation='none')
    axes[2].scatter(range(len(gate)), gate, alpha=0.5, color='red', marker='.', s=1)
    axes[2].set_xlim(0, len(gate))
    axes[3].scatter(range(len(audio)), audio, alpha=0.5, color='blue', marker='.', s=1)
    axes[3].set_xlim(0, len(audio))

    axes[0].set_title("mel")
    axes[1].set_title("alignment")
    axes[2].set_title("gate")
    axes[3].set_title("audio")

    plt.tight_layout()


def load_models(args):
    if args.waveglow_path:
        waveglow.load_waveglow_torch(args.waveglow_path)
    if args.melgan_path:
        melgan.load_melgan_torch(args.melgan_path)
    if args.mellotron_path:
        mellotron.load_mellotron_torch(args.mellotron_path)


def transform_mellotron_input_data(text, style='', speaker='', f0='', device=''):
    if not device:
        device = _device
    text_data = torch.LongTensor(phkit.chinese_text_to_sequence(text, cleaner_names='hanzi'))[None, :].to(device)
    style_data = 0

    hex_idx = hashlib.md5(speaker.encode('utf8')).hexdigest()
    out = (np.array([int(w, 16) for w in hex_idx])[None] - 7) / 10  # -0.7~0.8
    speaker_data = torch.FloatTensor(out).to(device)
    # speaker_data = torch.zeros([1], dtype=torch.long).to(device)
    f0_data = None
    return text_data, style_data, speaker_data, f0_data


def hello():
    waveglow.load_waveglow_torch('../models/waveglow/waveglow_v5_model.pt')
    # melgan.load_melgan_model(r'E:\githup\zhrtvc\models\vocoder\saved_models\melgan\melgan_multi_speaker.pt',
    #                          args_path=r'E:\githup\zhrtvc\models\vocoder\saved_models\melgan\args.yml')
    melgan.load_melgan_torch('../models/melgan/melgan_multi_speaker_model.pt')

    # mellotron.load_mellotron_model(r'E:\githup\zhrtvc\models\mellotron\samples\checkpoint\checkpoint-000000.pt',
    #                                hparams_path=r'E:\githup\zhrtvc\models\mellotron\samples\metadata\hparams.yml')
    #
    # torch.save(mellotron._model, '../models/mellotron/mellotron_samples_model.pt')
    mellotron.load_mellotron_torch('../models/mellotron/mellotron_samples_model.pt')

    # text, mel, speaker, f0
    text = torch.randint(0, 100, [4, 50]).cuda()
    style = 0  # torch.rand(4, 80, 400).cuda()
    speaker = torch.randint(0, 10, [4]).cuda()
    f0 = None  # torch.rand(4, 400)

    mels = mellotron.generate_mel(text=text, style=style, speaker=speaker, f0=f0)

    for mel in mels:
        print(mel.shape)

    mel = torch.rand(4, 80, 400).cuda()

    wav = waveglow.generate_wave(mel)
    print(wav.shape)


if __name__ == "__main__":
    logger.info(__file__)

    if not args.device:
        _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        _device = args.device

    # 模型导入
    load_models(args)

    # 模型测试
    text_test = '这是个试水的例子。\tspeaker'
    text, speaker = text_test.split('\t')
    text_data, style_data, speaker_data, f0_data = transform_mellotron_input_data(text=text, speaker=speaker, device=_device)

    mels, mels_postnet, gates, alignments = mellotron.generate_mel(text_data, style_data, speaker_data, f0_data)

    wavs = waveglow.generate_wave(mel=mels)

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_output = wavs.squeeze().cpu().numpy()
        aukit.save_wav(wav_output, os.path.join(tmpdir, 'demo_example.wav'), sr=args.sampling_rate)

    # 模型推理

    if os.path.isfile(args.input):
        text_inputs = [w.strip() for w in open(args.input, encoding='utf8')]
    else:
        text_inputs = [args.input]

    output_dir = args.output
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    for text_input in tqdm(text_inputs, 'TTS', ncols=100):
        # print('Running: {}'.format(text_input))
        text, speaker = text_input.split('\t')
        text_data, style_data, speaker_data, f0_data = transform_mellotron_input_data(text=text_input, speaker=speaker, device=_device)

        mels, mels_postnet, gates, alignments = mellotron.generate_mel(text_data, style_data, speaker_data, f0_data)

        wavs = waveglow.generate_wave(mel=mels)

        # 保存数据
        cur_text = filename_formatter_re.sub('', unidecode.unidecode(text_input[:4]))
        cur_time = time.strftime('%Y%m%d-%H%M%S')
        outpath = os.path.join(output_dir, "demo_{}_{}_out.wav".format(cur_time, cur_text))

        wav_output = wavs.squeeze(0).cpu().numpy()
        aukit.save_wav(wav_output, outpath, sr=args.sampling_rate)

        fig_path = os.path.join(output_dir, "demo_{}_{}_fig.jpg".format(cur_time, cur_text))

        plot_mel_alignment_gate_audio(mel=mels_postnet.squeeze(0).cpu().numpy(),
                                      alignment=alignments.squeeze(0).cpu().numpy(),
                                      gate=gates.squeeze(0).cpu().numpy(),
                                      audio=wav_output[::args.sampling_rate // 1000])
        plt.savefig(fig_path)
        plt.close()

        yml_path = os.path.join(output_dir, "demo_{}_{}_info.yml".format(cur_time, cur_text))
        info_dict = locals2dict(locals())
        with open(yml_path, 'wt', encoding='utf8') as fout:
            yaml.dump(info_dict, fout, default_flow_style=False, encoding='utf-8', allow_unicode=True)

        log_path = os.path.join(output_dir, "info_dict.txt".format(cur_time))
        with open(log_path, 'at', encoding='utf8') as fout:
            dt = {k: (str(v) if isinstance(v, Path) else v) for k, v in locals().items()
                  if isinstance(v, (int, float, str, Path, bool))}
            out = json.dumps(dt, ensure_ascii=False)
            fout.write('{}\n'.format(out))
