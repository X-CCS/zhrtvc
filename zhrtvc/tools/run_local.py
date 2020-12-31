#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/2/20
"""
"""
from pathlib import Path
from functools import partial
from multiprocessing.pool import Pool
from matplotlib import pyplot as plt
from tqdm import tqdm
import collections as clt
import os
import re
import json
import numpy as np
import shutil

import aukit
from aukit.audio_griffinlim import default_hparams, mel_spectrogram

# from hparams import hparams

my_hp = {
    "n_fft": 1024, "hop_size": 256, "win_size": 1024,
    "sample_rate": 22050, "max_abs_value": 4.0,
    "fmin": 0, "fmax": 8000,
    "preemphasize": True,
    'symmetric_mels': True,
}


# default_hparams.update(hparams.values())
# # default_hparams.update(my_hp)
#
# a = {(k, v) for k, v in hparams.values().items() if type(v) in {str, int, float, tuple, bool, type(None)}}
# b = {(k, v) for k, v in default_hparams.items() if type(v) in {str, int, float, tuple, bool, type(None)}}
# print(a - b)
# print(b - a)
#
# _pad_len = (default_hparams.n_fft - default_hparams.hop_size) // 2


def wavs2mels(indir: Path, outdir: Path):
    for fpath in tqdm(indir.glob("*.wav")):
        wav = aukit.load_wav(fpath, sr=16000)
        wav = np.pad(wav.flatten(), (_pad_len, _pad_len), mode="reflect")
        mel = mel_spectrogram(wav, default_hparams)
        np.save(outdir.joinpath(fpath.stem + ".npy"), mel, allow_pickle=False)


def get_train_files(indir: Path):
    others = []
    names = []
    for fpath in tqdm(sorted(indir.glob("**/*.wav"))):
        s = os.path.getsize(fpath)
        if s < 32000:
            print(s, fpath)
            others.append(fpath)
            continue
        name = "/".join(fpath.relative_to(indir).parts)
        names.append(name)

    with open(indir.joinpath("train_files.txt"), "w", encoding="utf8") as fout:
        for name in names:
            fout.write(name + "\n")


_hanzi_re = re.compile(r'[\u4E00-\u9FA5]')
_pause_aishell3_style_dict = {'#1': '%', '#2': '%', '#3': '$', '#4': '$'}
_pause_dict = {'#1': '#1', '#2': '#2', '#3': '#3', '#4': '#4'}


def convert_line(line):
    index, han_text, pny_text = line.strip().split('\t')
    pnys = pny_text.strip().split()
    parts = re.split(r'(#\d)', han_text)
    cnt = 0
    outs = []
    for part in parts:
        if part.startswith('#'):
            pny = _pause_dict[part]
            outs.append(pny)
        else:
            for zi in part:
                if _hanzi_re.search(zi):
                    if zi != '儿':
                        pny = pnys[cnt]
                        outs.append(pny)
                        cnt += 1
                    else:
                        if len(pnys) - 1 >= cnt and pnys[cnt].startswith('er'):
                            pny = pnys[cnt]
                            outs.append(pny)
                            cnt += 1
                else:
                    outs.append(zi)
    out_text = ' '.join(outs)
    # out_line = f'{index}|{out_text}|{han_text}\n'
    out_line = f'wav/biaobei/{index}.wav\t{out_text}\tbiaobei\n'
    return out_line


def biaobei2aishell3():
    """
    000085	现在是#2道儿#2越走#1越宽#3，人气#2越搞#1越旺#4。	xian4 zai4 shi4 daor4 yue4 zou3 yue4 kuan1 ren2 qi4 yue4 gao3 yue4 wang4
    """
    inpath = r'F:\bigdata\public_audio\bznsyp\metadata.csv'
    outpath = r'F:\bigdata\public_audio\bznsyp\train_biaobei.txt'

    with open(outpath, 'wt', encoding='utf8') as fout:
        for num, line in enumerate(tqdm(open(inpath, encoding='utf8'))):
            out_line = convert_line(line)
            fout.write(out_line)


def speaker_embedding():
    import hashlib
    out_32 = hashlib.md5('abc'.encode('utf8')).hexdigest()
    out = np.array([int(w, 16) + 1 for w in out_32]) / 16
    print(out)


def get_aishell_samples():
    """
    SSB00570415|fang4 yi4 shou3 % qing2 ge1 $|放一首%情歌$
    Returns:

    """
    indir = Path(r'E:\data\aishell3\AISHELL-3\AISHELL-3\train')
    txtpath = indir / 'label_train-set.txt'
    wavdir = indir / 'wav'
    idxdt = {w.split('|')[0]: w.strip() for w in open(txtpath) if not w.startswith('#')}
    outdir = indir / 'samples'
    outdir.mkdir(exist_ok=True)
    out_wavdir = outdir / 'wav'
    out_wavdir.mkdir(exist_ok=True)
    with open(outdir.joinpath('metadata.csv'), 'wt', encoding='utf8') as fout:
        for spkdir in tqdm(wavdir.glob('*')):
            wavpath = np.random.choice(list(spkdir.glob('*.wav')))
            shutil.copyfile(wavpath, out_wavdir.joinpath(wavpath.name))
            fout.write(f'wav/{wavpath.name}\t{idxdt[wavpath.stem]}\t{spkdir.name}\n')


if __name__ == "__main__":
    print(__file__)
    indir = Path(r"E:\lab\melgan\data\aliexamples")
    outdir = Path(r"E:\lab\melgan\data\aliexamples_mel")
    get_aishell_samples()
    # outdir.mkdir(exist_ok=True)
    # wavs2mels(indir=indir, outdir=outdir)

    # indir = Path(r"E:\data\aliaudio\alijuzi")
    # # get_train_files(indir=indir)
    #
    # line = '000085	现在是#2道儿#2越走#1越宽#3，人气#2越搞#1越旺#4。	xian4 zai4 shi4 daor4 yue4 zou3 yue4 kuan1 ren2 qi4 yue4 gao3 yue4 wang4'
    # out = convert_line(line)
    # print(out)
    #
    # biaobei2aishell3()
    # speaker_embedding()
