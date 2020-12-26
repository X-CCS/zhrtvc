# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/9/22
"""
demo_cli
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__name__).stem)

import json
import torch
import numpy as np
import librosa
import yaml

from .model import Tacotron2, load_model
from .hparams import create_hparams, Dict2Obj
from .data_utils import transform_mel, transform_text, transform_f0, transform_embed, transform_speaker
from .data_utils import TextMelLoader
from .layers import TacotronSTFT

_model = None
_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_mellotron_model(model_path: Path, hparams_path='', device=None):
    """
    导入训练得到的checkpoint模型。
    """
    global _model
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if str(hparams_path).endswith('.yml'):
        with open(hparams_path, "r") as f:
            hparams = yaml.load(f, Loader=yaml.FullLoader)
    else:
        hparams = json.load(open(hparams_path, encoding='utf8'))
    hparams = Dict2Obj(hparams)
    _model = load_model(hparams).to(device).eval()
    _model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])


def load_mellotron_torch(model_path, device=None):
    """
    用torch.load直接导入模型文件，不需要导入模型代码。
    """
    global _model
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _model = torch.load(model_path, map_location=device)
    return _model


def is_loaded():
    """
    判断模型是否已经loaded。
    """
    global _model
    return _model is not None


def generate_mel(text, style, speaker, f0, **kwargs):
    """
    用语音合成模型把文本转为mel频谱。
    """
    global _model
    if not is_loaded():
        load_mellotron_torch(**kwargs)

    with torch.no_grad():
        mels, mels_postnet, gates, alignments = _model.inference((text, style, speaker, f0))
        gates = torch.sigmoid(gates)
        alignments = alignments.permute(0, 2, 1)
        return mels, mels_postnet, gates, alignments


class MellotronSynthesizer():
    def __init__(self, model_path, speakers_path, hparams_path, texts_path, device=_device):
        self.device = device

        args_hparams = open(hparams_path, encoding='utf8').read()
        self.hparams = create_hparams(args_hparams)

        self.model = load_model(self.hparams).to(self.device).eval()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device)['state_dict'])

        self.speakers = json.load(open(speakers_path, encoding='utf8'))
        self.texts = [w.strip() for w in open(texts_path, encoding='utf8')]

        self.stft = TacotronSTFT(
            self.hparams.filter_length, self.hparams.hop_length, self.hparams.win_length,
            self.hparams.n_mel_channels, self.hparams.sampling_rate, self.hparams.mel_fmin,
            self.hparams.mel_fmax)

        self.dataloader = TextMelLoader(audiopaths_and_text=texts_path, hparams=self.hparams, speaker_ids=self.speakers)


    def synthesize(self, text, speaker, audio, with_show=False):
        text_data, mel_data, speaker_data, f0_data = self.dataloader.get_data_train_v2([audio, text, speaker])
        text_encoded = text_data[None, :].long().to(self.device)
        style_input = 0
        speaker_id = speaker_data.to(self.device)
        pitch_contour = f0_data

        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = self.model.inference(
                (text_encoded, style_input, speaker_id, pitch_contour))

        out_mel = mel_outputs_postnet.data.cpu().numpy()[0]
        out_align = alignments.data.cpu().numpy()[0]
        out_gate = torch.sigmoid(gate_outputs.data).cpu().numpy()[0]

        end_idx = np.argmax(out_gate > 0.2) or out_gate.shape[0]

        out_mel = out_mel[:, :end_idx]
        out_align = out_align.T[:, :end_idx]
        out_gate = out_gate[:end_idx]
        return (out_mel, out_align, out_gate) if with_show else out_mel


def save_model(model: MellotronSynthesizer, outpath=''):
    torch.save(model.model, outpath)
