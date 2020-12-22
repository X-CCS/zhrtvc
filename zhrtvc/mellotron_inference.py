#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/11/28
"""
mellotron_inference
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__name__).stem)

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--checkpoint_path', type=str,
                        default=r"../models/mellotron/samples/checkpoint/checkpoint-000000.pt",
                        help='模型路径。')
    parser.add_argument('-s', '--speakers_path', type=str,
                        default=r"../models/mellotron/samples/metadata/speakers.json",
                        help='发音人映射表路径。')
    parser.add_argument("-o", "--out_dir", type=Path, default=r"../models/mellotron/samples/test/000000",
                        help='保存合成的数据路径。')
    parser.add_argument("-p", "--play", type=int, default=0,
                        help='是否合成语音后自动播放语音。')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--hparams_path', type=str,
                        default=r"../models/mellotron/samples/metadata/hparams.json",
                        required=False, help='comma separated name=value pairs')
    parser.add_argument("-e", "--encoder_model_fpath", type=Path,
                        default=r"../models/encoder/saved_models/ge2e_pretrained.pt",
                        help="Path your trained encoder model.")
    parser.add_argument("--save_model_path", type=str, default=r"../models/mellotron/samples/mellotron-samples-000000.pt",
                        help='保存模型为可以直接torch.load的格式')
    parser.add_argument("--cuda", type=str, default='-1',
                        help='设置CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    return args


args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

import matplotlib.pyplot as plt
import aukit
import time
import json
import traceback
import torch
import numpy as np
import shutil
import re

import unidecode

from mellotron.inference import MellotronSynthesizer
from mellotron.inference import save_model
from utils.texthelper import xinqing_texts

aliaudio_fpaths = [str(w) for w in sorted(Path(r'../data/samples/aliaudio').glob('*/*.mp3'))]
filename_formatter_re = re.compile(r'[\s\\/:*?"<>|\']+')


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


if __name__ == "__main__":
    # args_hparams = open(args.hparams_path, encoding='utf8').read()
    # _hparams = create_hparams(args_hparams)
    #
    # model_path = args.checkpoint_path
    # load_model_mellotron(model_path, hparams=_hparams)

    ## Print some environment information (for debugging purposes)
    print("Running a test of your configuration...\n")
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
              "%.1fGb total memory.\n" %
              (torch.cuda.device_count(),
               device_id,
               gpu_properties.name,
               gpu_properties.major,
               gpu_properties.minor,
               gpu_properties.total_memory / 1e9))

    msyner = MellotronSynthesizer(model_path=args.checkpoint_path, speakers_path=args.speakers_path,
                                  hparams_path=args.hparams_path)

    if args.save_model_path:
        save_model(msyner, args.save_model_path)

    spec = msyner.synthesize(text='你好，欢迎使用语言合成服务。', speaker='speaker', audio=np.random.random(22050) * 2 - 1)

    ## Run a test


    print("Spectrogram shape: {}".format(spec.shape))
    # print("Alignment shape: {}".format(align.shape))
    wav_inputs = msyner.stft.griffin_lim(torch.from_numpy(spec[None]))
    wav = wav_inputs[0].cpu().numpy()
    print("Waveform shape: {}".format(wav.shape))

    print("All test passed! You can now synthesize speech.\n\n")

    print("Interactive generation loop")
    num_generated = 0
    args.out_dir.mkdir(exist_ok=True, parents=True)
    speaker_index_dict = json.load(open(args.speakers_path, encoding='utf8'))
    speaker_names = list(speaker_index_dict.keys())
    example_texts = xinqing_texts
    example_fpaths = aliaudio_fpaths
    while True:
        try:
            speaker = input("Speaker:\n")
            if not speaker.strip():
                speaker = np.random.choice(speaker_names)
            print('Speaker: {}'.format(speaker))

            text = input("Text:\n")
            if not text.strip():
                text = np.random.choice(example_texts)
            print('Text: {}'.format(text))

            audio = input("Audio:\n")
            if not audio.strip():
                audio = np.random.choice(aliaudio_fpaths)
            print('Audio: {}'.format(audio))

            # The synthesizer works in batch, so you need to put your data in a list or numpy array

            print("Creating the spectrogram ...")
            spec, align, gate = msyner.synthesize(text=text, speaker=speaker, audio=audio, with_show=True)

            print("Spectrogram shape: {}".format(spec.shape))
            print("Alignment shape: {}".format(align.shape))

            ## Generating the waveform
            print("Synthesizing the waveform ...")

            wav_outputs = msyner.stft.griffin_lim(torch.from_numpy(spec[None]), n_iters=30)
            wav_output = wav_outputs[0].cpu().numpy()

            print("Waveform shape: {}".format(wav.shape))

            # Save it on the disk
            cur_text = filename_formatter_re.sub('', unidecode.unidecode(text))[:15]
            cur_time = time.strftime('%Y%m%d-%H%M%S')
            out_path = args.out_dir.joinpath("demo_{}_{}_out.wav".format(cur_time, cur_text))
            aukit.save_wav(wav_output, out_path, sr=msyner.stft.sampling_rate)  # save

            ref_path = args.out_dir.joinpath("demo_{}_{}_ref.wav".format(cur_time, cur_text))
            shutil.copyfile(audio, out_path)

            fig_path = args.out_dir.joinpath("demo_{}_{}_fig.jpg".format(cur_time, cur_text))
            plot_mel_alignment_gate_audio(spec, align, gate, wav[::16])
            plt.savefig(fig_path)
            plt.close()

            txt_path = args.out_dir.joinpath("info_dict.txt".format(cur_time))
            with open(txt_path, 'at', encoding='utf8') as fout:
                dt = {k: (str(v) if isinstance(v, Path) else v) for k, v in locals().items()
                      if isinstance(v, (int, float, str, Path, bool))}
                out = json.dumps(dt, ensure_ascii=False)
                fout.write('{}\n'.format(out))

            num_generated += 1
            print("\nSaved output as %s\n\n" % out_path)
            if args.play:
                aukit.play_audio(out_path, sr=msyner.stft.sampling_rate)
        except Exception as e:
            print("Caught exception: %s" % repr(e))
            print("Restarting\n")
            traceback.print_exc()
