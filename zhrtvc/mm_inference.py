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
logger = logging.getLogger(Path(__file__).stem)

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--mellotron_model_fpath', type=str,
                        default=r"..\data\results\mellotron_tmp.pt",
                        help='模型路径。')
    parser.add_argument("-e", "--encoder_model_fpath", type=Path,
                        default=r"../models/encoder/saved_models/ge2e_pretrained.pt",
                        help="Path your trained encoder model.")
    parser.add_argument("-v", "--melgan_model_fpath", type=Path,
                        default=r"..\data\results\melgan_tmp.pt",
                        help="Path your trained encoder model.")
    parser.add_argument("-o", "--out_dir", type=Path, default=r"../results/test",
                        help='保存合成的数据路径。')
    parser.add_argument("-p", "--play", type=int, default=0,
                        help='是否合成语音后自动播放语音。')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument("--cuda", type=str, default='-1',
                        help='设置CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    return args


args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

import time
import json
import traceback
import torch
import numpy as np
import matplotlib.pyplot as plt
import aukit

from mellotron.data_utils import transform_text, transform_speaker
from utils.texthelper import xinqing_texts

_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def mellotron_inference(model, text, speaker, with_alignment=False):
    try:
        speaker_ids = {speaker: int(speaker)}
    except:
        speaker_ids = {}
    text_encoded = torch.LongTensor(transform_text(text, text_cleaners='hanzi'))[None, :].to(_device)
    speaker_id = torch.LongTensor(transform_speaker(speaker, speaker_ids=speaker_ids)).to(_device)

    style_input = 0
    pitch_contour = None
    with torch.no_grad():
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model.inference(
            (text_encoded, style_input, speaker_id, pitch_contour))
    out_mel = mel_outputs_postnet.data.cpu().numpy()[0]
    out_align = alignments.data.cpu().numpy()[0]
    out_gate = torch.sigmoid(gate_outputs.data).cpu().numpy()[0]

    end_idx = np.argmax(out_gate > 0.1) or out_gate.shape[0]

    out_mel = out_mel[:, :end_idx]
    out_align = out_align.T[:, :end_idx]
    out_gate = out_gate[:end_idx]
    return (out_mel, out_align, out_gate) if with_alignment else out_mel


def melgan_inference(model, spec):
    mel = aukit.linear2mel_spectrogram(spec)
    mel = torch.from_numpy(mel[None])
    with torch.no_grad():
        wav = model(mel.to(_device)).squeeze()
        return wav.cpu().numpy()


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


aliaudio_fpaths = [str(w) for w in sorted(Path(r'../data/samples/aliaudio').glob('*/*.mp3'))]

if __name__ == "__main__":
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

    mellotron_model = torch.load(args.mellotron_model_fpath, map_location=_device)
    melgan_model = torch.load(args.melgan_model_fpath, map_location=_device)

    spec = mellotron_inference(model=mellotron_model, text='你好，欢迎使用语言合成服务。', speaker='speaker')
    wav = melgan_inference(melgan_model, spec)

    ## Run a test
    # spec, align = synthesize_one('你好，欢迎使用语言合成服务。', aliaudio_fpaths[0], with_alignment=True,
    #                              hparams=_hparams, encoder_fpath=args.encoder_model_fpath)
    print("Spectrogram shape: {}".format(spec.shape))
    # print("Alignment shape: {}".format(align.shape))
    # wav = griffinlim_vocoder(spec)
    print("Waveform shape: {}".format(wav.shape))

    print("All test passed! You can now synthesize speech.\n\n")

    print("Interactive generation loop")
    num_generated = 0
    args.out_dir.mkdir(exist_ok=True, parents=True)
    speaker_index_dict = {'0': 0}  # json.load(open(args.speakers_path, encoding='utf8'))
    speaker_names = list(speaker_index_dict.keys())
    example_texts = xinqing_texts
    example_fpaths = aliaudio_fpaths
    while True:
        try:
            # Get the reference audio filepath
            speaker = input("Speaker:\n")
            if not speaker.strip():
                speaker = np.random.choice(speaker_names)
            print('Speaker: {}'.format(speaker))

            ## Generating the spectrogram
            text = input("Text:\n")
            if not text.strip():
                text = np.random.choice(example_texts)
            print('Text: {}'.format(text))
            # The synthesizer works in batch, so you need to put your data in a list or numpy array

            print("Creating the spectrogram ...")
            spec, align, gate = mellotron_inference(model=mellotron_model, text=text, speaker=speaker,
                                                    with_alignment=True)
            # spec, align = synthesize_one(text, speaker=speaker, with_alignment=True,
            #                              hparams=_hparams, encoder_fpath=args.encoder_model_fpath)
            print("Spectrogram shape: {}".format(spec.shape))
            print("Alignment shape: {}".format(align.shape))

            ## Generating the waveform
            print("Synthesizing the waveform ...")
            wav = melgan_inference(model=melgan_model, spec=spec)
            wav_tf = aukit.inv_linear_spectrogram(spec)
            print("Waveform shape: {}".format(wav.shape))

            # Save it on the disk
            cur_time = time.strftime('%Y%m%d_%H%M%S')
            fpath = args.out_dir.joinpath("demo_out_{}_melgan.wav".format(cur_time))
            outpath = fpath
            # librosa.output.write_wav(fpath, generated_wav.astype(np.float32), synthesizer.sample_rate)
            aukit.save_wav(wav, fpath, sr=16000)  # save

            fpath = args.out_dir.joinpath("demo_out_{}_griffinlim.wav".format(cur_time))
            aukit.save_wav(wav, fpath, sr=16000)  # save

            fpath = args.out_dir.joinpath("demo_out_{}_figure.jpg".format(cur_time))
            plot_mel_alignment_gate_audio(spec, align, gate, wav[::16])
            plt.savefig(fpath)
            plt.close()

            txt_path = args.out_dir.joinpath("info_dict.txt".format(cur_time))
            with open(txt_path, 'at', encoding='utf8') as fout:
                dt = dict(text=text, audio_path=str(fpath), speaker=speaker, time=cur_time)
                out = json.dumps(dt, ensure_ascii=False)
                fout.write('{}\n'.format(out))

            num_generated += 1
            print("\nSaved output as %s\n\n" % outpath)
            if args.play:
                aukit.play_audio(fpath, sr=16000)
        except Exception as e:
            print("Caught exception: %s" % repr(e))
            print("Restarting\n")
            traceback.print_exc()
