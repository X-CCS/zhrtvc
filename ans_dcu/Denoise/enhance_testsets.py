import argparse
import os
import sys
sys.path.insert(0, '/exp')
import soundfile as sf
import numpy as np
import torch
from Denoise.exp import utils
from Denoise.models.unet import Unet
from Denoise.models.layers.istft import ISTFT
from torch.utils.data import DataLoader
import librosa
import numpy as np


window = torch.hann_window(1024).cuda()
def stft(x):
    return torch.stft(x, 1024, 256, window=window)
istft = ISTFT(1024, 256, window='hanning').cuda()


def denoisy(signal):

    json_path = os.path.join('/home/project/zhrtvc/ans_dcu/Denoise/exp/unet16.json')
    params = utils.Params(json_path)
    net = Unet(params.model).cuda()
    checkpoint = torch.load('/home/project/zhrtvc/ans_dcu/Denoise/ckpt/step00110.pth.tar')
    net.load_state_dict(checkpoint)
    torch.set_printoptions(precision=10, profile="full")
    
    
    maxt = len(signal)
    input_mat = np.zeros((1, maxt),dtype=np.float32)
    input_mat[0:,] = signal
    seq_len = np.zeros((1),dtype=np.int32)
    seq_len [0]= maxt
    with torch.no_grad():
        train_mixed = torch.from_numpy(input_mat).type(torch.FloatTensor)    
        seq_len = torch.from_numpy(seq_len).type(torch.IntTensor)  
        train_mixed = train_mixed.cuda()
        seq_len = seq_len.cuda()
        mixed = stft(train_mixed).unsqueeze(dim=1)
        real, imag = mixed[..., 0], mixed[..., 1]
        out_real, out_imag = net(real, imag)
        out_real, out_imag = torch.squeeze(out_real, 1), torch.squeeze(out_imag, 1)
        out_audio = istft(out_real, out_imag, train_mixed.size(1))
        out_audio = torch.squeeze(out_audio, dim=1)
        for i, l in enumerate(seq_len):
            out_audio[i, l:] = 0
            
        return (np.array(out_audio[0].cpu().data.numpy()[:seq_len[0].cpu().data.numpy()], dtype=np.float32))


