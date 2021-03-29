import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES']='2'
import numpy as np
import soundfile as sf
import librosa
import wave
from Denoise.enhance_testsets import denoisy

# parser = argparse.ArgumentParser()
# parser.add_argument('--inputwav', default='input/noisy.wav', help="ckpt dir")
# parser.add_argument('--outputwav', default='output/denoisy.wav', help="Directory containing params.json")
# args = parser.parse_args()


# def main():


#     noisywav, sr = librosa.load(args.inputwav,sr = 16000)  
#     denoisy2 = denoisy(noisywav)  
#     sf.write(data=denoisy2,file =args.outputwav, samplerate=16000)
# def noisy_processing(inputwav: str,outputwav: str,sampleRate: str):
def noisy_processing(inputwav: str,outputwav: str):

    f = wave.open(inputwav)
    SampleRate = f.getframerate() #获取输入音频的采样率

    noisywav, sr = librosa.load(inputwav,sr = SampleRate)  
    denoisy2 = denoisy(noisywav)  
    sf.write(data=denoisy2,file =outputwav, samplerate=SampleRate)
    print('返回信号处理完的音频为：',outputwav)
    return outputwav

if __name__ == '__main__':
    main()
