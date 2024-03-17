from torch.nn import Module
from torch import norm,pow,roll,squeeze,Tensor
import torch
from librosa import feature
import random
import sys
"""
    Paper:ImportantAug: a data augmentation agent for speech
    https://arxiv.org/abs/2112.07156
"""
def SNR(audio,noise):
    """
    :param audio:pure sound       tensor          n*1
    :param noise:pure noise       tensor          m*1
    :return: signal to noise ratio
    """
    if not isinstance(audio,Tensor):
        print("audio or noise is not tensor in file DataProcess/DataArgument/IMPORTANTAUG")
        sys.exit(-1)
    else:
        audio = squeeze(audio) # convert to pressure
        audio = pow(audio,2)
        noise = squeeze(noise)
        noise = pow(noise,2)
    # attention: audio and noise with different length should be uniformed
    audio_len = audio.shape[-1]
    noise_len = noise.shape[-1]

    # if audio_len != noise_len: # it should be 1 dim
        # print("audio and noise with different shape,auto changed")
    S = torch.sum(audio,dim=-1)/audio_len
    N = torch.sum(noise,dim=-1)/noise_len

    return 10*torch.log10(S/N+1)


def Generator(audio_feature,threshold): # changeable
    # i do not think it is useful,but that's what it origin be
    """
    :param audio_feature:                           tensor          T*F
    :return: importance or saliency map of audio    tensor [0,1]    T*F
    """
    power = pow(audio_feature,2)
    map = power>threshold
    return map

def IMPORTANTAUG(audio,noise,roll_requirement = True,D = 3,sr = 22050):
    """
    :param audio:           tensor          n*1
    :param noise:           tensor          m*1
    :param roll_requirement:
    :param D: delta from range -D to +D     int
    :return:
    """
    audio = squeeze(audio)
    audio_len = audio.shape[-1]

    noise = squeeze(noise)
    noise_len = noise.shape[-1]

    audio = audio[0:min(audio_len,noise_len)]
    noise = noise[0:min(audio_len,noise_len)]

    audio = feature.chroma_stft(squeeze(audio).numpy(),sr=sr)
    noise = feature.chroma_stft(squeeze(noise).numpy(),sr=sr)

    map = Generator(audio) # importance map
    if roll_requirement == True:
        delta = random.randint(-D,D)
        map = roll(map,delta)
    snr = SNR(audio,noise)
    A  = pow(norm(audio)/(norm(noise) * pow(10,exponent=snr/10)),exponent=0.5)
    return audio+A*torch.mul(noise,map)


# class IMPORTANTAUG(Module):
#     def __init__(self):
#         super(IMPORTANTAUG, self).__init__()
#     def forward(self,input):
#         pass

if __name__ == '__main__':
    import torchaudio
    path = r"D:\DataBase\GIS\Normal\LMS_C1 Point1_+Z_1.wav"
    data ,sr= torchaudio.load(path)
    fea = feature.chroma_stft(squeeze(data).numpy(),sr=sr)