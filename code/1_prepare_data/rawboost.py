#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import copy
import random
import shutil
import librosa
import argparse
import numpy as np
import soundfile as sf

from tqdm import tqdm
from scipy import signal



def randRange(x1, x2, integer):
    y = np.random.uniform(low=x1, high=x2, size=(1,))
    if integer:
        y = int(y[0])
    return y


def normWav(x,always):
    if always:
        x = x/np.amax(abs(x))
    elif np.amax(abs(x)) > 1:
            x = x/np.amax(abs(x))
    return x


def genNotchCoeffs(nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs):
    b = 1
    for i in range(0, nBands):
        fc = randRange(minF,maxF,0);
        bw = randRange(minBW,maxBW,0);
        c = randRange(minCoeff,maxCoeff,1);
          
        if c/2 == int(c/2):
            c = c + 1
        f1 = fc - bw/2
        f2 = fc + bw/2
        if f1 <= 0:
            f1 = 1/1000
        if f2 >= fs/2:
            f2 =  fs/2-1/1000
        b = np.convolve(signal.firwin(c, [float(f1), float(f2)], window='hamming', fs=fs),b)

    G = randRange(minG,maxG,0); 
    _, h = signal.freqz(b, 1, fs=fs)    
    b = pow(10, G/20)*b/np.amax(abs(h))   
    return b


def filterFIR(x,b):
    N = b.shape[0] + 1
    xpad = np.pad(x, (0, N), 'constant')
    y = signal.lfilter(b, 1, xpad)
    y = y[int(N/2):int(y.shape[0]-N/2)]
    return y


# Linear and non-linear convolutive noise
def LnL_convolutive_noise(x,N_f,nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,minBiasLinNonLin,maxBiasLinNonLin,fs):
    y = [0] * x.shape[0]
    for i in range(0, N_f):
        if i == 1:
            minG = minG-minBiasLinNonLin;
            maxG = maxG-maxBiasLinNonLin;
        b = genNotchCoeffs(nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs)
        y = y + filterFIR(np.power(x, (i+1)),  b)     
    y = y - np.mean(y)
    y = normWav(y,0)
    return y


# Impulsive signal dependent noise
def ISD_additive_noise(x, P, g_sd):
    beta = randRange(0, P, 0)
    
    y = copy.deepcopy(x)
    x_len = x.shape[0]
    n = int(x_len*(beta/100))
    p = np.random.permutation(x_len)[:n]
    f_r= np.multiply(((2*np.random.rand(p.shape[0]))-1),((2*np.random.rand(p.shape[0]))-1))
    r = g_sd * x[p] * f_r
    y[p] = x[p] + r
    y = normWav(y,0)
    return y


# Stationary signal independent noise
def SSI_additive_noise(x,SNRmin,SNRmax,nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs):
    noise = np.random.normal(0, 1, x.shape[0])
    b = genNotchCoeffs(nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs)
    noise = filterFIR(noise, b)
    noise = normWav(noise,1)
    SNR = randRange(SNRmin, SNRmax, 0)
    noise = noise / np.linalg.norm(noise,2) * np.linalg.norm(x,2) / 10.0**(0.05 * SNR)
    x = x + noise
    return x



#--------------RawBoost data augmentation algorithms---------------------------##

def process_Rawboost_feature(feature, sr, args, algo):
    
    # Data process by Convolutive noise (1st algo)
    if algo == 1:
        feature = LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)
                            
    # Data process by Impulsive noise (2nd algo)
    elif algo==2:
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
                            
    # Data process by coloured additive noise (3rd algo)
    elif algo==3:
        
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)
    
    # Data process by all 3 algo. together in series (1+2+3)
    elif algo==4:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)  
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,
                args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)                 

    # Data process by 1st two algo. together in series (1+2)
    elif algo==5:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)                
                            

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo==6:  
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo==7: 
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 
   
    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo==8:
        
        feature1 =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature2=ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para=feature1+feature2
        feature=normWav(feature_para,0)  #normalized resultant waveform
 
    # original data without Rawboost processing           
    else:
        
        feature=feature
    
    return feature



def main(args):
    """
    Process the training, validation, and test datasets for the RawBoost system.

    Parameters:
    args: Command-line arguments for RawBoost processing.
    """
    data_labels = []

    for mode in ["train", "mix"]:
        # Read original training data
        with open(os.path.join(args.base_data_path, f"data16k/{mode}.csv"), "r") as f:
            f.readline()
            for line in f.readlines():
                id, path, real, fake = line.strip().split(",")
                data_labels.append((id, path, real, fake))

    random.shuffle(data_labels)

    # Rawboost processing
    options = range(9) # [0, 8]
    batch = len(data_labels) // len(options)

    rawboost_data = []
    with open(os.path.join(args.base_data_path, "data16k_rawboost/train.csv"), "w") as trf:
        trf.write("id,path,real,fake\n")
        for i in options:
            start = batch * i

            if i == len(options) - 1:
                current_batch = data_labels[start:]
            else:
                current_batch = data_labels[start:start+batch]

            for audio_name, audio_path, real, fake in tqdm(current_batch, desc=f"RawBoost mode={i}"):
                y, sr = librosa.load(audio_path, sr=None)
                y = process_Rawboost_feature(y, sr, args, i)

                output_path = os.path.join(args.base_data_path, f"data16k_rawboost/train/{audio_name}.wav")
                sf.write(output_path, y, sr, format="WAV")
                trf.write(f"{audio_name},{output_path},{real},{fake}\n")
                rawboost_data.append((audio_name, audio_path, real, fake))

    random.shuffle(rawboost_data)

    # split
    trn_sz = int(len(rawboost_data) * args.train_ratio)
    trn_data = rawboost_data[:trn_sz]
    vld_data = rawboost_data[trn_sz:]

    # train, valid
    for mode, dataset in zip(["train", "val"], [trn_data, vld_data]):
        with open(os.path.join(args.base_data_path, f"aasist/rawboost/{mode}.csv"), "w") as f:
            f.write("id,path,real,fake\n")
            for _id, _path, _real, _fake in dataset:
                f.write(f"{_id},{_path},{_real},{_fake}\n")

    # test
    shutil.copyfile(
        os.path.join(args.base_data_path, "data16k/test.csv"),
        os.path.join(args.base_data_path, "aasist/rawboost/test.csv")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RawBoost system')

    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                        help='Number of notch filters. The higher the number of bands, the more aggressive the distortion. [default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                        help='Minimum center frequency [Hz] of notch filter. [default=20]')
    parser.add_argument('--maxF', type=int, default=8000, 
                        help='Maximum center frequency [Hz] (<sr/2) of notch filter. [default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                        help='Minimum width [Hz] of filter. [default=100]')
    parser.add_argument('--maxBW', type=int, default=1000, 
                        help='Maximum width [Hz] of filter. [default=1000]')
    parser.add_argument('--minCoeff', type=int, default=10, 
                        help='Minimum filter coefficients. More coefficients result in a more ideal filter slope. [default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                        help='Maximum filter coefficients. More coefficients result in a more ideal filter slope. [default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                        help='Minimum gain factor of linear component. [default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                        help='Maximum gain factor of linear component. [default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                        help='Minimum gain difference between linear and non-linear components. [default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                        help='Maximum gain difference between linear and non-linear components. [default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                        help='Order of the (non-)linearity where N_f=1 refers only to linear components. [default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                        help='Maximum number of uniformly distributed samples in [%]. [default=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                        help='Gain parameter > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                        help='Minimum SNR value for colored additive noise. [default=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                        help='Maximum SNR value for colored additive noise. [default=40]')
    parser.add_argument('--base_data_path', type=str, default="/root/data/")
    parser.add_argument('--train_ratio', type=float, default=0.8)
    args = parser.parse_args()

    main(args)
