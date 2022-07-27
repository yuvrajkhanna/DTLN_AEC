import os, fnmatch
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dense, LSTM, Dropout, \
    Lambda, Input, Multiply, Layer, Conv1D, Concatenate
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, \
    EarlyStopping, ModelCheckpoint
import tensorflow as tf
import soundfile as sf
import math
from wavinfo import WavInfoReader
from random import shuffle, seed
import numpy as np
import matplotlib.pyplot as plt
import librosa as lb
from scipy import signal
from scipy.signal import butter, lfilter

def rms(audio):
    v_rms = (audio ** 2).mean() ** 0.5
    return v_rms

def normalize(audio):
    norm_audio = audio / np.max(np.abs(audio))
    return norm_audio

def process_rir(rir):
    norm_Rir=normalize(rir)
    gain=np.random.uniform(-25,0)
    temp=add_gain(norm_Rir,gain)
    temp[np.argmax(temp,axis=0)]=1
    return temp

def add_gain(audio,gain):
    new_audio=audio*(10**(gain/20))
    return new_audio

def add_noise(signal,noise_SNR,spectral_shaping=False):
    RMS=math.sqrt(np.mean(signal**2))
    STD_n=RMS/(10**(noise_SNR/20))
    noise=np.random.normal(0, STD_n, signal.shape[0])
    if spectral_shaping:
        noise=random_spectral_shaping(noise)
    signal_noise = signal+noise
    return signal_noise

def add_delay(audio,fs,time):
    # time in ms
    delay=int(time*fs/1000)
    delayed_audio=np.zeros(np.shape(audio))
    delayed_audio[delay:]=audio[delay:]
    return delayed_audio

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def conv_RIR(audio,rir):
    signal_rev =signal.fftconvolve(audio,rir,mode="full")
    signal_rev = signal_rev[0 : audio.shape[0]]
    return signal_rev

def random_spectral_shaping(audio):
    r=np.random.uniform(-3/8,3/8,4)
    b=[1,r[0],r[1]]
    a=[1,r[2],r[3]]
    y = lfilter(b, a, audio)
    return y

def discard(audio,fs):
    z=np.random.uniform(0,1,1)
    start=np.random.randint(0,fs*2-2)
    if z<0.5:
        end=start+fs
        discarded_audio=audio
        discarded_audio[start:end]=0
    else:
        end=start+2*fs
        discarded_audio=audio
        discarded_audio[start:end]=0
    return discarded_audio,start,end

def add_signals(audio1,audio2,snr):
    # add 2 signals with given snr ratio
    rms1=rms(audio1)
    rms2=rms(audio2)
    scaler=rms1/(rms2*(10**(snr/20)))
    scaled_audio2=scaler*audio2
    y=audio1+scaled_audio2
    return y

def create_farend_signal(farend,rir,fs):
    ##################################################################################
    #create farend signal from farend speech and farend room impulse response 
    ##################################################################################    

    # The echo signal is convolved with IR of Far-end room.
    
    processed_rir=process_rir(rir)
    room_farend=conv_RIR(farend,processed_rir)
    
    # In 50% of the cases, a noise sample is added with an SNR randomly taken from a 
    # normal distribution with a mean 5 dB and standard deviation 10 dB to account for
    # a noisy far-end signal.
    
    z=np.random.uniform(0,1,1)
    if z>0.5:
        noise_SNR=np.random.normal(5,10,1)
        noisy_farend=add_noise(room_farend,noise_SNR)
    else:
        noisy_farend=room_farend
    
    return noisy_farend

def create_nearend_signal(nearend,rir,fs):
    ##################################################################################
    #create nearend signal from nearend speech and nearend room impulse response 
    ##################################################################################
    
    # The echo signal is convolved with IR of near-end room.
    processed_rir=process_rir(rir)
    room_nearend=conv_RIR(nearend,processed_rir)
    
    # Random spectral shaping for noise reduction is applied to the speech signal to 
    # increase robustness and model various transmission effects.
    
    spectral = random_spectral_shaping(room_nearend)
    
    # In 70% of the cases, a noise sample is added with an SNR randomly taken from a
    # normal distribution with a mean 5 dB and standard deviation 10 dB to account for
    # a noisy far-end signal. Random spectral shaping is also applied to the noise 
    # signal independently.
    
    z=np.random.uniform(0,1,1)
    if z>0.3:
        noise_SNR=np.random.normal(5,10,1)
        noisy_nearend=add_noise(room_nearend,noise_SNR,spectral_shaping = True)
    else:
        noisy_nearend=room_nearend
    
    return noisy_nearend

def create_echo_signal(farend,rir,fs):
    ##################################################################################
    #create echo signal from farend and room impulse response used in near end signal
    ##################################################################################
        
    # the previously created far-end signal is delayed by a random value between 10 
    # and 100 ms to simulate a processing and transmission delay
    
    delay=np.random.uniform(10,100,1)
    delayed_audio = add_delay(farend,fs,delay)
    
    # The delayed signal is filtered by a band-pass signal with a random lower cut-off
    # frequency between 100 and 400 Hz and a higher cut-off frequency between 6000 and
    # 7500 Hz.
    
    low=np.random.uniform(100,400,1)
    high=np.random.uniform(6000,7500,1)
    fil_audio=butter_bandpass_filter(delayed_audio,low,high,fs)
    
    # The echo signal is finally convolved with same IR as the near-end signal.
    
    processed_rir=process_rir(rir)
    echo=conv_RIR(fil_audio,processed_rir)
    return echo

def preprocess(nearend,farend,rir_nearend,rir_farend,fs):
    ##################################################################################
    #create inputs and output for the model
    ##################################################################################
    flag=0
    
    # In 5% of the cases, a near-end speech segment of random duration
    # is discarded to account for far-end-only scenarios
    z=np.random.uniform(0,1,1)
    if z<0.05:
        flag=1
        discarded_nearend_speech,start,end=discard(nearend,fs)
    else:
        discarded_nearend_speech=nearend
    
    # In 90% of the cases, the echo signal is added to the near-end speech with a 
    # speechto-echo ratio taken from a normal distribution with a 0 dB mean and 
    # standard deviation of 10 dB.. The echo signal as well as the far-end speech 
    # signal is applied with random spectral shaping.If no echo is applied, the far-end 
    # signal is set to zero or to low-level noise in the range between -70 and -120 dB 
    # RMS with random spectral shaping.All signals used as input to the model are 
    # subject to a random gain chosen from a uniform distribution ranging from -25 to 0 dB
    
    nearend_speech_sig = create_nearend_signal(discarded_nearend_speech,rir_nearend,fs)
    z=np.random.uniform(0,1,1)
    if z>0.1:
        farend_signal = create_farend_signal(farend,rir_farend,fs)
        echo = create_echo_signal(farend_signal,rir_farend,fs)
        farend_signal = random_spectral_shaping(farend_signal)
        echo = random_spectral_shaping(echo)
        ser = np.random.uniform(0,10,1)
        nearend_sig = add_signals(nearend_speech_sig,echo,ser)
        rand_gain = np.random.uniform(-25,0,1)
        #nearend_sig=add_gain(nearend_sig,rand_gain)
        #farend_signal=add_gain(nearend_sig,rand_gain)
        #discarded_nearend_speech=add_gain(nearend_sig,rand_gain)
        return nearend_sig,farend_signal,discarded_nearend_speech
    else:
        p=np.random.uniform(0,1,1)
        if p>0.5:
            farend_signal=np.zeros(nearend.shape[0])
            return nearend_speech_sig,farend_signal,discarded_nearend_speech
        else:
            STD_n_db = np.random.uniform(-70,-120,1)
            STD_n = 10**(STD_n_db/20)
            farend_signal = np.random.normal(0, STD_n, nearend.shape[0])
            farend_signal = random_spectral_shaping(farend_signal)
            return nearend_speech_sig,farend_signal,discarded_nearend_speech