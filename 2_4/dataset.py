import pandas as pd
import numpy as np
from pathlib import Path
import soundfile as sf

farend_speech_path='E:/intern/codes/dataset/farend_speech/'
nearend_speech_path='E:/intern/codes/dataset/nearend_speech/'
RIR_path='E:/intern/codes/dataset/RIR/'

fs=16000
time=4
len_of_samples=fs*time

farend_speechs=[]
for path in Path(farend_speech_path).rglob('*.wav'):
    farend_speechs.append(str(path.resolve()))
farend_speech_chunk_list=[]
for i in farend_speechs:
    audio,fs=sf.read(i)
    num_samples = int(np.fix(audio.shape[0]/len_of_samples))
    for j in range(num_samples):
        farend_speech_chunk_list.append(i+"{:02d}".format(j+1))

nearend_speechs=[]
for path in Path(nearend_speech_path).rglob('*.wav'):
    nearend_speechs.append(str(path.resolve()))
nearend_speech_chunk_list=[]
for i in nearend_speechs:
    audio,fs=sf.read(i)
    num_samples = int(np.fix(audio.shape[0]/len_of_samples))
    for j in range(num_samples):
        nearend_speech_chunk_list.append(i+"{:02d}".format(j+1))

Rirs=[]
for path in Path(RIR_path).rglob('*.wav'):
    Rirs.append(str(path.resolve()))
    
np.save('meta/farend_speech_chunk_list',farend_speech_chunk_list)
np.save('meta/nearend_speech_chunk_list',nearend_speech_chunk_list)
np.save('meta/Rirs',Rirs)