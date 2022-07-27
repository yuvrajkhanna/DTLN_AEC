from DTLN_model import DTLN_model
import os
import librosa
import numpy as np
import soundfile as sf

farend='samples/farend_speech_fileid_0.wav'
nearend='samples/nearend_mic_fileid_0.wav'

farend_speech,fs = librosa.core.load(farend, sr=16000, mono=True)
nearend_signal,fs = librosa.core.load(nearend, sr=16000, mono=True)

modelClass = DTLN_model();

modelClass.build_DTLN_model(norm_stft=True)

modelClass.model.load_weights('weights/DTLN_model.h5')

farend_speech2=np.expand_dims(farend_speech,axis=0)
nearend_signal2=np.expand_dims(nearend_signal,axis=0)

print('Model Running ....')
out=modelClass.model.predict_on_batch([farend_speech2,nearend_signal2])

out = np.squeeze(out)

sf.write('samples/predicted_speech.wav', out,fs)

print('Done')