import pyaudio
import wave
from scipy.io import wavfile
from python_speech_features import mfcc
import os
import numpy as np
import joblib
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "file.wav"
 
audio = pyaudio.PyAudio()
 
# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
print ("recording...")
frames = []
 
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print ("finished recording")
 
 
# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()
waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()
X=[]
audio = 'file.wav'
(rate,sig) = wavfile.read(audio)
mfcc_feat = mfcc(sig,rate,nfft=2048)
mfc = np.mean(mfcc_feat,axis=0)
X.append(mfc)
                
X = np.array(X)
print(X.shape)

from sklearn.metrics import classification_report,accuracy_score

model = joblib.load('model_name.npy')
folder = model.predict(X)
if folder==1:
    print("AC OFF")
if folder==2:
    print("AC ON")
if folder==3:
    print("Door close")
if folder==4:
    print("Door Open")