from scipy.io import wavfile
from python_speech_features import mfcc
import os
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

extensions = ('.wav')
X = []
Y = []
for subdir, dirs, files in os.walk('Database_clean\\Testing'):
    for file in files:
        ext = os.path.splitext(file)[-1].lower()
        if ext in extensions:
            audio = os.path.join(subdir, file)
            (rate,sig) = wavfile.read(audio)
            mfcc_feat = mfcc(sig,rate,nfft=2048)
            mfc = np.mean(mfcc_feat,axis=0)
            X.append(mfc)
            folder = subdir[subdir.rfind('\\') + 1:]
            if folder=="AC OFF":
                Y.append(1)
            if folder=="AC ON":
                Y.append(2)
            if folder=="Door close":
                Y.append(3)
            if folder=="Door Open":
                Y.append(4)
                
X = np.array(X)
Y = np.array(Y)
print(X.shape)
print(Y.shape)

from sklearn.metrics import classification_report,accuracy_score

model = joblib.load('model.npy')
p = model.predict(X)
print(classification_report(p, Y))
print(accuracy_score(Y, p))