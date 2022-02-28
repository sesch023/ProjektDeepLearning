import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

path = "../data/GTZAN Dataset/blues/blues.00095.wav"
out_test = "../out/test.wav"
n_mfcc = 40

y, sr = librosa.load(path)
mfcc_per_second = 10

duration = librosa.get_duration(y=y, sr=sr)
n_sample = y.shape[0]


S = librosa.feature.melspectrogram(y=y, sr=sr, fmax=8000)
mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=n_mfcc)

audio = librosa.feature.inverse.mfcc_to_audio(mfccs)
sf.write(out_test, audio, samplerate=sr)

print(mfccs)
print(mfccs.shape)

fig, ax = plt.subplots(nrows=1, sharex=True)
img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                               x_axis='time', y_axis='mel', fmax=8000,
                               ax=ax)
fig.colorbar(img, orientation="vertical", format=lambda x, pos: str(x) + " dB")
ax.set(title='Mel-Spektrogramm')
ax.set_xlabel("Zeit")

ax.label_outer()

fig.savefig("../out/mel.png")
