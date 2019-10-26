import librosa
import numpy as np
for item in range(1000):
    out_path = "/home/lijingru/3classes/cqt/"+str(item)+".npy"
    y, sr = librosa.load("MP3/"+str(item)+".mp3", sr=40960, duration=5.0)
    data = np.abs(librosa.cqt(y, sr=sr, hop_length=2048))
    np.save(out_path, data)