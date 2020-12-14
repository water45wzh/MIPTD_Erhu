import os
import librosa as lb
import numpy as np
import fire
from pydub import AudioSegment
from tqdm import tqdm

from config import *

class rand:
    def __init__(self, mode):
#**************************************
# 用到的列表
# 
        self.randList = [] #记录所有生成过的随机序列 
        self.file_list = []
        self.labelList = [] #记录每首曲子 start end label 三元组
        self.chosen = [] #记录所有用过的组合
        self.technique = opt.technique
        self.classes = opt.classes

#**************************************
# init path
# 
        self.len = 10
        if mode == "train":
            self.file_path = opt.train_raw
            self.out_path = opt.train_gen
            self.len = opt.total_len

        elif mode == "test":
            self.file_path = opt.test_raw
            self.out_path = opt.test_gen
            self.len = opt.test_len

        self.mp3_path = os.path.join(self.out_path, "mp3")
        self.data_path = os.path.join(self.out_path, "data")
        self.npy_path = os.path.join(self.out_path, "label")

        for it in [self.mp3_path, self.data_path, self.npy_path]:
            if not os.path.exists(it):
                os.mkdir(it)
        #if cqt mode
        self.cqt_path = os.path.join(self.out_path, "cqt")
        if not os.path.exists(self.cqt_path):
            os.makedirs(self.cqt_path)

    def change_pth(self, rt_pth, i, mode="mp3"):
        return os.path.join(rt_pth, str(i) + "." + mode)

#**************************************
# get file from folder
# 
    def init_file(self):
        for ii, (root, dirs, files) in tqdm(enumerate(os.walk(self.file_path))):
            for file in files:
                label = os.path.join(root,file).split("\\")[-2]
                name = os.path.join(root,file).split("\\")[-1]
                self.file_list.append((os.path.join(root,file), 
                                       label, 
                                       name))

#**************************************
# generate song from segment
# 
    def gen_song(self, rand, chosen):
        first = False
        for ii, item in enumerate(rand):
            file, label, name = self.file_list[item]
            song_part = AudioSegment.from_wav(file)
            if first == False:
                first = True
                song = song_part
                start = 0
                end = song.duration_seconds
            else:
                start = song.duration_seconds
                song = song.append(song_part, crossfade = 50)
                end = min(opt.time, song.duration_seconds)
            self.labelList.append((start, end, label))
            chosen.append(name)
            if song.duration_seconds >= opt.time:
                break
        return song, chosen


    def save_data(self, i, song):
#**********************************************************
# save mp3
#
        # mp3_path = self.change_pth(self.mp3_path, i, mode="mp3")
        # song.export(mp3_path, format="mp3")
        mp3_path = self.change_pth(self.mp3_path, i, mode="wav")
        song.export(mp3_path, format="wav")

#**********************************************************
# save feature
#
        y, sr = lb.load(mp3_path, 
                        sr = None, 
                        duration = opt.time)
        feature = lb.feature.melspectrogram(y=y, 
                                            sr=sr, 
                                            hop_length=2205, 
                                            n_mels=128)
        data_path = self.change_pth(self.data_path, i, mode="npy")
        np.save(data_path, feature)

#**********************************************************
# save label
#
        arr = np.zeros((self.classes, opt.duration))
        for item in self.labelList:
            start, end, label = item
            for ii in range(opt.duration):
                if ii*0.05>=start and ii*0.05<=end:
                    arr[self.technique.index(label)][ii] = 1
        out_path = self.change_pth(self.npy_path, i, mode="npy")
        np.save(out_path, arr)


#**************************************
# get a rand sequence
# 
    def randd(self, i, chosen):
        rand = list(np.random.randint(0, 
                                      len(self.file_list), 
                                      size = 1000))
        self.labelList.clear()

        song, chosen = self.gen_song(rand, chosen)
        self.save_data(i, song)

        return chosen

#**************************************
# main entry
# 
    def gen_dataset(self):
        self.init_file()
        i = 0
        for cnt in range(10000):
            if i == self.len:
            #if i == 10:
                break
            self.chosen = []
            self.chosen = self.randd(i, self.chosen)
            if self.chosen in self.randList:
                print("exist before!")
                continue;
            print(self.chosen)
            # 如果之前生成过同样的列表，则跳过循环
            self.randList.append(self.chosen)
            i += 1

    def gen_cqt(self):
        for cnt in range(self.len):
            mp3_path = self.change_pth(self.mp3_path, cnt, mode="wav")
            y, sr = lb.load(mp3_path, 
                            sr = 40960, 
                            duration = opt.time)
            feature = np.abs(lb.cqt(y, sr=sr, hop_length=2048))
            #print(feature.shape)
            data_path = self.change_pth(self.cqt_path, cnt, mode="npy")
            np.save(data_path, feature)