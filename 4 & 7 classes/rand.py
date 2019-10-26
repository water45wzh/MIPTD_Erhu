# numpy.random.randint(low, high=None, size=None, dtype='l')
# #这个方法产生离散均匀分布的整数，这些整数大于等于low，小于high。
# low : int        #产生随机数的最小值
# high : int, optional    #给随机数设置个上限，即产生的随机数必须小于high
# size : int or tuple of ints, optional    #整数，生成随机元素的个数或者元组，数组的行和列
# dtype : dtype, optional    #期望结果的类型

import os
import librosa as lb
import numpy as np
from tqdm import tqdm
from pydub import AudioSegment

file_path = "/home/lijingru/3classes/data3/train"
file_list = []
for ii, (root, dirs, files) in tqdm(enumerate(os.walk(file_path))):
    for file in files:
        label = os.path.join(root,file).split("/")[-2]
        file_list.append((os.path.join(root,file), 
                          label))
        # print(label)
randList = []
i = 0
for cnt in range(10000):
    if i == 1000:
        break
    rand = list(np.random.randint(0, len(file_list), size = 1000))
    # print(rand)
    labelList = []
    first = False
    chosen = []
    cnt = 0
    for ii, item in enumerate(rand):
        file, label = file_list[item]
        # 颤音不得连续出现
        if ii != 0:
            file_pre, label_pre = file_list[rand[ii-1]]
            if label_pre == label and label == "颤音":
                continue
        # 颤音不得超过两次
        if label == "颤音":
            if cnt == 2:
                continue
            cnt += 1
        song_part = AudioSegment.from_wav(file)
        if first == False:
            first = True
            song = song_part
            start = 0
            end = song.duration_seconds
        else:
            start = song.duration_seconds
            song = song.append(song_part)
            end = song.duration_seconds
        labelList.append((start, end, label))
        # print(labelList)
        chosen.append(item)
        # 选择了哪些角标
        # 大于5s则停止
        if song.duration_seconds >= 5.0:
            break
    if chosen in randList:
        print("exist before!")
        continue;
    # 如果之前生成过同样的列表，则跳过循环
    # 试试查重算法有了之后是否
    randList.append(chosen)
    song.export("MP3/"+str(i)+".mp3", format="mp3")
    y, sr = lb.load("MP3/"+str(i)+".mp3", sr=None, duration=5.0)
    feature = lb.feature.melspectrogram(y=y, sr=sr, hop_length=2205, n_mels=128)
    out_pth = "data/"+str(i)+".npy"
    np.save(out_pth, feature)
    technique = ["颤音","滑音","顿弓"]
    arr = np.zeros((3, 101))
    # print(labelList)
    for item in labelList:
        start, end, label = item
        for ii in range(101):
            if ii*0.05>=start and ii*0.05<=end:
                arr[technique.index(label)][ii] = 1
    # print(arr)
    # arr[ii] = technique.index(label)
    # print(arr.shape)
    out_path = "npy/"+str(i)+".npy"
    np.save(out_path, arr)
    i += 1
