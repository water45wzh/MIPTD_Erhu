from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import json
import torch.nn.functional as F
from torch.nn.parallel.data_parallel import DataParallel
import os
import pandas as pd
import numpy as np
import time
import fire
from config import *
import models
from data_loader import *
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as Dataloader
import torchnet
import visdom
from tqdm import tqdm
import util
import os
import csv
file_path = "/home/lijingru/11classes/test"

# test 怎么写标签的问题。
def test(**kwargs):
    opt._parse(kwargs)
    model = models.myNet2(nclass=7)
    opt.notes=""
    model.load_latest(opt.notes)
    model.to(opt.device)
    # dataLoader 只是构造了一个迭代器，网络输入其实就是那个 npy 文件。
    # 一种是都load出来，再对npy进行分段。
    with open(os.path.join(file_path, "赛马_R.txt"), "r", encoding='gbk') as fp:
        file_list = [line.rstrip() for line in fp]
    technique = ["vibrato","trill","tremelo","staccato","ricochet","pizzicato","percussive","legato&slide&glissando","harmonic","diangong","detache"]

    y, sr = lb.load(os.path.join(file_path, "赛马_R.wav"), sr=None)
    feature = lb.feature.melspectrogram(y=y, sr=sr, hop_length=2205, n_mels=128)
    out_path=os.path.join(file_path, "npy/赛马.npy")
    np.save(out_path, feature)
    #print(feature.shape)
    arr = np.zeros((11,feature.shape[1]))
    for ii, lines in enumerate(file_list):
        end = 0
        if ii == 0:
            continue
        lis = lines.split("\t")
        # print(lis)
        if ii != len(file_list)-1:
            lis2 = file_list[ii+1].split("\t")
        label = lis[2]
        if ":" in lis[1]:
            start = float(lis[1].split(":")[0])*60+float(lis[1].split(":")[1])
        else:
            start = float(lis[1])
        if ":" in lis2[1]:
            end = float(lis2[1].split(":")[0])*60+float(lis2[1].split(":")[1])
        else:
            end = float(lis2[1])
        for j in range(feature.shape[1]):
            if j*0.05>=start and (ii == len(file_list)-1 or j*0.05<=end):
                arr[int(label)][j] = 1
    cnt_ = 0
    total = 0
    for ii in range(int(feature.shape[1]/201)):
        if ii*201+201 >= feature.shape[1]:
            break
        data = torch.from_numpy(lb.power_to_db(feature[:,ii*201:ii*201+201])).float().unsqueeze(0)
        # data = torch.from_numpy(lb.db_to_power(np.zeros((128, 201)))).float().unsqueeze(0)
        label = torch.from_numpy(arr[:,ii*201:ii*201+201]).float()
        #print(label.shape)
        # 每次取出201长的数据
        # for k in range(1):
        #     for j in range(128):
        #         for i in range(201):
        #             if data[k,j,i] > 20:
        #                 print("hey")
        data = data[np.newaxis,:,:,:]
        # print(data.shape)
        label = label.cuda()
        input = data.cuda()
        score = model(input)
        print(torch.argmax(score, dim=1).squeeze(0))
        print(torch.argmax(label, dim=0))
        cnt = 0
        for j in range(201):
            total += 1
            if torch.argmax(score, dim=1).squeeze(0)[j] == torch.argmax(label, dim=0)[j]:
                cnt += 1
                cnt_ += 1
        print("precision is:", cnt/201)
    print(cnt_/total)
if __name__=='__main__':
    fire.Fire()