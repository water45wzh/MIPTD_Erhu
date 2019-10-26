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
file_path = "/home/lijingru/technique/test"

# test 怎么写标签的问题。
def test(**kwargs):
    opt._parse(kwargs)
    model = models.myNet2(nclass=7)
    opt.notes=""
    model.load_latest(opt.notes)
    model.to(opt.device)
    # dataLoader 只是构造了一个迭代器，网络输入其实就是那个 npy 文件。
    # 一种是都load出来，再对npy进行分段。
    with open(os.path.join(file_path, "良宵.txt"), "r", encoding='gbk') as fp:
        file_list = [line.rstrip() for line in fp]
    technique = ["下滑音", "上滑音", "上颤音", "连线滑音", "顿弓", "垫指滑音", "颤音"]
    y, sr = lb.load(os.path.join(file_path, "05 良宵.wav"), sr=None)
    feature = lb.feature.melspectrogram(y=y, sr=sr, hop_length=2205, n_mels=128)
    out_path=os.path.join(file_path, "npy/良宵.npy")
    np.save(out_path, feature)
    #print(feature.shape)
    arr = np.zeros((7,feature.shape[1]))
    for lines in file_list:
        lis = lines.split("\t")
        label = lis[0]
        start = float(lis[1].split(":")[0])*60+float(lis[1].split(":")[1])
        end = float(lis[2].split(":")[0])*60+float(lis[2].split(":")[1])
        for ii in range(feature.shape[1]):
            if ii*0.05>=start and ii*0.05<=end:
                arr[technique.index(label)][ii] = 1
    for ii in range(int(feature.shape[1]/101)):
        if ii*101+101 >= feature.shape[1]:
            break
        data = torch.from_numpy(lb.db_to_power(feature[:,ii*101:ii*101+101])).float().unsqueeze(0)
        # data = torch.from_numpy(lb.db_to_power(np.zeros((128, 101)))).float().unsqueeze(0)
        label = torch.from_numpy(arr[:,ii*101:ii*101+101]).float()
        #print(label.shape)
        # 每次取出101长的数据
        # for k in range(1):
        #     for j in range(128):
        #         for i in range(101):
        #             if data[k,j,i] > 20:
        #                 print("hey")
        data = data[np.newaxis,:,:,:]
        print(data)
        label = label.cuda()
        input = data.cuda()
        score = model(input)
        print(score)
        cnt = 0
        for j in range(101):
            if torch.argmax(score, dim=1).squeeze(0)[j] == torch.argmax(label, dim=0)[j]:
                cnt += 1
        print("precision is:", cnt/101)
if __name__=='__main__':
    fire.Fire()