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
import matplotlib.pyplot as plt
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
    # print(feature.shape)
    arr = np.zeros((7,feature.shape[1]))
# ==========================================================生成标签=======================================================
    for lines in file_list:
        lis = lines.split("\t")
        label = lis[0]
        start = float(lis[1].split(":")[0])*60+float(lis[1].split(":")[1])
        end = float(lis[2].split(":")[0])*60+float(lis[2].split(":")[1])
        for ii in range(feature.shape[1]):
            arr[3][ii] = 0.5
        for ii in range(feature.shape[1]):
            if ii*0.05>=start and ii*0.05<=end:
                if label == "上颤音" or label == "颤音":
                    arr[0][ii] = 1
                if label == "上滑音" or label == "下滑音" or label == "垫指滑音" or label == "连线滑音":
                    arr[1][ii] = 1
                if label == "顿弓":
                    arr[2][ii] = 1
# ===========================================================做测试========================================================
    total = 0
    count = 0
    for ii in range(int(feature.shape[1]/101)):
        if ii*101+101 >= feature.shape[1]:
            break
        data = torch.from_numpy(lb.power_to_db(feature[:,ii*101:ii*101+101])).float().unsqueeze(0)
        label = torch.from_numpy(arr[:,ii*101:ii*101+101]).float()
        data = data[np.newaxis,:,:,:]
        label = label.cuda()
        input = data.cuda()
        score = model(input)
        cnt = 0
        cnt_ = 0
        print(torch.argmax(score, dim=1).squeeze(0))
        print(torch.argmax(label, dim=0))
        for j in range(101):
            if torch.argmax(label, dim=0)[j] != 3:
                cnt_ += 1
                total += 1
            if torch.argmax(score, dim=1).squeeze(0)[j] == torch.argmax(label, dim=0)[j]:
                cnt += 1
                count += 1
        if cnt_ != 0:
            print("precision is:", cnt/cnt_)
    print("total is:", count/total)
if __name__ =='__main__':
    fire.Fire()