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
file_path = "/home/lijingru/3classes/data3/test"

# test 怎么写标签的问题。
def test(**kwargs):
    opt._parse(kwargs)
    model = models.myNet7(nclass=3)
    opt.notes=""
    model.load_latest(opt.notes)
    model.to(opt.device)
    # dataLoader 只是构造了一个迭代器，网络输入其实就是那个 npy 文件。
    # 一种是都load出来，再对npy进行分段。
    with open(os.path.join(file_path, "二胡-独奏label.txt"), "r", encoding='gbk') as fp:
        file_list = [line.rstrip() for line in fp]
    technique = ["下滑音", "上滑音", "上颤音", "连线滑音", "顿弓", "垫指滑音", "颤音"]
    y, sr = lb.load(os.path.join(file_path, "二胡-独奏.wav"), sr=40960)
    feature = np.abs(lb.cqt(y, sr=sr, hop_length=2048))
    out_path=os.path.join(file_path, "npy/独奏_cqt.npy")
    np.save(out_path, feature)
    # print(feature.shape)
    arr = np.zeros((7,feature.shape[1]))
# ==========================================================生成标签=======================================================
    for lines in file_list:
        name = lines.split("--")[0]
        for ii, item in enumerate(lines.split("--")):
            if ii != 0:
                toki = item[:-1]
                start = toki.split(",")[0]
                start = float(start.split(":")[0])*60+float(start.split(":")[1])
                end = toki.split(",")[1]
                end = float(end.split(":")[0])*60+float(end.split(":")[1])
                for ii in range(feature.shape[1]):
                    arr[3][ii] = 0.5
                for ii in range(feature.shape[1]):
                    if ii*0.05>=start and ii*0.05<=end:
                        if name == "上颤音" or name == "颤音":
                            arr[0][ii] = 1
                        if name == "上滑音" or name == "下滑音" or name == "垫指滑音" or name == "连线滑音":
                            arr[1][ii] = 1
                        if name == "顿弓":
                            arr[2][ii] = 1
# ===========================================================做测试========================================================
    total = 0
    num = 0
    for ii in range(int(feature.shape[1]/101)):
        if ii*101+101 >= feature.shape[1]:
            break
        data = torch.from_numpy(lb.power_to_db(feature[:,ii*101:ii*101+101])).float().unsqueeze(0)
        label = torch.from_numpy(arr[:,ii*101:ii*101+101]).float()
        data = data[np.newaxis,:,:,:]
        # print(data)
        label = label.cuda()
        input = data.cuda()
        score = model(input)
        # print(score)
        cnt = 0
        print(torch.argmax(score, dim=1))
        print(torch.argmax(label, dim=0))
        for j in range(101):
            if torch.argmax(score, dim=1).squeeze(0)[j] == torch.argmax(label, dim=0)[j]:
                cnt += 1
                total += 1
        print("precision is:", cnt/101)
        num += 1
    print("total is", total/(101*num))
if __name__=='__main__':
    fire.Fire()