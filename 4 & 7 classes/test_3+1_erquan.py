# pure test
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

def test(**kwargs):
    opt._parse(kwargs)
    model = models.myNet2(nclass=3)
    opt.notes="_3classes"
    model.load_latest(opt.notes)
    model.to(opt.device)
    # dataLoader 只是构造了一个迭代器，网络输入其实就是那个 npy 文件。
    # 一种是都load出来，再对npy进行分段。
    fo = open("reference.ann", "w")
    fi = open("system.ann", "w")
    technique = ["颤音","滑音","顿弓","其他"]
    another = ["chanyin", "huayin", "dungong", "other"]
    y, sr = lb.load(os.path.join(file_path, "二泉映月.wav"), sr=None)
    feature = lb.feature.melspectrogram(y=y, sr=sr, hop_length=2205, n_mels=128)

# ===========================================================做测试========================================================
    with open(os.path.join(file_path, "二泉映月_3+1_ref.txt"), "r", encoding='gbk') as fp:
        file_list = [line.rstrip() for line in fp]
    arr = np.zeros((4,feature.shape[1]))
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
        if ii != len(file_list)-1:
            fo.write(str(start)+"\t"+str(end)+"\t"+another[int(label)]+"\n")
        else:
            fo.write(str(start)+"\t"+str(feature.shape[1]*0.05)+"\t"+another[int(label)]+"\n")
        for j in range(feature.shape[1]):
            if j*0.05>=start and (ii == len(file_list)-1 or j*0.05<=end):
                arr[int(label)][j] = 1

# ===========================================================做测试========================================================
    total = 0
    num = 0
    full = 0
    frame_length = 201
    hop_length = 100
    # 原来是40
    lis = []
    for i in range(feature.shape[1]):
        lt = [] # 每次新建一个列表
        lis.append(lt)
    for ii in range(5000):
        if ii*hop_length+frame_length >= feature.shape[1]:
            break
        data = torch.from_numpy(lb.power_to_db(feature[:,ii*hop_length:ii*hop_length+frame_length])).float().unsqueeze(0)
        data = data[np.newaxis,:,:,:]
        input = data.cuda()
        score = model(input)
        ha = score.cpu()
        for j in range(frame_length):
            lis[j+ii*hop_length].append(ha.squeeze(0)[:,j])
    last = []
    for li in lis:
        if len(li) != 0:
            res = torch.zeros(4)
            for item in li:
                res = torch.add(res, item)
            last.append(torch.argmax(res, dim=0).item())
    cnt = 0
    for ii in range(len(last)):
        if np.argmax(arr, axis=0)[ii] == last[ii]:
            cnt += 1
    print(cnt/len(last))
    tmp = 0
    for ii in range(len(last)):
        if ii == len(last)-1:
            break
        if last[ii] != last[ii+1]:
            start = tmp
            end = ii * 0.05
            name = another[last[ii]]
            fi.write(str(start)+"\t"+str(end)+"\t"+name+"\n")
            tmp = ii * 0.05
    fo.close()
    fi.close()
if __name__=='__main__':
    fire.Fire()