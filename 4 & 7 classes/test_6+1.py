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
file_path = "/home/lijingru/3classes/data5/test"

# test 怎么写标签的问题。
def test(**kwargs):
    opt._parse(kwargs)
    model = models.myNet2(nclass=7)
    opt.notes="_7classes"
    model.load_latest(opt.notes)
    model.to(opt.device)
    # dataLoader 只是构造了一个迭代器，网络输入其实就是那个 npy 文件。
    # 一种是都load出来，再对npy进行分段。
    fo = open("reference.ann", "w")
    fi = open("system.ann", "w")
    with open(os.path.join(file_path, "二胡-独奏label.txt"), "r", encoding='gbk') as fp:
        file_list = [line.rstrip() for line in fp]
    technique = ["颤音","上滑音","下滑音","顿弓","上颤音","连线滑音","其他"]
    another = ["chanyin", "shanghua", "xiahua", "dungong", "shangchan", "lianxian", "other"]
    y, sr = lb.load(os.path.join(file_path, "二胡-独奏.wav"), sr=None)
    feature = lb.feature.melspectrogram(y=y, sr=sr, hop_length=2205, n_mels=128)
    out_path=os.path.join(file_path, "npy/独奏.npy")
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
                if name != "垫指滑音":
                    fo.write(str(start)+"\t"+str(end)+"\t"+another[technique.index(name)]+"\n")
                else:
                    fo.write(str(start)+"\t"+str(end)+"\t"+another[6]+"\n")
                # fi.write(os.path.join(file_path, "二胡-独奏label.txt")+" "+str(start)+"\t"+str(end)+"\t"+name+"\n")
                for ii in range(feature.shape[1]):
                    arr[6][ii] = 1 # 如果都不是
                    if ii*0.05>=start and ii*0.05<=end:
                        if name == "垫指滑音":
                            arr[6][ii] = 1
                        elif name in technique:
                            arr[technique.index(name)][ii] = 1
                            arr[6][ii] = 0

# ===========================================================做测试========================================================
    total = 0
    num = 0
    full = 0
    frame_length = 201
    hop_length = 100
    lis = []
    for i in range(feature.shape[1]):
        lt = [] # 每次新建一个列表
        lis.append(lt)
    for ii in range(5000):
        if ii*hop_length+frame_length >= feature.shape[1]:
            break
        data = torch.from_numpy(lb.power_to_db(feature[:,ii*hop_length:ii*hop_length+frame_length])).float().unsqueeze(0)
        label = torch.from_numpy(arr[:,ii*hop_length:ii*hop_length+frame_length]).float()
        data = data[np.newaxis,:,:,:]
        # print(data)
        label = label.cuda()
        input = data.cuda()
        score = model(input)
        # print(score)
        cnt = 0
        cnt_ = 0
        # print(torch.argmax(score, dim=1))
        # print(torch.argmax(label, dim=0))
        ha = score.cpu()
        for j in range(frame_length):
            lis[j+ii*hop_length].append(ha.squeeze(0)[:,j])
        for j in range(201):
            full += 1
            cnt_ += 1
            if torch.argmax(score, dim=1).squeeze(0)[j] == torch.argmax(label, dim=0)[j]:
                cnt += 1
                total += 1
        # if cnt_ != 0:
            # print("precision is:", cnt/cnt_)
        num += 1
    print("total is", total/full)
    last = []
    for li in lis:
        if len(li) != 0:
            res = torch.zeros(7)
            for item in li:
                res = torch.add(res, item)
            last.append(torch.argmax(res, dim=0).item())
    # print(last)
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
    print(len(last))
    print(np.argmax(arr, axis=0).shape)
    # print(lis)
    fo.close()
    fi.close()
if __name__=='__main__':
    fire.Fire()