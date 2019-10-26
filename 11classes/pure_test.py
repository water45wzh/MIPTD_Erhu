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
file_path = "/home/lijingru/11classes/test"

def test(**kwargs):
    opt._parse(kwargs)
    model = models.myNet2(nclass=7)
    opt.notes=""
    model.load_latest(opt.notes)
    model.to(opt.device)
    # dataLoader 只是构造了一个迭代器，网络输入其实就是那个 npy 文件。
    # 一种是都load出来，再对npy进行分段。
    fi = open("system.ann", "w")
    another = ["vibrato","trill","tremelo","staccato","ricochet","pizzicato","percussive","legato&slide&glissando","harmonic","diangong","detache"]
    y, sr = lb.load(os.path.join(file_path, "sbdys.mp3"), sr=None)
    feature = lb.feature.melspectrogram(y=y, sr=sr, hop_length=2205, n_mels=128)

# ===========================================================做测试========================================================
    total = 0
    num = 0
    full = 0
    frame_length = 201
    hop_length = 20
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
            res = torch.zeros(11)
            for item in li:
                res = torch.add(res, item)
            last.append(torch.argmax(res, dim=0).item())
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
    fi.close()
if __name__=='__main__':
    fire.Fire()