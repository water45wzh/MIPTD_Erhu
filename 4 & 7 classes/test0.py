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
    npy_path = "/home/lijingru/technique/npy"
    mp3_path = "/home/lijingru/technique/data"
    transform_test = transforms.Compose([
        lambda x : x.astype(np.float32) / (np.max(np.abs(x))+ 1e-6),
    ])
    print(np.max(np.abs(np.load(os.path.join(mp3_path, "100.npy")))))
    print(np.min(np.abs(np.load(os.path.join(mp3_path, "100.npy")))))
    data = torch.from_numpy(lb.db_to_power(transform_test(np.load(os.path.join(mp3_path, "100.npy"))))).float().unsqueeze(0)
    # data = torch.from_numpy(lb.db_to_power(np.zeros((128, 101)))).float().unsqueeze(0)
    # data = transform_test(data)
    label = torch.from_numpy(np.load(os.path.join(npy_path, "100.npy"))).float()
    data = data[np.newaxis,:,:,:]
    print(data)
    mm = torch.min(data)
    num = 0
    for x in data.squeeze(0).squeeze(0):
        for y in x:
            if y == mm:
                num += 1
    print(num)
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