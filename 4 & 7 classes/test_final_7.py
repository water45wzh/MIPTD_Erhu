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
import math
def test(**kwargs):
    opt._parse(kwargs)
    data = new_7class(opt.train_data_pth)
    # 取出2000条数据，作为一个batch送进去
    test_dataloader = Dataloader(data, batch_size = 1,shuffle =False,num_workers = opt.num_workers)
    model = models.myNet2(nclass=7)
    opt.notes="_7classes"
    model.load_latest(opt.notes)
    #model.load_latest(opt.notes)
    model.to(opt.device)
    total = 0
    for ii, (data, label) in tqdm(enumerate(test_dataloader)):
        input = data.cuda()
        target = label.cuda()
        score = model(input)
        #===========================================================
        # 计算准确率
        # print(score.shape[0])
        for i in range(score.shape[0]):
            for j in range(201):
                if torch.argmax(score, dim=1)[i][j] == torch.argmax(target, dim=1)[i][j]:
                    total += 1
            # print("accuracy is:", cnt/101)
    print("***************************************************total is:", total/(1000*201))

if __name__ =='__main__':
    fire.Fire()