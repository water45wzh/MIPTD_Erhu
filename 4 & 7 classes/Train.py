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

def train(**kwargs):
    opt._parse(kwargs)
    model = models.myNet2(nclass=3)
    print(model.get_model_name())

    if opt.load_latest is True:
        model.load_latest(opt.notes)
    elif opt.load_model_path:
        print("load_model:", opt.load_model_path)
        model.load(opt.load_model_path)
    model.to(opt.device)
    model.save()
    print("Loading data..")
    # =================================================================
    data = mel_3class(opt.train_data_pth)
    lengths = [2000, 500]
    train_mel_dataset, test_mel_dataset = torch.utils.data.dataset.random_split(data, lengths)
    train_dataloader = Dataloader(train_mel_dataset,batch_size = opt.batch_size,shuffle =False,num_workers = opt.num_workers)
    test_dataloader = Dataloader(test_mel_dataset,batch_size = opt.batch_size,shuffle =False,num_workers = opt.num_workers)
    # =================================================================
    # 随便用了一个loss
    criterion = torch.nn.BCELoss()
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = lr,
                                 weight_decay = opt.weight_decay)
    loss_meter = torchnet.meter.AverageValueMeter()
    previous_loss = 1e100
    # =================================================================
    eva_result = eva_test(test_dataloader, model, criterion)
    print("Begin train..")
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        #confusicon_matrix.reset()
        loss_sum =0.0
        count =0.0

        for ii,(data,label) in tqdm(enumerate(train_dataloader)):
            input = data.cuda()
            target = label.cuda()
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            loss_sum+=loss.item()
            count+=1
            optimizer.step()
            loss_meter.add(loss.item())
        eva_result = eva_test(test_dataloader, model, criterion)
        if epoch % opt.print_freq == opt.print_freq - 1:
            print("train_loss:",loss_meter.value()[0]," ",loss_sum/count)
            model.save()

@torch.no_grad()
def eva_test(dataloader,model,criterion):
    #===============================================================
    # output size == 7*201
    model.eval()
    print("Eval..")
    Apmeter = torchnet.meter.APMeter()
    lossmeter = torchnet.meter.AverageValueMeter()
    mapmeter = torchnet.meter.mAPMeter()
    count =0.0
    acc, err, recall, precision, F1 =0.0,0.0,0.0,0.0,0.0
    lossmeter.reset()
    for ii, (data, label) in tqdm(enumerate(dataloader)):
        input = data.cuda()
        target = label.cuda()
        # print(input.shape)
        # print(target.shape)
        score = model(input)
        # print(score.shape)
        loss = criterion(score, target)
        #===========================================================
        # 计算准确率
        total = 0
        print(score.shape[0])
        for i in range(score.shape[0]):
            # cnt = 0
            for j in range(201):
                # if i == 0 or i == 1:
                #     print(torch.argmax(score, dim=1)[i])
                #     print(torch.argmax(target, dim=1)[i])
                if torch.argmax(score, dim=1)[i][j] == torch.argmax(target, dim=1)[i][j]:
                    # cnt += 1
                    total += 1
            # print("accuracy is:", cnt/201)
        print("***************************************************total is:", total/(40*201))
    model.train()
    return 0
if __name__ =='__main__':
    fire.Fire()