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
from rand import *
import models
from data_loader import *

from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as Dataloader

#本机注释掉这一堆

import torchnet
import visdom
import math

#*******************************************************
# generate data 
#
def gen(**kwargs):
    opt._parse(kwargs)
    gen_rand = rand(opt.mode)
    gen_rand.gen_dataset()

def gen_cqt(**kwargs):
    opt._parse(kwargs)
    gen_rand = rand(opt.mode)
    gen_rand.gen_cqt()

#*******************************************************
# train
#
def train(**kwargs):
    # parse args, load network, set class number as 4/7
    opt._parse(kwargs)
    model = eval("models."+opt.model+"_"+opt.feature+"_"+str(opt.classes)+"class()")
    print(model.get_model_name())

    if opt.load_latest is True:
        model.load_latest(opt.notes)
    elif opt.load_model_path:
        print("load_model:", opt.load_model_path)
        model.load(opt.load_model_path)
    model.to(opt.device)
    model.save(opt.notes)
    print("Loading data..")
    # load data 
    data = eval(opt.feature+"(\"train\")")

    lengths = [opt.train_len, opt.eval_len]
    # if opt.dataset == "zhudi":
    #     other_len = 1021
    #     lengths = [opt.train_len, opt.eval_len]
        # lengths = [int(0.75*(opt.total_len+other_len)), opt.total_len+other_len-int(0.75*(opt.total_len+other_len))]
    train_mel_dataset, test_mel_dataset = torch.utils.data.dataset.random_split(data, lengths)
    train_dataloader = Dataloader(train_mel_dataset,
                                  batch_size = opt.batch_size,
                                  shuffle =False,
                                  num_workers = opt.num_workers)
    test_dataloader = Dataloader(test_mel_dataset,
                                 batch_size = opt.batch_size,
                                 shuffle =False,
                                 num_workers = opt.num_workers)
    # set loss and learning parameters
    criterion = torch.nn.BCELoss()
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = lr,
                                 weight_decay = opt.weight_decay)
    loss_meter = torchnet.meter.AverageValueMeter()
    previous_loss = 1e100

    eva_result = eva_test(test_dataloader, 
                          model,
                          criterion)
    # begin train 
    print("Begin train..")
    for epoch in range(opt.max_epoch):
        print("this is ", epoch,"epoch")
        loss_meter.reset()
        loss_sum =0.0
        count =0.0

        for ii,(data,label) in tqdm(enumerate(train_dataloader)):
            if opt.feature == "mel_plus_cqt":
                input = [_data.cuda() for _data in data]
            else:
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

        eva_result = eva_test(test_dataloader, 
                              model, 
                              criterion)
        print("train_loss:",loss_meter.value()[0]," ",loss_sum/count)

        if epoch % opt.print_freq == opt.print_freq - 1:
            print("train_loss:",loss_meter.value()[0]," ",loss_sum/count)
            model.save(opt.notes)

#*******************************************************
# eval
#
@torch.no_grad()
def eva_test(dataloader,model,criterion):
    model.eval()
    print("Eval..")
    Apmeter = torchnet.meter.APMeter()
    lossmeter = torchnet.meter.AverageValueMeter()
    mapmeter = torchnet.meter.mAPMeter()
    count =0.0
    acc, err, recall, precision, F1 =0.0,0.0,0.0,0.0,0.0
    lossmeter.reset()
    for ii, (data, label) in tqdm(enumerate(dataloader)):
        if opt.feature == "mel_plus_cqt":
            input = [_data.cuda() for _data in data]
        else:
            input = data.cuda()
        target = label.cuda()
        score = model(input)
        loss = criterion(score, target)
        # calculate precision
        total = 0
        for i in range(score.shape[0]):
            for j in range(opt.duration):
                if torch.argmax(score, dim=1)[i][j] == torch.argmax(target, dim=1)[i][j]:
                    total += 1
        print("***************************************************total is:", 
              total/(score.shape[0]*opt.duration))
    model.train()
    return 0

#*******************************************************
# test
#
def test(**kwargs):
    opt._parse(kwargs)
    model = eval("models."+opt.model+"_"+opt.feature+"_"+str(opt.classes)+"class()")
    data = eval(opt.feature+"(\"test\")")
    test_dataloader = Dataloader(data, 
                                 batch_size = 1,
                                 shuffle =False,
                                 num_workers = opt.num_workers)

    opt.nodes = ''

    f1 = open('./confusion_matrix/'+opt.notes+opt.model+"_"+opt.feature+"_"+str(opt.classes)+"class"+'_real_label.txt', mode='w', encoding='utf-8')
    f2 = open('./confusion_matrix/'+opt.notes+opt.model+"_"+opt.feature+"_"+str(opt.classes)+"class"+'_pred_label.txt', mode='w', encoding='utf-8')

    model.load_latest(opt.notes)
    model.to(opt.device)
    total = 0
    cnt = 0
    for ii, (data, label) in tqdm(enumerate(test_dataloader)):
        if opt.feature == "mel_plus_cqt":
            input = [_data.cuda() for _data in data]
        else:
            input = data.cuda()
        target = label.cuda()
        score = model(input)
        # calculate precision
        for i in range(score.shape[0]):
            for j in range(opt.duration):
                cnt += 1
                if torch.argmax(score, dim=1)[i][j] == torch.argmax(target, dim=1)[i][j]:
                    total += 1
                f1.write(str(torch.argmax(target, dim=1)[i][j].item())+'\n')
                f2.write(str(torch.argmax(score, dim=1)[i][j].item())+'\n')
    f1.close()
    f2.close()
    # print("***************************************************total is:",total/(opt.test_len*opt.duration))
    print(cnt)
    print("***************************************************total is:",total/cnt)

#************************************************
# test real pieces
# 
def real_test(**kwargs):
    opt._parse(kwargs)
    model = eval("models."+opt.model+"_"+opt.feature+"_"+str(opt.classes)+"class()")
    model.load_latest(opt.notes)
    model.to(opt.device)

    # construct 2 version of test data for zhudi:
    if opt.test_ver == "real":
        f1 = open('./confusion_matrix/real_'+opt.notes+opt.model+"_"+opt.feature+"_"+str(opt.classes)+"class"+'_real_label.txt', mode='w', encoding='utf-8')
        f2 = open('./confusion_matrix/real_'+opt.notes+opt.model+"_"+opt.feature+"_"+str(opt.classes)+"class"+'_pred_label.txt', mode='w', encoding='utf-8')
        data_pth = os.path.join(opt.real_world, "data")
        label_pth = os.path.join(opt.real_world, "label")
        piece_num = opt.real_num

    elif opt.test_ver == "iso":
        f1 = open('./confusion_matrix/iso_'+opt.notes+opt.model+"_"+opt.feature+"_"+str(opt.classes)+"class"+'_real_label.txt', mode='w', encoding='utf-8')
        f2 = open('./confusion_matrix/iso_'+opt.notes+opt.model+"_"+opt.feature+"_"+str(opt.classes)+"class"+'_pred_label.txt', mode='w', encoding='utf-8')
        data_pth = os.path.join(opt.iso_pth, "data")
        label_pth = os.path.join(opt.iso_pth, "label")
        piece_num = opt.iso_num

    total = 0
    cnt = 0


    for pieces in range(piece_num):
        feature = np.load(os.path.join(data_pth, str(pieces)+".npy"))
        arr = np.load(os.path.join(label_pth, str(pieces)+".npy"))

        # cut all others label in feature and arr
        # print(np.argmax(arr, axis = 0))
        arr_new = arr[:, np.argmax(arr, axis=0) != 7]
        feature_new = feature[:, np.argmax(arr, axis=0) != 7]

        arr = arr_new
        feature = feature_new

        # print(arr.shape)
        # print(feature.shape)

        num = 0
        frame_length = 201
        hop_length = 40
        # 原来是40
        lis = []
        for i in range(feature.shape[1]):
            lt = [] # 每次新建一个列表
            lis.append(lt)
        for ii in range(5000*2):
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
                res = torch.zeros(opt.classes)
                for item in li:
                    res = torch.add(res, item)
                last.append(torch.argmax(res, dim=0).item())
        for ii in range(len(last)):
            target = np.argmax(arr, axis=0)[ii]
            score = last[ii]
            if target != 7:
                total += 1
            if target == score and target != 7:
            # if target == score:
                cnt += 1
            f1.write(str(target)+'\n')
            f2.write(str(score)+'\n')

    print(cnt/total)
    f1.close()
    f2.close()
if __name__ =='__main__':
    fire.Fire()