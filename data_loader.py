import os
import numpy as np

import torch
import torch.utils.data.dataloader as dataloader
from torch.utils.data.dataset import Dataset

from config import *

from tqdm import tqdm
import librosa as lb

def process_label(label_pth, num, classes=0):
    label_feature = os.path.join(label_pth, str(num)+".npy")
    label_feature = np.load(label_feature)
    if classes == 7:
        label_feature = label_feature[:-1,:]
    label = torch.from_numpy(label_feature).float()
    return label

def process_mel(mel_pth, num):
    data_feature = os.path.join(mel_pth, str(num)+".npy")
    data_feature = lb.power_to_db(np.load(data_feature))
    data = torch.from_numpy(data_feature).float().unsqueeze(0)
    return data

def process_cqt(cqt_pth, num):
    data_feature = os.path.join(cqt_pth, str(num)+".npy")
    data_feature = lb.amplitude_to_db(np.abs(np.load(data_feature)), ref=np.max)
    data = torch.from_numpy(data_feature).float().unsqueeze(0)
    return data

class mel(Dataset):
    def __init__(self, mode):
        self.file_list = []

        # ------------------add random label into folder-------------------
        if mode == "train":
            self.root_path = opt.train_gen
            self.len = opt.total_len

            self.data_path = self.root_path+"/data"
            self.label_path = self.root_path+"/label"

            for item in range(opt.total_len):
                data = process_mel(self.data_path, item)
                if opt.dataset == "zhudi":
                    label = process_label(self.label_path, item, classes=opt.classes)
                else:
                    label = process_label(self.label_path, item)
                self.file_list.append((data, label))

        elif mode == "test":
            self.root_path = opt.test_gen
            self.len = opt.test_len

            self.data_path = self.root_path+"/data"
            self.label_path = self.root_path+"/label"

            for item in range(opt.test_len):
                data = process_mel(self.data_path, item)
                label = process_label(self.label_path, item)
                self.file_list.append((data, label))

    def __getitem__(self, item):
        return self.file_list[item]

    def __len__(self):
        return self.len

class cqt(Dataset):
    def __init__(self, mode):
        if mode == "train":
            self.root_path = opt.train_gen
            #self.len = opt.total_len
            self.other_len = 523
            self.len = opt.test_len + self.other_len
            self.other_pth = "data/gen/others/train"

        elif mode == "test":
            self.root_path = opt.test_gen
            #self.len = opt.test_len
            self.other_len = 128
            self.len = opt.test_len + self.other_len
            self.other_pth = "data/gen/others/test"

        self.data_path = self.root_path+"/cqt"
        self.label_path = self.root_path+"/label"

        self.file_list = []

        for item in range(self.len):
            data = process_cqt(self.data_path, item)
            label = process_label(self.label_path, item)
            self.file_list.append((data, label))

        self.data_path = self.other_pth+"/cqt"
        self.label_path = self.other_pth+"/label"

        for item in range(self.other_len):
            data = process_cqt(self.data_path, item)
            label = process_label(self.label_path, item)
            self.file_list.append((data, label))


    def __getitem__(self, item):
        return self.file_list[item]

    def __len__(self):
        return self.len

class mel_plus_cqt(Dataset):
    def __init__(self, mode):
        if mode == "train":
            self.root_path = opt.train_gen
            self.len = opt.total_len
        elif mode == "test":
            self.root_path = opt.test_gen
            self.len = opt.test_len

        self.mel_path = self.root_path+"/data"
        self.cqt_path = self.root_path+"/cqt"
        self.label_path = self.root_path+"/label"
        self.file_list = []

    def __getitem__(self, item):
        mel = process_mel(self.mel_path, item)
        cqt = process_cqt(self.cqt_path, item)
        label = process_label(self.label_path, item)
        return (mel, cqt), label

    def __len__(self):
        return self.len

class zhudi_others(Dataset):
    def __init__(self, mode):
        self.root_path = opt.train_gen

        self.file_list = []
        # ------------------add random label into folder-------------------
        self.data_path = self.root_path+"/data"
        self.label_path = self.root_path+"/label"

        if mode == "train":
            self.root_path = opt.train_gen
            self.other_len = 1021
            self.len = 3021
            self.other_pth = "CBF-periDB/others/train"

            for item in range(opt.total_len):
                data = process_mel(self.data_path, item)
                if opt.dataset == "zhudi":
                    label = process_label(self.label_path, item, classes=opt.classes)
                else:
                    label = process_label(self.label_path, item)
                self.file_list.append((data, label))

        elif mode == "test":
            self.root_path = opt.test_gen
            self.len = opt.test_len
            # self.other_len = 128
            # if opt.dataset == "zhudi":
            #     # self.len = 1126
            #     self.len = 126
            # self.other_pth = "CBF-periDB/others/test"

            for item in range(opt.test_len):
                data = process_mel(self.data_path, item)
                label = process_label(self.label_path, item)
                self.file_list.append((data, label))

        # ------------------add others label into folder-------------------
        '''
        if opt.dataset == "zhudi":
            self.data_path = self.other_pth+"/data"
            self.label_path = self.other_pth+"/label"

            for item in range(self.other_len):
                data = process_cqt(self.data_path, item)
                label = process_label(self.label_path, item)
                # ----------------------since there are pieces that is not whole-----------------
                # ----------------------ignore these pieces-----------------
                # print(data.size())
                if np.size(data, axis = 2) == opt.duration:
                    self.file_list.append((data, label))
                # self.file_list.append((data, label))
        '''

    def __getitem__(self, item):
        return self.file_list[item]

    def __len__(self):
        return self.len
