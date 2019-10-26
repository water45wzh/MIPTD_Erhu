import torch
import torch.utils.data.dataloader as dataloader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import util
from torchvision import transforms
from tqdm import tqdm
import librosa as lb
import pickle
import librosa
class mel_pk(Dataset):
    def __init__(self, rootdir):
        self.file_list = []
        self.npy_path = "/home/lijingru/11classes/npy"
        self.mp3_path = "/home/lijingru/11classes/data"
    def __getitem__(self, item):
        # y, sr = lb.load("MP3/"+str(item)+".mp3", sr=None, duration=5.0)
        # transform_test = transforms.Compose([
        #     lambda x : x.astype(np.float32) / (np.max(np.abs(x))+ 1e-6),
        # ])
        # data = torch.from_numpy(lb.power_to_db(transform_test(np.load(os.path.join(self.mp3_path, str(item)+".npy"))))).float().unsqueeze(0)
        data = torch.from_numpy(lb.power_to_db(np.load(os.path.join(self.mp3_path, str(item)+".npy")))).float().unsqueeze(0)
        label = torch.from_numpy(np.load(os.path.join(self.npy_path, str(item)+".npy"))).float()
        return data, label
    def __len__(self):
        return 5000

class cqt(Dataset):
    def __init__(self, rootdir):
        self.file_list = []
        self.npy_path = "/home/lijingru/3classes/npy"
        self.cqt_path = "/home/lijingru/3classes/cqt"
    def __getitem__(self, item):
        data = torch.from_numpy(np.load(os.path.join(self.cqt_path, str(item)+".npy"))).float().unsqueeze(0) # 2048
        label = torch.from_numpy(np.load(os.path.join(self.npy_path, str(item)+".npy"))).float()
        return data, label
    def __len__(self):
        return 1000

