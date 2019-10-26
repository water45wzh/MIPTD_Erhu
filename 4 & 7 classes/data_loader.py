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
        self.npy_path = "/home/lijingru/3classes/npy"
        self.mp3_path = "/home/lijingru/3classes/data"
    def __getitem__(self, item):
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
        return 2000

class mel_test(Dataset):
    def __init__(self, rootdir):
        self.file_list = []
        self.npy_path = "/home/lijingru/3classes/npy"
        self.mp3_path = "/home/lijingru/3classes/data"
    def __getitem__(self, item):
        data = torch.from_numpy(lb.power_to_db(np.load(os.path.join(self.mp3_path, str(item+5000)+".npy")))).float().unsqueeze(0)
        label = torch.from_numpy(np.load(os.path.join(self.npy_path, str(item+5000)+".npy"))).float()
        return data, label
    def __len__(self):
        return 2000

class mel_3class(Dataset):
    def __init__(self, rootdir):
        self.file_list = []
        self.npy_path = "/home/lijingru/3classes/npy_3"
        self.mp3_path = "/home/lijingru/3classes/data_3"
    def __getitem__(self, item):
        data = torch.from_numpy(lb.power_to_db(np.load(os.path.join(self.mp3_path, str(item)+".npy")))).float().unsqueeze(0)
        label = torch.from_numpy(np.load(os.path.join(self.npy_path, str(item)+".npy"))).float()
        return data, label
    def __len__(self):
        return 2500

class test_3class(Dataset):
    def __init__(self, rootdir):
        self.file_list = []
        self.npy_path = "/home/lijingru/3classes/npy_3"
        self.mp3_path = "/home/lijingru/3classes/data_3"
    def __getitem__(self, item):
        data = torch.from_numpy(lb.power_to_db(np.load(os.path.join(self.mp3_path, str(item+2500)+".npy")))).float().unsqueeze(0)
        label = torch.from_numpy(np.load(os.path.join(self.npy_path, str(item+2500)+".npy"))).float()
        return data, label
    def __len__(self):
        return 500

class new_3class(Dataset):
    def __init__(self, rootdir):
        self.file_list = []
        self.npy_path = "/home/lijingru/test_han/npy_3"
        self.mp3_path = "/home/lijingru/test_han/data_3"
    def __getitem__(self, item):
        data = torch.from_numpy(lb.power_to_db(np.load(os.path.join(self.mp3_path, str(item)+".npy")))).float().unsqueeze(0)
        label = torch.from_numpy(np.load(os.path.join(self.npy_path, str(item)+".npy"))).float()
        return data, label
    def __len__(self):
        return 1000

class new_7class(Dataset):
    def __init__(self, rootdir):
        self.file_list = []
        self.npy_path = "/home/lijingru/test_han/npy_7"
        self.mp3_path = "/home/lijingru/test_han/data_7"
    def __getitem__(self, item):
        data = torch.from_numpy(lb.power_to_db(np.load(os.path.join(self.mp3_path, str(item)+".npy")))).float().unsqueeze(0)
        label = torch.from_numpy(np.load(os.path.join(self.npy_path, str(item)+".npy"))).float()
        return data, label
    def __len__(self):
        return 1000

