
import warnings

import configparser
import torch
import os
config = configparser.ConfigParser()


class Config(object):
    model = "myNet2"
    max_epoch = 2000
    num_workers = 16 # how many workers for loading data\
    load_model_path = None
    load_latest = False
    dataset = "erhu"

#**********************************************************
# hyper params
# 
    lr = 1e-3
    weight_decay = 1e-5
    lr_decay = 0.8
    batch_size = 40
    mode = "train"
    parallel = False
    Device = "2"
    
    print_freq = 200

#**********************************************************
# path for data loader
#   
    train_raw = "data/raw/train"
    test_raw = "data/raw/test"

    train_gen = "data/gen/train"
    test_gen = "data/gen/test"

    technique = []

#**********************************************************
# len for training
# 
    train_len = 1500
    eval_len = 500
    total_len = 2000

    test_len = 1000

    classes = 0

    duration = 201
    time = 10.0

#**********************************************************
# path for data, label
# 这几个是干啥来着
    train_data_pth = "data/train_file.txt"
    train_label = "train_label.npy"
    test_data_pth = "data/test_file.txt"
    test_label = "test_label.npy"
    notes = ""

    test_ver = "real"
#**********************************************************
# type of feature
# 
    feature = "mel"
#**********************************************************
# add number of classes
# 
    def append_pth(self, pth):
        return os.path.join(pth,str(self.classes)+"_classes")

#**********************************************************
# input parameters
# 
    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        self.device = torch.device('cuda')

        if self.dataset == 'erhu':
            self.train_raw = self.append_pth(self.train_raw)
            self.test_raw = self.append_pth(self.test_raw)
            self.train_gen = self.append_pth(self.train_gen)
            self.test_gen = self.append_pth(self.test_gen)
            for it in [self.train_raw, self.test_raw, self.train_gen, self.test_gen]:
                if not os.path.exists(it):
                    os.mkdir(it)

#**********************************************************
# update technique list
#
            if self.classes == 4:
                self.technique = ["chanyin","huayin","dungong","others"]
            elif self.classes == 7:
                self.technique = ["chanyin","shanghuayin","xiahuayin","dungong","shangchanyin","lianxianhuayin","others"]
            elif self.classes == 11:
                self.technique = ["detache","diangong","harmonic","legato&slide&glissando","percussive","pizzicato","ricochet","staccato","tremolo","trill","vibrato"]
            # elif self.dataset == 'zhudi':
                # self.train_raw

        elif self.dataset == "zhudi":
            self.train_raw = "CBF-periDB/raw/train"
            self.test_raw = "CBF-periDB/raw/test"
            self.train_gen = "CBF-periDB/gen/train"
            self.test_gen = "CBF-periDB/gen/test"
            self.real_world = "CBF-periDB/real"
            self.iso_pth = "CBF-periDB/iso"


            self.real_num = 4
            self.iso_num = 12

            for it in [self.train_raw, self.test_raw, self.train_gen, self.test_gen]:
                if not os.path.exists(it):
                    os.mkdir(it)
            self.notes = "zhudi"

            if self.classes == 8:
                self.technique = ['FT', 'Tremolo', 'Trill', 'Vibrato', 'Acciacatura', 'Portamento', 'Glissando', 'others'] #here no others?
            if self.classes == 7:
                self.technique = ['FT', 'Tremolo', 'Trill', 'Vibrato', 'Acciacatura', 'Portamento', 'Glissando']

        self.total_len = self.train_len + self.eval_len


#**********************************************************
# print args
#
        print('+------------------------------------------------------+')
        print('|', 'user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print('|', k, getattr(self, k))
        print('+------------------------------------------------------+')


opt = Config()