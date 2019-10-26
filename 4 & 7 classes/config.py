import configparser
import torch
config = configparser.ConfigParser()


class Config(object):

    model = "myNet2"
    lr = 0.0002
    max_epoch = 2000
    num_workers = 16 # how many workers for loading data\
    load_model_path = None
    load_latest = False
    weight_decay = 1e-5
    lr_decay = 0.8
    batch_size = 40
    manner = "train"
    parallel = False
    Device = "2"
    print_freq = 500
    train_data_pth = "data/train_file.txt"
    train_label = "train_label.npy"
    test_data_pth = "data/test_file.txt"
    test_label = "test_label.npy"
    notes = None
    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        self.device = torch.device('cuda')
        print('+------------------------------------------------------+')
        print('|', 'user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print('|', k, getattr(self, k))
        print('+------------------------------------------------------+')


opt = Config()