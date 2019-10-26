# -*- coding: utf-8 -*-
import librosa as lb
import torch
import torch.utils.data
import os
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import tqdm
import multiprocessing as mp
from multiprocessing import Process
from multiprocessing import Pool
import os, time, random
from tqdm import trange
import pydub
from pydub import AudioSegment
import pickle
from multiprocessing import Process
class primary_to_numpy:

    def __init__(self,train_file='../test/train',test_file='../test'):
        self.train_file = train_file
        self.test_file  = test_file
    def cut_audio(self,rootdir,cut_len,is_front,target_dir):
        new_data_file_list = []
        for i in rootdir:
            sound = AudioSegment.from_wav(i)
            if is_front==False:
                sound = sound[:-cut_len]
            else:
                sound = sound[cut_len:]
            file_name = i.split('/')[-1]
            print(file_name)
            new_data_file_list.append(target_dir+"/"+file_name)
            sound.export(target_dir+"/"+file_name,format="wav")

        return new_data_file_list
    def get_filelist(self,train_or_test):
        if train_or_test == 'train':
            rootdir = self.train_file
        else:
            rootdir = self.test_file
        allfile = os.listdir(rootdir)

        wav_file_list,txt_file_list = [],[]
        for tempfile in allfile:
            if tempfile.find("wav",-4) != -1:
                wav_file_list.append(rootdir+'/'+tempfile)
            elif tempfile.find("txt",-4)!= -1:
                txt_file_list.append(rootdir+'/'+tempfile)
        wav_file_list.sort(),txt_file_list.sort()

        return wav_file_list,txt_file_list
    def train_wav_2_numpy(self,train_wav):
        print(train_wav)
        temp_y,temp_sr = lb.load(train_wav)
        temp_logmel_spect = librosa.power_to_db(lb.feature.melspectrogram(temp_y,n_fft=22050,hop_length=22050,n_mels=128))
        np.save("train_mel_npy/"+train_wav[22:-4]+".npy",temp_logmel_spect)
    def test_wav_2_numpy(self,test_wav):
        print(test_wav)
        print("test_mel_npy/"+test_wav[16:-4]+".npy")
        temp_y,temp_sr = lb.load(test_wav)
        temp_logmel_spect = librosa.power_to_db(lb.feature.melspectrogram(temp_y,n_fft=22050,hop_length=22050,n_mels=128))
        np.save("test_mel_npy/"+test_wav[16:-4]+".npy",temp_logmel_spect)

    def train_wav_2_numpy_50ms(self, train_wav):
        print(train_wav)
        temp_y, temp_sr = lb.load(train_wav)
        temp_logmel_spect = np.abs(librosa.power_to_db(
            lb.feature.melspectrogram(temp_y, n_fft=2048, hop_length=1045, n_mels=128)))
        mean_size = 21
        height, length = temp_logmel_spect.shape
        new_temp_logmel_spect = np.zeros((height, int(length / mean_size)), dtype=np.float64)
        for i in range(int(length / mean_size)):
            new_temp_logmel_spect[:, i] = temp_logmel_spect[:, i * mean_size:(i + 1) * mean_size].mean(axis=1)
        np.save("data/train_mel_50ms/" + train_wav[22:-4] + ".npy", new_temp_logmel_spect)

    def test_wav_2_numpy_50ms(self, test_wav):
        print(test_wav)

        temp_y, temp_sr = lb.load(test_wav)
        temp_logmel_spect = np.abs(librosa.power_to_db(
            lb.feature.melspectrogram(temp_y, n_fft=2048, hop_length=1045, n_mels=128)))
        mean_size = 21
        height, length = temp_logmel_spect.shape
        new_temp_logmel_spect = np.zeros((height, int(length / mean_size)), dtype=np.float64)
        for i in range(int(length / mean_size)):
            new_temp_logmel_spect[:, i] = temp_logmel_spect[:, i * mean_size:(i + 1) * mean_size].mean(axis=1)
        print(new_temp_logmel_spect.shape)
        np.save("data/test_mel_50ms/" + test_wav[16:-4] + ".npy", new_temp_logmel_spect)
    def train_wav_2_numpy_25ms_unmean(self, train_wav):
        #print(train_wav)
        temp_y, temp_sr = lb.load(train_wav)
        temp_logmel_spect = np.abs(librosa.power_to_db(
            lb.feature.melspectrogram(temp_y,  n_fft=2048, hop_length=511, n_mels=128)))
        print(train_wav[22:-4],train_wav)
        np.save("data/train_mel_25ms_unmean/" + train_wav[22:-4] + ".npy",  temp_logmel_spect)

    def test_wav_2_numpy_25ms_unmean(self, test_wav):
        #print(test_wav)

        temp_y, temp_sr = lb.load(test_wav)
        temp_logmel_spect = np.abs(librosa.power_to_db(
            lb.feature.melspectrogram(temp_y, n_fft=2048, hop_length=511, n_mels=128)))
        print(test_wav[16:-4],test_wav )
        np.save("data/test_mel_25ms_unmean/" + test_wav[16:-4] + ".npy",  temp_logmel_spect)
    def save_file_list(self,rootdir,target_pth):
        file_list = os.listdir(rootdir)
        with open(target_pth,"w+") as fr:
            for i in range(len(file_list)):
                fr.writelines(rootdir+'/'+file_list[i]+'\n')
    def txt_2_lable(self,label_txt,train_or_test):
        """
        返回形式为[N,10,362]的numpy数组
        """
        class_id_dic={}
        cut = 0
        if train_or_test =="train":
            cut = 19
        elif train_or_test =="test":
            cut =13
        with open("class_id.txt", "r") as fr:
            line_list = fr.readlines()
            for line in line_list:
                classkey = line.split(' ')[0]
                value = float(line.split(' ')[1][0])
                class_id_dic[classkey] = value
        label_ndarray_dic={}
        for i in range(len(label_txt)):
            label_ndarray = np.zeros([ 10, 362])
            # print("Load:",label_txt[i])
            with open(label_txt[i], "r", encoding='gb2312') as fr:
                line_list = fr.readlines()
                for line_i in line_list:
                    line_i_split = line_i.split('--')
                    i_temp_class = int(class_id_dic[line_i_split[0]])
                    temp_label = [j.split(';')[0] for j in line_i_split[1:]]
                    for temp_label_i in temp_label:
                        start, end = temp_label_i.split(',')
                        start = int(float(start)) - 1
                        end = int(float(end))
                        label_ndarray[i_temp_class, start:end] = 1
            print(label_txt[i][cut:-4],"  ",label_txt[i])
            label_ndarray_dic[label_txt[i][cut:-4]] = label_ndarray

        # print(label_ndarray_dic.keys())

        return label_ndarray_dic
class music_2_npy():
    def __init__(self, train_file='../test/train', test_file='../test'):
        self.train_file = train_file
        self.test_file = test_file
        self.pn = primary_to_numpy()
        self.id=0
    def write_data_path_2_txt(self):
        train_wavpth_list = os.listdir(self.train_file)
        test_wavpth_list = os.listdir(self.test_file)
        fw,fl = open("data/file/train_wav_pth_list.txt","w+"), open("data/file/train_label_pth_list.txt","w+")
        for i in range(len(train_wavpth_list)):
            if train_wavpth_list[i][-3:] =='wav':
                fw.writelines(self.train_file+'/'+train_wavpth_list[i]+'\n')
            elif train_wavpth_list[i][-3:] == 'txt':
                fl.writelines(self.train_file + '/' + train_wavpth_list[i] + '\n')
        fw.close(),fl.close()
        fw,fl = open("data/file/test_wav_pth_list.txt","w+"), open("data/file/test_label_pth_list.txt","w+")
        for i in range(len(test_wavpth_list)):
            if test_wavpth_list[i][-3:] == "wav":
                fw.writelines(self.test_file+'/'+test_wavpth_list[i]+'\n')
            elif test_wavpth_list[i][-3:] =='txt':
                fl.writelines(self.test_file+'/'+test_wavpth_list[i]+'\n')
            else:
                pass
        fw.close(),fl.close()
    def music_clip_2_npy(self,par):
        train_or_test = "train"
        file_list, label_list, id =par
        # with open(file_list,"r") as f:
        #     wav_file_list = f.readlines()
        # print("read finishde")
        # for i in range(len(wav_file_list)):
        #     print(wav_file_list[i][:-1])
        #sound =  AudioSegment.from_wav(file_list[:-1])
        y,sr = lb.load(file_list[:-1])
        print(file_list[:-1]," ",label_list[:-1])
        class_id_dic = {}
        cut = 0
        if train_or_test == "train":
            cut = 19
            #target_pth = "data/train"
            target_pth = "data/train"
        elif train_or_test == "test":
            cut = 13
            #target_pth = "data/test"
            target_pth = "../test"
        with open("class_id.txt", "r") as fr:
            line_list = fr.readlines()
            for line in line_list:
                classkey = line.split(' ')[0]
                value = float(line.split(' ')[1][0])
                class_id_dic[classkey] = value
        label_ndarray = np.zeros([10, 361])
            # print("Load:",label_txt[i])
        with open(label_list[:-1], "r", encoding='gb2312') as fr:
            line_list = fr.readlines()
            for line_i in line_list:
                line_i_split = line_i.split('--')
                i_temp_class = int(class_id_dic[line_i_split[0]])
                temp_label = [j.split(';')[0] for j in line_i_split[1:]]
                for temp_label_i in temp_label:
                        start, end = temp_label_i.split(',')
                        start = int(float(start)) - 1
                        end = int(float(end))
                        if end>361:
                            end=361
                        label_ndarray[i_temp_class, start:end] = 1
        label_ndarray =label_ndarray.swapaxes(1,0)
    # print(label_ndarray.shape)
        head = 0
        coun=0
        #print(len(y))
        cid = id*361

        data_label_tuple =[]
        for i in range(0,len(y),22050):
            data =y[i:i+22050]
            logmel = librosa.power_to_db(
            lb.feature.melspectrogram(data, n_fft=1024, hop_length=512, n_mels=128))
            label = label_ndarray[int(i/22050)].squeeze()
            #print(label.shape)
            #print("save_path:",target_pth+'/'+str(cid+int(i/22050))+'.npz')
            #np.savez(target_pth+'/'+str(cid+int(i/22050))+'.npz',data=data,label=label)
            data_label_tuple.append((logmel,label))
            #print(int(i/22050),"   ",label)
        with open('data/train/'+str(id)+".pk","wb") as f:
            pickle.dump(data_label_tuple,f,protocol=pickle.HIGHEST_PROTOCOL)
if __name__=='__main__':
    x = primary_to_numpy()
    # train_wav,train_label_txt = x.get_filelist("train")
    # test_wav,test_label_txt = x.get_filelist("test")
    # x.cut_audio(test_wav,target_dir="data/cut_audio/test",cut_len=2,is_front=False)
    x.save_file_list("data/train","data/train_file.txt")
    y = music_2_npy()
    with open("data/file/train_wav_pth_list.txt","r") as f:
        wav_file_list = f.readlines()
    with open("data/file/train_label_pth_list.txt","r") as s:
        label_file_list = s.readlines()
    wav_file_list.sort(),label_file_list.sort()
   # v = [(wav_file_list[i],label_file_list[i],i) for i in range(1,len(wav_file_list),2)]
    # for i in range(len(wav_file_list)):
    #     print(wav_file_list[i][22:-4]," ",test_file_list[i][19:-4])
    #     if wav_file_list[i][22:-4] != test_file_list[i][19:-4]:
    #         print("error")
    #y.music_clip_2_npy(wav_file_list[0],label_file_list[0],id=0,train_or_test="train")
    #y.write_data_path_2_txt()
    # pool = Pool(8)
    # pool.map(y.music_clip_2_npy,v)
    # # for i in range(0,len(wav_file_list),2):
    # #     pool.apply(y.music_clip_2_npy, (wav_file_list[i],label_file_list[i],i,"train",))
    # pool.close()
    # pool.join()
   # p1=Process(y.music_clip_2_npy,())
   #x.txt_2_lable(train_label_t xt,"train")
    # np.save("test_label.npy",x.txt_2_lable(test_label_txt,"test"))
    # train_label = np.load("train_label.npy")
    # print(train_label[()]["110"].shape)


