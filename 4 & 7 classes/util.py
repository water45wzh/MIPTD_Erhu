import os
import numpy as np
import torch
import pandas
import re
import data_loader
class eval(object):
    def __init__(self,rootdir =None):
        return
    def segment_eva(self,score,target):
       # print(score.shape," ",target.shape)

        TN ,TP ,FN ,FP = 0.0,0.0,0.0,0.0
        for i in range(len(score)):
            if score[i] >0.5:
                if target[i] == 1:
                    TP+=1
                else:
                    FP+=1
            else:
                if target[i]==1:
                    FN+=1
                else:
                    TN+=1
        acc = (TP+TN)/(TP+TN+FP+FN)
        err = (FN+FP)/(TP+TN+FP+FN)
        recall = TP/(TP+FN+0.0000001)
        precision = TP/(TP+FP+0.0000001)
        F1 = (2*recall*precision)/(recall + precision+0.0000001)
        return acc,err,recall,precision,F1
    def produce_event_dic(self,data):
        """
        根据模型段级预测产生对应的事件级预测
        :param data:
            data 满足大小应该为 362*10
        :return:
            event_dic ={key:value} 以字典形式包含了所有发生的事件的边界及事件类别
                key = 事件类别
                value = 对应事件发生起点和终点 [[begin,end],[begin,end],..]

        """
        if type(data)!= type(np.ndarray([0])):
            data.numpy()
        data.reshape(10,320).astype(np.int)
        event_list ={}
        for i in range(len(data)):
            temp_seg = self.transform_list(data[i])
            event_list['i'] = temp_seg

            # for j in range(len(data[i])):
        return event_list
    def transform_list(self,s):
        """
        返回传入列表中元素连续为1的字串在列表中的起始位置
        :param s:
            元素值为 0 或 1 的列表 [ 0 , 0 , 1 , ... ,]
        :return:
            列表中元素连续为1的字串在列表中的起始位置 [[begin,end],[begin,end],..]
        """
        s = [str(i) for i in s]
        s = "".join(s)
        print(s)
        begin = [i.start() + 1 for i in re.finditer('01', s)]
        if s[0] == '1':
            begin.insert(0, 0)
        t = s.split('0')
        offset = [ len(i) - 1  for i in t if i != '']
        if len(begin) == len(offset):
            return [[begin[i],begin[i]+offset[i]] for i in range(len(begin))]
        else:
            print(len(begin))
            print(len(offset))
            print("Error! -- len(begin)!=len(offset)")
            assert len(begin) == len(offset)
    def produce_eval_csv(self,score,target):
        if type(score)!= type(np.ndarray(1)):
            score,target = score.numpy(),target.numpy()

if __name__ =="__main__":
    s = [1,0,0,0,0,1,1,1,0,0,1,1,0,0,1,1,0,1,0]
    eval = eval("eval")
    print(eval.transform_list(s))