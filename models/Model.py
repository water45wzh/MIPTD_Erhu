import torch
import torchvision
import torch.nn as nn
import torch.autograd.function as F
import torch.autograd.variable as variable
import torch.cuda
from .FCN import VGGNet,FCN8s,FCN32s
import time
import os

class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name = str(type(self))
    def get_model_name(self):
        return self.model_name
    def load(self,path):
        self.load_state_dict(torch.load(path))
#***********************************************************************
# save model, using name+time
#
    def save(self, notes):
        prefix = notes + 'check_points/' + self.model_name + '/'
        if not os.path.isdir(prefix):
            os.mkdir(prefix)
        name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        print('model name', name.split('/')[-1])
        torch.save(self.state_dict(), name)
        torch.save(self.state_dict(), prefix + 'latest.pth')
        return name
    def load_latest(self, notes):
        print(self.model_name)
        #=================================================================
        #path = 'check_points/' + self.model_name +notes+ '/latest.pth'
        path = notes + 'check_points/' + self.model_name + '/latest.pth'
        self.load_state_dict(torch.load(path))


#*************************************************************************
# fcn + mel
# 
class fcn_mel_4class(BasicModule):
    def __init__(self):
        super(fcn_mel_4class,self).__init__()
        self.VGG = VGGNet(requires_grad=True, model='vgg16')
        self.n_class = 4
        self.relu = nn.ReLU(inplace=True)
        self.bn0 = nn.BatchNorm2d(1)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=[3,3], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=[3,3], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=[3,4], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=[3,5], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn4 = nn.BatchNorm2d(32)
        self.deconv5 = nn.ConvTranspose2d(32, 16, kernel_size=[3,4], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn5 = nn.BatchNorm2d(16)
        self.classifier = nn.Conv2d(16, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        x =  self.VGG(x)
        x5 = x['x5']
        # print(x5)
        score = self.bn1(self.relu(self.deconv1(x5)))
        # print(score)
        score = self.bn2(self.relu(self.deconv2(score)))
        # print(score)
        score = self.bn3(self.relu(self.deconv3(score)))
        # print(score)
        score = self.bn4(self.relu(self.deconv4(score)))
        # print(score)
        score = self.bn5(self.relu(self.deconv5(score)))
        # print(score)
        score = torch.sigmoid(self.classifier(score)).squeeze(1)
        return self.softmax(score)

class fcn_mel_7class(BasicModule):
    def __init__(self):
        super(fcn_mel_7class,self).__init__()
        self.VGG = VGGNet(requires_grad=True, model='vgg16')
        self.n_class = 7
        self.relu = nn.ReLU(inplace=True)
        self.bn0 = nn.BatchNorm2d(1)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=[3,3], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=[3,3], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=[3,4], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=[3,5], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn4 = nn.BatchNorm2d(32)
        self.deconv5 = nn.ConvTranspose2d(32, 16, kernel_size=[3,4], stride=[2,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn5 = nn.BatchNorm2d(16)
        self.classifier = nn.Conv2d(16, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        x =  self.VGG(x)
        x5 = x['x5']
        # print(x5)
        score = self.bn1(self.relu(self.deconv1(x5)))
        # print(score)
        score = self.bn2(self.relu(self.deconv2(score)))
        # print(score)
        score = self.bn3(self.relu(self.deconv3(score)))
        # print(score)
        score = self.bn4(self.relu(self.deconv4(score)))
        # print(score)
        score = self.bn5(self.relu(self.deconv5(score)))
        # print(score)
        score = torch.sigmoid(self.classifier(score)).squeeze(1)
        return self.softmax(score)

# ********************************** for zhudi ***************************************
class fcn_mel_8class(BasicModule):
    def __init__(self):
        super(fcn_mel_8class,self).__init__()
        self.VGG = VGGNet(requires_grad=True, model='vgg16')
        self.n_class = 7
        self.relu = nn.ReLU(inplace=True)
        self.bn0 = nn.BatchNorm2d(1)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=[3,3], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=[3,3], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=[3,4], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=[3,5], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn4 = nn.BatchNorm2d(32)
        self.deconv5 = nn.ConvTranspose2d(32, 16, kernel_size=[4,4], stride=[2,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn5 = nn.BatchNorm2d(16)
        self.classifier = nn.Conv2d(16, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        x =  self.VGG(x)
        x5 = x['x5']
        # print(x5)
        score = self.bn1(self.relu(self.deconv1(x5)))
        # print(score)
        score = self.bn2(self.relu(self.deconv2(score)))
        # print(score)
        score = self.bn3(self.relu(self.deconv3(score)))
        # print(score)
        score = self.bn4(self.relu(self.deconv4(score)))
        # print(score)
        score = self.bn5(self.relu(self.deconv5(score)))
        # print(score)
        score = torch.sigmoid(self.classifier(score)).squeeze(1)
        return self.softmax(score)

class fcn_mel_11class(BasicModule):
    def __init__(self):
        super(fcn_mel_11class,self).__init__()
        self.VGG = VGGNet(requires_grad=True, model='vgg13')
        # self.pool = nn.MaxPool2d(kernel_size=[1,4])
        self.n_class = 11
        self.relu = nn.ReLU(inplace=True)
        self.bn0 = nn.BatchNorm2d(1)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=[3,3], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=[3,3], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=[3,4], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=[4,5], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn4 = nn.BatchNorm2d(32)
        self.deconv5 = nn.ConvTranspose2d(32, 16, kernel_size=[5,4], stride=[2,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn5 = nn.BatchNorm2d(16)
        self.classifier = nn.Conv2d(16, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        x =  self.VGG(x)
        x5 = x['x5']
        #x5 = self.pool(x5)
        print(x5.shape)
        score = self.bn1(self.relu(self.deconv1(x5)))
        # print(score.shape)
        score = self.bn2(self.relu(self.deconv2(score)))
        # print(score.shape)
        score = self.bn3(self.relu(self.deconv3(score)))
        # print(score.shape)
        score = self.bn4(self.relu(self.deconv4(score)))
        # print(score.shape)
        score = self.bn5(self.relu(self.deconv5(score)))
        # print(score.shape)
        score = torch.sigmoid(self.classifier(score)).squeeze(1)
        return self.softmax(score)

#*************************************************************************
# fcn + cqt
# 
class fcn_cqt_4class(BasicModule):
    def __init__(self):
        super(fcn_cqt_4class,self).__init__()
        self.VGG = VGGNet(requires_grad=True, model='vgg16')
        self.n_class = 4
        self.relu = nn.ReLU(inplace=True)
        self.bn0 = nn.BatchNorm2d(1)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=[3,3], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=[3,3], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=[4,4], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=[4,5], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn4 = nn.BatchNorm2d(32)
        self.deconv5 = nn.ConvTranspose2d(32, 16, kernel_size=[3,4], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn5 = nn.BatchNorm2d(16)
        self.classifier = nn.Conv2d(16, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        x =  self.VGG(x)
        x5 = x['x5']
        # print(x5)
        score = self.bn1(self.relu(self.deconv1(x5)))
        # print(score.shape)
        score = self.bn2(self.relu(self.deconv2(score)))
        # print(score.shape)
        score = self.bn3(self.relu(self.deconv3(score)))
        # print(score.shape)
        score = self.bn4(self.relu(self.deconv4(score)))
        # print(score.shape)
        score = self.bn5(self.relu(self.deconv5(score)))
        # print(score.shape)
        score = torch.sigmoid(self.classifier(score)).squeeze(1)
        return self.softmax(score)

class fcn_cqt_7class(BasicModule):
    def __init__(self):
        super(fcn_cqt_7class,self).__init__()
        self.VGG = VGGNet(requires_grad=True, model='vgg16')
        self.n_class = 7
        self.relu = nn.ReLU(inplace=True)
        self.bn0 = nn.BatchNorm2d(1)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=[3,3], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=[3,3], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=[4,4], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=[4,5], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn4 = nn.BatchNorm2d(32)
        self.deconv5 = nn.ConvTranspose2d(32, 16, kernel_size=[3,4], stride=[2,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn5 = nn.BatchNorm2d(16)
        self.classifier = nn.Conv2d(16, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        x =  self.VGG(x)
        x5 = x['x5']
        # print(x5)
        score = self.bn1(self.relu(self.deconv1(x5)))
        # print(score)
        score = self.bn2(self.relu(self.deconv2(score)))
        # print(score)
        score = self.bn3(self.relu(self.deconv3(score)))
        # print(score)
        score = self.bn4(self.relu(self.deconv4(score)))
        # print(score)
        score = self.bn5(self.relu(self.deconv5(score)))
        # print(score)
        score = torch.sigmoid(self.classifier(score)).squeeze(1)
        return self.softmax(score)

class fcn_cqt_11class(BasicModule):
    def __init__(self):
        super(fcn_cqt_11class,self).__init__()
        self.VGG = VGGNet(requires_grad=True, model='vgg13')
        # self.pool = nn.MaxPool2d(kernel_size=[1,4])
        self.n_class = 11
        self.relu = nn.ReLU(inplace=True)
        self.bn0 = nn.BatchNorm2d(1)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=[3,3], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=[3,3], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=[4,4], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=[5,5], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn4 = nn.BatchNorm2d(32)
        self.deconv5 = nn.ConvTranspose2d(32, 16, kernel_size=[5,4], stride=[2,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn5 = nn.BatchNorm2d(16)
        self.classifier = nn.Conv2d(16, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        x =  self.VGG(x)
        x5 = x['x5']
        #x5 = self.pool(x5)
        print(x5.shape)
        score = self.bn1(self.relu(self.deconv1(x5)))
        # print(score.shape)
        score = self.bn2(self.relu(self.deconv2(score)))
        # print(score.shape)
        score = self.bn3(self.relu(self.deconv3(score)))
        # print(score.shape)
        score = self.bn4(self.relu(self.deconv4(score)))
        # print(score.shape)
        score = self.bn5(self.relu(self.deconv5(score)))
        # print(score.shape)
        score = torch.sigmoid(self.classifier(score)).squeeze(1)
        return self.softmax(score)

#*************************************************************************
# fcn + mel + cqt
# 
class fcn_mel_plus_cqt_11class(BasicModule):
    def __init__(self):
        super(fcn_mel_plus_cqt_11class,self).__init__()
        self.VGG = VGGNet(requires_grad=True, model='vgg13')
        # self.pool = nn.MaxPool2d(kernel_size=[1,4])
        self.n_class = 11
        self.relu = nn.ReLU(inplace=True)
        self.bn0 = nn.BatchNorm2d(1)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=[3,3], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=[3,3], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=[3,4], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=[3,5], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn4 = nn.BatchNorm2d(32)
        self.deconv5 = nn.ConvTranspose2d(32, 16, kernel_size=[3,4], stride=[2,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn5 = nn.BatchNorm2d(16)
        self.classifier = nn.Conv2d(16, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

        # self.deconv6 = 
    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        x1 = self.VGG(x1)
        x2 = self.VGG(x2)
        x5_1 = x1['x5']
        x5_2 = x2['x5']
        x5 = torch.cat((x5_1, x5_2), 2)
        # print(x5.shape)
        score = self.bn1(self.relu(self.deconv1(x5)))
        # print(score.shape)
        score = self.bn2(self.relu(self.deconv2(score)))
        # print(score.shape)
        score = self.bn3(self.relu(self.deconv3(score)))
        # print(score.shape)
        score = self.bn4(self.relu(self.deconv4(score)))
        # print(score.shape)
        score = self.bn5(self.relu(self.deconv5(score)))
        # print(score.shape)
        score = torch.sigmoid(self.classifier(score)).squeeze(1)
        return self.softmax(score)
 
#*************************************************************************
# cnn + mel
# 
class cnn_mel_4class(BasicModule):
    def __init__(self):
        super(cnn_mel_4class,self).__init__()

        self.conv1 = nn.Conv2d(1, 7, stride = 1, kernel_size = (3, 20), padding=(1, 9))
        self.conv2 = nn.Conv2d(1, 7, stride = 1, kernel_size = (20, 3), padding=(9, 1))
        self.conv3 = nn.Conv2d(1, 7, stride = 1, kernel_size = (20, 20), padding=(9, 9))
        
        self.pad1 = nn.ZeroPad2d((0,1,0,0))
        self.pad2 = nn.ZeroPad2d((0,0,0,1))
        self.pad3 = nn.ZeroPad2d((0,1,0,1))
        
        self.bn1 = nn.BatchNorm2d(7)
        self.bn2 = nn.BatchNorm2d(21)
        self.bn3 = nn.BatchNorm2d(16)
        
        self.conv = nn.Conv2d(7, 21, stride = 1, kernel_size = (3, 3))
        self.conv4 = nn.Conv2d(21, 21, stride = 1, kernel_size = (3, 3))
        #这个pool的地方找liangbeici的代码研究一下，需要用不一样的pooling实例？
        self.pool_small = nn.MaxPool2d(2)
        self.pool_large = nn.MaxPool2d(4)

        self.relu = nn.ReLU(inplace = True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, None))
        # self.classifier = nn.Linear(1155, 4)

        self.softmax = nn.Softmax(dim=1)

        # 4 classes deconv layers
        self.deconv1 = nn.ConvTranspose2d(21, 16, kernel_size=[1,7], stride=[1,4], padding=1, dilation=1, output_padding=[0,1])
        self.deconv2 = nn.ConvTranspose2d(16, 16, kernel_size=[1,6], stride=[1,3], padding=1, dilation=1, output_padding=[0,1])
        self.deconv3 = nn.ConvTranspose2d(16, 16, kernel_size=[2,3], stride=[1,3], padding=2, dilation=1, output_padding=[0,1])

        self.classifier = nn.Conv2d(16, 1, kernel_size=1)

        self.cnn_model = nn.Sequential(
                        nn.BatchNorm2d(7),
                        nn.ReLU(inplace = True),
                        nn.MaxPool2d(2),
                        nn.Dropout(p=0.25),

                        nn.Conv2d(7, 21, stride = 1, kernel_size = (3, 3)),
                        nn.BatchNorm2d(21),
                        nn.ReLU(inplace = True),
                        nn.MaxPool2d(2),
                        nn.Dropout(p=0.25),

                        nn.Conv2d(21, 21, stride = 1, kernel_size = (3, 3)),
                        nn.BatchNorm2d(21),
                        nn.ReLU(inplace = True),
                        nn.MaxPool2d(2),
                        nn.Dropout(p=0.25),

                        nn.Conv2d(21, 21, stride = 1, kernel_size = (3, 3)),
                        nn.BatchNorm2d(21),
                        nn.ReLU(inplace = True),
                        nn.MaxPool2d(4),
                        nn.Dropout(p=0.25),

                        nn.ConvTranspose2d(21, 16, kernel_size=[1,7], stride=[1,4], padding=1, dilation=1, output_padding=[0,1]),
                        nn.ReLU(inplace = True),
                        nn.BatchNorm2d(16),

                        nn.ConvTranspose2d(16, 16, kernel_size=[1,6], stride=[1,3], padding=1, dilation=1, output_padding=[0,1]),
                        nn.ReLU(inplace = True),
                        nn.BatchNorm2d(16),

                        nn.ConvTranspose2d(16, 16, kernel_size=[2,3], stride=[1,3], padding=2, dilation=1, output_padding=[0,1]),
                        nn.ReLU(inplace = True),
                        nn.BatchNorm2d(16)
                    )
    def forward(self,x):
        #print(x.shape)
        x1 = self.pad1(self.conv1(x))
        x2 = self.pad2(self.conv2(x))
        x3 = self.pad3(self.conv3(x))

        score = self.cnn_model(torch.cat((x1, x2, x3), 2))

        score = torch.sigmoid(self.classifier(score)).squeeze(1)

        return self.softmax(score)

class cnn_mel_7class(BasicModule):
    def __init__(self):
        super(cnn_mel_7class,self).__init__()

        self.conv1 = nn.Conv2d(1, 7, stride = 1, kernel_size = (3, 20), padding=(1, 9))
        self.conv2 = nn.Conv2d(1, 7, stride = 1, kernel_size = (20, 3), padding=(9, 1))
        self.conv3 = nn.Conv2d(1, 7, stride = 1, kernel_size = (20, 20), padding=(9, 9))
        
        self.pad1 = nn.ZeroPad2d((0,1,0,0))
        self.pad2 = nn.ZeroPad2d((0,0,0,1))
        self.pad3 = nn.ZeroPad2d((0,1,0,1))
        
        self.softmax = nn.Softmax(dim=1)
        self.classifier = nn.Conv2d(16, 1, kernel_size=1)

        self.cnn_model = nn.Sequential(
                        nn.BatchNorm2d(7),
                        nn.ReLU(inplace = True),
                        nn.MaxPool2d(2),
                        nn.Dropout(p=0.25),

                        nn.Conv2d(7, 21, stride = 1, kernel_size = (3, 3)),
                        nn.BatchNorm2d(21),
                        nn.ReLU(inplace = True),
                        nn.MaxPool2d(2),
                        nn.Dropout(p=0.25),

                        nn.Conv2d(21, 21, stride = 1, kernel_size = (3, 3)),
                        nn.BatchNorm2d(21),
                        nn.ReLU(inplace = True),
                        nn.MaxPool2d(2),
                        nn.Dropout(p=0.25),

                        nn.Conv2d(21, 21, stride = 1, kernel_size = (3, 3)),
                        nn.BatchNorm2d(21),
                        nn.ReLU(inplace = True),
                        nn.MaxPool2d(4),
                        nn.Dropout(p=0.25),

                        nn.ConvTranspose2d(21, 16, kernel_size=[1,7], stride=[1,4], padding=1, dilation=1, output_padding=[0,1]),
                        nn.ReLU(inplace = True),
                        nn.BatchNorm2d(16),

                        nn.ConvTranspose2d(16, 16, kernel_size=[1,6], stride=[2,3], padding=1, dilation=1, output_padding=[0,1]),
                        nn.ReLU(inplace = True),
                        nn.BatchNorm2d(16),

                        nn.ConvTranspose2d(16, 16, kernel_size=[2,3], stride=[1,3], padding=2, dilation=1, output_padding=[0,1]),
                        nn.ReLU(inplace = True),
                        nn.BatchNorm2d(16)
                    )
    def forward(self,x):
        #print(x.shape)
        x1 = self.pad1(self.conv1(x))
        x2 = self.pad2(self.conv2(x))
        x3 = self.pad3(self.conv3(x))

        score = self.cnn_model(torch.cat((x1, x2, x3), 2))
        print(score.shape)
        score = torch.sigmoid(self.classifier(score)).squeeze(1)

        return self.softmax(score)

class cnn_mel_11class(BasicModule):
    def __init__(self):
        super(cnn_mel_11class,self).__init__()

        self.conv1 = nn.Conv2d(1, 7, stride = 1, kernel_size = (3, 20), padding=(1, 9))
        self.conv2 = nn.Conv2d(1, 7, stride = 1, kernel_size = (20, 3), padding=(9, 1))
        self.conv3 = nn.Conv2d(1, 7, stride = 1, kernel_size = (20, 20), padding=(9, 9))
        
        self.pad1 = nn.ZeroPad2d((0,1,0,0))
        self.pad2 = nn.ZeroPad2d((0,0,0,1))
        self.pad3 = nn.ZeroPad2d((0,1,0,1))
        
        self.softmax = nn.Softmax(dim=1)

        self.classifier = nn.Conv2d(16, 1, kernel_size=1)

        self.cnn_model = nn.Sequential(
                        nn.BatchNorm2d(7),
                        nn.ReLU(inplace = True),
                        nn.MaxPool2d(2),
                        nn.Dropout(p=0.25),

                        nn.Conv2d(7, 21, stride = 1, kernel_size = (3, 3)),
                        nn.BatchNorm2d(21),
                        nn.ReLU(inplace = True),
                        nn.MaxPool2d(2),
                        nn.Dropout(p=0.25),

                        nn.Conv2d(21, 21, stride = 1, kernel_size = (3, 3)),
                        nn.BatchNorm2d(21),
                        nn.ReLU(inplace = True),
                        nn.MaxPool2d(2),
                        nn.Dropout(p=0.25),

                        nn.Conv2d(21, 21, stride = 1, kernel_size = (3, 3)),
                        nn.BatchNorm2d(21),
                        nn.ReLU(inplace = True),
                        nn.MaxPool2d(4),
                        nn.Dropout(p=0.25),

                        nn.ConvTranspose2d(21, 16, kernel_size=[1,7], stride=[1,4], padding=1, dilation=1, output_padding=[0,1]),
                        nn.ReLU(inplace = True),
                        nn.BatchNorm2d(16),

                        nn.ConvTranspose2d(16, 16, kernel_size=[1,6], stride=[1,3], padding=1, dilation=1, output_padding=[0,1]),
                        nn.ReLU(inplace = True),
                        nn.BatchNorm2d(16),

                        nn.ConvTranspose2d(16, 16, kernel_size=[3,3], stride=[2,3], padding=2, dilation=1, output_padding=[0,1]),
                        nn.ReLU(inplace = True),
                        nn.BatchNorm2d(16)
                    )
    def forward(self,x):
        #print(x.shape)
        x1 = self.pad1(self.conv1(x))
        x2 = self.pad2(self.conv2(x))
        x3 = self.pad3(self.conv3(x))

        score = self.cnn_model(torch.cat((x1, x2, x3), 2))
        # print(score.shape)
        score = torch.sigmoid(self.classifier(score)).squeeze(1)

        return self.softmax(score)

#*************************************************************************
# fcn + mel
# 
class myNet8(BasicModule):
    def __init__(self):
        super(myNet8,self).__init__()
        self.VGG = nn.Sequential(
            nn.Conv2d(kernel_size=3,in_channels=1,out_channels=64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(kernel_size=3,in_channels=64,out_channels=128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(kernel_size=3,in_channels=128,out_channels=256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(kernel_size=3,in_channels=256,out_channels=512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.rnn_block=nn.Sequential(
            nn.GRU(128,64,5,dropout=0.5,bidirectional=True)
        )
        self.fc_block = nn.Sequential(
            nn.Linear(256*12,3)
        )
        self.pool = nn.AdaptiveAvgPool2d((None,1))
        self.classifier = nn.Conv1d(201, 201, 3, stride=2) #means in_channels, out_channels, kernel_size
        self.classifier1 = nn.Conv1d(201, 201, 4)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        #x =  self.VGG(x1)
        # print(x.shape)
        x = x.permute(0,3,2,1)
        x = x.view(x.shape[0],x.shape[1],-1)
        # print(x.shape)
        x, _ = self.rnn_block(x)
        # print(x.shape)
        #x = torch.unsqueeze(x, 2)
        #print(x.shape)
        x = self.relu(self.classifier(x))
        # print(x.shape)
        x = self.relu(self.classifier(x))
        # print(x.shape)
        x = self.relu(self.classifier(x))
        # print(x.shape)
        x = self.relu(self.classifier(x))
        # print(x.shape)
        x = self.relu(self.classifier1(x))
        # print(x.shape)
        x = torch.sigmoid(x.permute(0,2,1))
        # print(x.shape)
        # x = x.view(x.size()[0],-1)
        # x =  self.fc_block(x)
        # x = torch.sigmoid(x)
        return self.softmax(x)

class rnn_mel_4class(BasicModule):
    def __init__(self):
        super(rnn_mel_4class,self).__init__()
        self.VGG = nn.Sequential(
            nn.Conv2d(kernel_size=3,in_channels=1,out_channels=64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(kernel_size=3,in_channels=64,out_channels=128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(kernel_size=3,in_channels=128,out_channels=256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(kernel_size=3,in_channels=256,out_channels=512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.rnn_block=nn.Sequential(
            nn.GRU(128,128)
        )
        self.fc_block = nn.Sequential(
            nn.Linear(256*12,3)
        )
        self.classifier = nn.Sequential(
            nn.Conv1d(201, 201, 3, stride=2),
            nn.BatchNorm1d(201),
            nn.ReLU(),
            nn.Conv1d(201, 201, 3, stride=2),
            nn.BatchNorm1d(201),
            nn.ReLU(),
            nn.Conv1d(201, 201, 3, stride=2),
            nn.BatchNorm1d(201),
            nn.ReLU(),
            nn.Conv1d(201, 201, 3, stride=2),
            nn.BatchNorm1d(201),
            nn.ReLU(),
            nn.Conv1d(201, 201, 4),
            nn.BatchNorm1d(201),
            nn.ReLU(),
        )
        # self.pool = nn.AdaptiveAvgPool2d((None,1))
        # self.classifier = nn.Conv1d(201, 201, 3, stride=2) #means in_channels, out_channels, kernel_size
        # self.classifier1 = nn.Conv1d(201, 201, 4)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        #x =  self.VGG(x1)
        # print(x.shape)
        x = x.permute(0,3,2,1)
        x = x.view(x.shape[0],x.shape[1],-1)
        # print(x.shape)
        x, _ = self.rnn_block(x)
        # print(x.shape)
        x = self.classifier(x)
        x = torch.sigmoid(x.permute(0,2,1))
        # print(x.shape)
        # x = x.view(x.size()[0],-1)
        # x =  self.fc_block(x)
        # x = torch.sigmoid(x)
        return self.softmax(x)