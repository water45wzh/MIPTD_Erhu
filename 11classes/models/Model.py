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
    def save(self):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        prefix = 'check_points/' + self.model_name + '/'
        if not os.path.isdir(prefix):
            os.mkdir(prefix)
        name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        print('model name', name.split('/')[-1])
        torch.save(self.state_dict(), name)
        torch.save(self.state_dict(), prefix + 'latest.pth')
        return name
    def load_latest(self, notes):
        path = 'check_points/' + self.model_name +notes+ '/latest.pth'
        self.load_state_dict(torch.load(path))


class myNet1(BasicModule):
    def __init__(self,nclass):
        super(myNet1,self).__init__()
        self.VGG = VGGNet(requires_grad=True, model='vgg13')
        self.pool = nn.MaxPool2d(kernel_size=[1,4])
        self.n_class = nclass
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=[3,3], stride=[2,1], padding=1, dilation=1, output_padding=[1,0])
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=[3,3], stride=[2,1], padding=1, dilation=1, output_padding=[1,0])
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=[3,3], stride=[2,1], padding=1, dilation=1, output_padding=[1,0])
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=[5,3], stride=[2,1], padding=1, dilation=1, output_padding=[1,0])
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=[5,3], stride=[2,1], padding=1, dilation=1, output_padding=[1,0])
        self.bn5 = nn.BatchNorm2d(32)
        self.deconv6 = nn.ConvTranspose2d(32, 32, kernel_size=[7,3], stride=1, padding=1, dilation=1, output_padding=0)
        self.bn6 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, nclass, kernel_size=1)
    def forward(self,x):
        x =  self.VGG(x)
        x5 = x['x5']
        x5 = self.pool(x5)
        print(x5.shape)
        score = self.bn1(self.relu(self.deconv1(x5)))
        score = self.bn2(self.relu(self.deconv2(score)))
        score = self.bn3(self.relu(self.deconv3(score)))
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.bn6(self.relu(self.deconv6(score)))
        score = torch.sigmoid(self.classifier(score)).squeeze(3)
        return score

# ================================================================================================================================
# 
class myNet2(BasicModule):
    def __init__(self,nclass):
        super(myNet2,self).__init__()
        self.VGG = VGGNet(requires_grad=True, model='vgg16')
        self.n_class = nclass
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
        # self.deconv5 = nn.ConvTranspose2d(32, 16, kernel_size=[3,3], stride=[2,2], padding=1, dilation=1, output_padding=0)
        # self.bn5 = nn.BatchNorm2d(16)
        self.deconv5 = nn.ConvTranspose2d(32, 16, kernel_size=[4,4], stride=[3,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn5 = nn.BatchNorm2d(16)
        self.classifier = nn.Conv2d(16, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        x =  self.VGG(x)
        x5 = x['x5']
        score = self.bn1(self.relu(self.deconv1(x5)))
        # print(score)
        score = self.bn2(self.relu(self.deconv2(score)))
        # print(score)
        score = self.bn3(self.relu(self.deconv3(score)))
        # print(score)
        score = self.bn4(self.relu(self.deconv4(score)))
        # print(score)
        score = self.bn5(self.relu(self.deconv5(score)))
        score = torch.sigmoid(self.classifier(score)).squeeze(1)
        return self.softmax(score)

# ================================================================================================================================

class myNet3(BasicModule):
    def __init__(self, nclass):
            super(myNet3, self).__init__()
            self.VGG = VGGNet(requires_grad=True, model='vgg13')
            self.n_class = nclass
            self.relu = nn.ReLU(inplace=True)
            self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=[3, 3], stride=2, padding=1, dilation=1,
                                              output_padding=[1, 1])
            self.bn1 = nn.BatchNorm2d(512)
            self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=[3, 3], stride=2, padding=1, dilation=1,
                                              output_padding=[1, 1])
            self.bn2 = nn.BatchNorm2d(256)
            self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=[3, 3], stride=2, padding=1, dilation=1,
                                              output_padding=[1, 1])
            self.bn3 = nn.BatchNorm2d(128)
            self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=[5, 5], stride=2, padding=1, dilation=1,
                                              output_padding=[1, 1])
            self.bn4 = nn.BatchNorm2d(64)
            self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=[5, 5], stride=2, padding=1, dilation=1,
                                              output_padding=[1, 1])
            self.bn5 = nn.BatchNorm2d(32)
            self.deconv6 = nn.ConvTranspose2d(32, 32, kernel_size=[7, 7], stride=1, padding=1, dilation=1,
                                              output_padding=0)
            self.bn6 = nn.BatchNorm2d(32)
            self.classifier = nn.Conv2d(32, 1, kernel_size=1)
            self.fc = nn.Linear(138,10)
    def forward(self, x):
            x = self.VGG(x)
            x5 = x['x5']
            print("x5:",x5.shape)
            x4 = x['x4']
            print("x4:",x4.shape)
            x3 = x['x3']
            print("x3",x3.shape)
            x2 = x['x2']
            print("x2", x2.shape)
            score = self.bn1(self.relu(self.deconv1(x5)))
            print("score1:",score.shape)
            score = self.bn2(self.relu(self.deconv2(score)))
            print("score2:", score.shape)
            score = self.bn3(self.relu(self.deconv3(score)))
            print("score3:", score.shape)
            score = self.bn4(self.relu(self.deconv4(score)))
            print("score4:", score.shape)
            score = self.bn5(self.relu(self.deconv5(score)))
            print("score5:", score.shape)
            score = self.bn6(self.relu(self.deconv6(score)))
            print("score6:", score.shape)
           # print(score.shape)
            score = torch.sigmoid(self.fc(self.relu(self.classifier(score)).squeeze(1)))

            # score = score.view(score.shape[0],-1)
            return score.permute(0,2,1)
class myNet4(BasicModule):
    def __init__(self, nclass):
            super(myNet4, self).__init__()
            self.VGG = VGGNet(requires_grad=True, model='vgg13')
            self.n_class = nclass
            self.relu = nn.ReLU(inplace=True)
            self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=[3, 3], stride=2, padding=1, dilation=1,
                                              output_padding=[1, 1])
            self.bn1 = nn.BatchNorm2d(512)
            self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=[5, 3], stride=2, padding=1, dilation=1,
                                              output_padding=[0, 1])
            self.bn2 = nn.BatchNorm2d(256)
            self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=[3, 3], stride=2, padding=1, dilation=1,
                                              output_padding=[1, 1])
            self.bn3 = nn.BatchNorm2d(128)
            self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=[5, 3], stride=2, padding=1, dilation=1,
                                              output_padding=[0, 1])
            self.bn4 = nn.BatchNorm2d(64)
            self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=[3, 3], stride=2, padding=1, dilation=1,
                                              output_padding=[1, 1])
            self.bn5 = nn.BatchNorm2d(32)

            self.classifier = nn.Conv2d(32, 10, kernel_size=1)
            self.fc = nn.Linear(1280,10)
    def forward(self, x):
            x = self.VGG(x)
            x5 = x['x5']
            #print("x5:",x5.shape)
            x4 = x['x4']
            #print("x4:",x4.shape)
            x3 = x['x3']
            #print("x3",x3.shape)
            x2 = x['x2']
            #print("x2", x2.shape)
            x1 = x['x1']
            #print("x1",x1.shape)
            # score = self.bn1(self.relu(self.deconv1(x5)))
            # #print("score1:",score.shape)
            # score = score+x4
            # score = self.bn2(self.relu(self.deconv2(score)))
            # #print("score2:", score.shape)
            score = x3
            score = self.bn3(self.relu(self.deconv3(score)))
            #print("score3:", score.shape)
            score = score+x2
            score = self.bn4(self.relu(self.deconv4(score)))
            #print("score4:", score.shape)
            # score = score+x1
            score = self.bn5(self.relu(self.deconv5(score)))
            #print("score5:", score.shape)
            score = self.relu(self.classifier(score))
           # print(score.shape)
            score = torch.sigmoid(self.fc(score.view(score.shape[0],score.shape[2],-1)))

            # score = score.view(score.shape[0],-1)
            return score.permute(0,2,1)
class myNet5(BasicModule):
    def __init__(self, nclass):
            super(myNet5, self).__init__()
            self.VGG = VGGNet(requires_grad=True, model='vgg13')
            self.n_class = nclass
            self.relu = nn.ReLU(inplace=True)
            self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=[3, 3], stride=2, padding=1, dilation=1,
                                              output_padding=[1, 1])
            self.bn1 = nn.BatchNorm2d(512)
            self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=[5, 3], stride=2, padding=1, dilation=1,
                                              output_padding=[0, 1])
            self.bn2 = nn.BatchNorm2d(256)
            self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=[3, 3], stride=2, padding=1, dilation=1,
                                              output_padding=[1, 1])
            self.bn3 = nn.BatchNorm2d(128)
            self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=[5, 3], stride=2, padding=1, dilation=1,
                                              output_padding=[0, 1])
            self.bn4 = nn.BatchNorm2d(64)
            self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=[3, 3], stride=2, padding=1, dilation=1,
                                              output_padding=[1, 1])
            self.bn5 = nn.BatchNorm2d(32)

            self.classifier = nn.Conv2d(32, 10, kernel_size=1)
            self.fc = nn.Linear(1280,10)
            self.adavpool = torch.nn.AdaptiveAvgPool2d((None,1))
    def forward(self, x):
            x = self.VGG(x)
            x5 = x['x5']
            #print("x5:",x5.shape)
            x4 = x['x4']
            #print("x4:",x4.shape)
            x3 = x['x3']
            #print("x3",x3.shape)
            x2 = x['x2']
            #print("x2", x2.shape)
            x1 = x['x1']
            #print("x1",x1.shape)
            score = self.bn1(self.relu(self.deconv1(x5)))
            #print("score1:",score.shape)
            score = score+x4
            score = self.bn2(self.relu(self.deconv2(score)))
            #print("score2:", score.shape)
            score = score+x3
            score = self.bn3(self.relu(self.deconv3(score)))
            #print("score3:", score.shape)
            score = score+x2
            score = self.bn4(self.relu(self.deconv4(score)))
            #print("score4:", score.shape)
            score = score+x1
            score = self.bn5(self.relu(self.deconv5(score)))
            #print("score5:", score.shape)
            score = torch.sigmoid(self.classifier(score))
           # print(score.shape)
            score = self.adavpool(score).squeeze(3)
           # print(score.shape)

            # score = score.view(score.shape[0],-1)
            return score
class myNet6(BasicModule):
    def __init__(self):
        super(myNet6,self).__init__()
        self.VGG = VGGNet(requires_grad=True, model='vgg11')

        self.fc_block = nn.Sequential(
            nn.Linear(512*4,512),
            nn.ReLU(),
            torch.nn.Dropout(0.5),
            nn.Linear(512,128),
            nn.ReLU(),
            torch.nn.Dropout(0.5),
            nn.Linear(128,10)
        )

    def forward(self, x):
        x =  self.VGG(x)['x5']
        x = x.view(x.size()[0],-1)
        x =  self.fc_block(x)
        x = torch.sigmoid(x)
        return x

class myNet7(BasicModule):
    def __init__(self,nclass):
        super(myNet7,self).__init__()
        self.VGG = VGGNet(requires_grad=True, model='vgg16')
        self.n_class = nclass
        self.relu = nn.ReLU(inplace=True)
        self.bn0 = nn.BatchNorm2d(1)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=[3,3], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=[3,3], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=[3,4], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=[3,4], stride=[1,2], padding=1, dilation=1, output_padding=[0,1])
        self.bn4 = nn.BatchNorm2d(32)
        # self.deconv5 = nn.ConvTranspose2d(32, 16, kernel_size=[3,3], stride=[2,2], padding=1, dilation=1, output_padding=0)
        # self.bn5 = nn.BatchNorm2d(16)
        self.deconv5 = nn.ConvTranspose2d(32, 16, kernel_size=[4,3], stride=[1,2], padding=1, dilation=1, output_padding=0)
        self.bn5 = nn.BatchNorm2d(16)
        self.classifier = nn.Conv2d(16, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        x =  self.VGG(x)
        x5 = x['x5']
        print(x5.shape)
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
class myNet9(BasicModule):
    def __init__(self):
        super(myNet9,self).__init__()
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
            nn.GRU(512,256)
        )
        self.fc_block = nn.Sequential(
            nn.Linear(256*12,3)
        )
        self.pool = nn.AdaptiveAvgPool2d((None,1))
    def forward(self, x1):
        x =  self.VGG(x1)
        print(x.shape)
        x = self.pool(x.permute(0,3,1,2))
        print(x.shape)
        x = x.view(x.shape[0],x.shape[1],-1)
        print(x.shape)
        x, _ = self.rnn_block(x)
        x = x.view(x.size()[0],-1)
        x =  self.fc_block(x)
        x = torch.sigmoid(x)
        return x

if __name__ =='__main__':
    s = myNet6( ).cuda()
    print(torch.randn([1,1,128,43]))
    print(s(torch.randn([1,1,128,43]).cuda()).shape)