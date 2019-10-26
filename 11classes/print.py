import torch
import models
from config import *
from torchsummary import summary

device = torch.device('cuda')


model = models.myNet2(nclass=3)
# must add input size to forward
opt.notes=''
model.load_latest(opt.notes)
model.to(device)
summary(model, (1,128,201))