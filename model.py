import torch
import numpy

import torchtext
from torchtext import data
import spacy

import argparse

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Autoencoder(torch.nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.layer1=nn.Sequential(nn.Linear(15000,8000),nn.Tanh())
        self.layer2=nn.Sequential(nn.Linear(8000,1000),nn.Tanh())
        self.layer3 = nn.Sequential(nn.Linear(1000, 100), nn.Tanh())

        self.layer4 = nn.Sequential(nn.Linear(100, 1000), nn.Tanh())
        self.layer5 = nn.Sequential(nn.Linear(1000, 8000), nn.Tanh())
        self.layer6 = nn.Sequential(nn.Linear(8000, 15000), nn.Sigmoid())

    def forward(self,input):
        x=self.layer1(input)
        x=self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x