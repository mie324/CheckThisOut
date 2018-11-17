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
        self.layer1=nn.Sequential(nn.Linear(15000,800),nn.Sigmoid())
        #self.layer2=nn.Sequential(nn.Linear(9000,800),nn.Tanh())
        self.layer3 = nn.Sequential(nn.Linear(800, 50),nn.Sigmoid())

        self.layer4 = nn.Sequential(nn.Linear(50, 800),nn.Sigmoid())
        #self.layer5 = nn.Sequential(nn.Linear(800, 2000), nn.Tanh())
        #self.layer6 = nn.Sequential(nn.Linear(800, 15000),nn.Sigmoid())
        self.layer6 = nn.Linear(800, 15000)

    def forward(self,input):
        x=self.layer1(input)
        #x=self.layer2(x)
        output = self.layer3(x)
        x = self.layer4(output)
        #x = self.layer5(x)
        x = self.layer6(x)
        return x,output

class Decision_maker(torch.nn.Module):
    def __init__(self):
        super(Decision_maker,self).__init__()
        self.layer1=nn.Sequential(nn.Linear(560,100),nn.Tanh())
        self.layer2=nn.Linear(100,1)

    def forward(self,input):
        x=self.layer1(input)
        x=self.layer2(x)
        return x

class full_mlp(torch.nn.Module):
    def __init__(self):
        super(full_mlp,self).__init__()
        self.linear_net_list = []
        for i in range(11):
            temp_net = Autoencoder()
            self.linear_net_list.append(temp_net)
        self.decision_net=Decision_maker()

    def forward(self,input1,input2):
        #input1 to this function should be a "list", but in the list, each element is a 15000*1 variable
        #input2 should be a Variable which has a size of 10*1, which contain the number of hour that the player has been playing
        # in the top 10 games.
        t, abstract_for_decision = self.linear_net_list[0].forward(input1[0])
        for i in range(1, 11):
            final, middle = self.linear_net_list[i].forward(input1[i])
            abstract_for_decision = torch.cat((abstract_for_decision, middle), 0)

        input_to_next=torch.cat((abstract_for_decision,input2),0)
        ans=self.decision_net.forward(input_to_next)
        return ans
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(nn.Conv1d(100,10,5),nn.Sigmoid())
        self.fc1=nn.Linear(1460,1)

    def forward(self,input):
        x=self.conv1(input)
        x=x.view(-1,len(x)*len(x[0]))
        x=x.view(-1,len(x)*len(x[0]))
        #print(len(x[0]))
        x=self.fc1(x[0])
        return x

