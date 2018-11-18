import torch
import numpy

import torchtext
from torchtext import data
import spacy

import argparse

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# this block of code is for the mlp version of the model
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



# this block of code is for the cnn version of model
class CNN(torch.nn.Module):
    def __init__(self,kernel_size):
        super(CNN,self).__init__()
        self.kernel_num=10
        #self.conv1=nn.Sequential(nn.Conv1d(100,self.kernel_num,kernel_size),nn.Sigmoid())
        self.conv1=nn.Conv1d(100,self.kernel_num,kernel_size)
        self.fc1=nn.Linear((150-(kernel_size-1))*self.kernel_num,50)

    def forward(self,input):
        x=self.conv1(input)
        x=x.view(-1,len(x)*len(x[0]))
        x=x.view(-1,len(x)*len(x[0]))
        #print(len(x[0]))
        x=self.fc1(x[0])
        return x

class Decision_maker_for_cnn(torch.nn.Module):
    def __init__(self):
        super(Decision_maker_for_cnn,self).__init__()
        self.fc1=nn.Linear(50*11+10,100)
        self.fc2=nn.Sequential(nn.Linear(100,1),nn.ReLU())

    def forward(self,input):
        x=self.fc1(input)
        x=self.fc2(x)
        return x


class full_cnn(torch.nn.Module):
    def __init__(self):
        super(full_cnn,self).__init__()
        self.kernel_size=100
        self.cnn_list = []
        for i in range(11):
            temp_net = CNN(self.kernel_size)
            self.cnn_list.append(temp_net)
        self.decision_net=Decision_maker_for_cnn()

    def forward(self,input1,input2):
        #input1 to this function should be a "list", but in the list, each element is a 1*100*150 variable
        #input2 should be a Variable which has a size of 10*1, which contain the number of hour that the player has been playing
        # in the top 10 games.
        into_decision = self.cnn_list[0].forward(input1[0])
        for i in range(1, 11):
            temp = self.cnn_list[i].forward(input1[i])
            into_decision = torch.cat((into_decision, temp), 0)

        into_decision=torch.cat((into_decision,input2),0)
        ans=self.decision_net.forward(into_decision)
        return ans




class RNN(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.GRU = nn.GRU(embedding_dim,hidden_size=hidden_dim)

    def forward(self, x, lengths=None):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        # x = nn.utils.rnn.pack_padded_sequence(x,lengths)
        _, x = self.GRU(x)
        return x


class full_rnn(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim):
        super().__init__()
        self.RNNs = []
        for i in range(11):
            temp = RNN(embedding_dim,vocab,hidden_dim)
            self.RNNs.append(temp.cuda())
        self.decision_net = Decision_maker_for_cnn()
        # self.nameindex = [0,2,4,6,8,10,12,14,16,18,20]
        # self.hoursindex = [1,3,5,7,9,11,13,15,17,19]

    def forward(self, x, lengths=None):

        into_decision = self.RNNs[0].forward(x[0].long())
        for i in range(1, 11):
            temp = self.RNNs[i].forward(x[i].long())
            into_decision = torch.cat((into_decision, temp), 0)
        into_decision = into_decision.squeeze().reshape(-1)
        for j in range(11,21):
            into_decision = torch.cat((into_decision, torch.tensor([float(x[j])]).cuda()), 0)
        ans = self.decision_net.forward(into_decision.squeeze())
        return ans

