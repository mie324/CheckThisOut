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
    def __init__(self):
        super(CNN,self).__init__()
        self.input_length=150

        self.kernel_size1=2
        self.kernel_size2=5
        self.kernel_size3=10
        self.kernel_size4=20

        self.kernel_num1=5
        self.kernel_num2=5
        self.kernel_num3=5
        self.kernel_num4=5

        self.pool_size1 = self.input_length - (self.kernel_size1 - 1)
        self.pool_size2 = self.input_length - (self.kernel_size2 - 1)
        self.pool_size3 = self.input_length - (self.kernel_size3 - 1)
        self.pool_size4 = self.input_length - (self.kernel_size4 - 1)

        self.conv1=nn.Sequential(nn.Conv1d(100,self.kernel_num1,self.kernel_size1),nn.ReLU())
        self.pool1=nn.MaxPool1d(self.pool_size1)

        self.conv2 = nn.Sequential(nn.Conv1d(100, self.kernel_num2, self.kernel_size2),nn.ReLU())
        self.pool2 = nn.MaxPool1d(self.pool_size2)

        self.conv3 = nn.Sequential(nn.Conv1d(100, self.kernel_num3, self.kernel_size3),nn.ReLU())
        self.pool3 = nn.MaxPool1d(self.pool_size3)

        self.conv4 = nn.Sequential(nn.Conv1d(100, self.kernel_num4, self.kernel_size4),nn.ReLU())
        self.pool4 = nn.MaxPool1d(self.pool_size4)




    def forward(self,input):
        x1=self.conv1(input)
        x1=self.pool1(x1)
        x1=x1.view(-1,self.kernel_num1)

        x2=self.conv2(input)
        x2=self.pool2(x2)
        x2=x2.view(-1,self.kernel_num2)

        x3=self.conv3(input)
        x3=self.pool3(x3)
        x3=x3.view(-1,self.kernel_num3)

        x4=self.conv4(input)
        x4=self.pool4(x4)
        x4=x4.view(-1,self.kernel_num4)

        ans=torch.cat((x1,x2,x3,x4),1)
        return ans


class Decision_maker_for_cnn(torch.nn.Module):
    def __init__(self):
        super(Decision_maker_for_cnn,self).__init__()
        #self.fc1=nn.Linear(50*11+10*4,100)
        self.fc1 =nn.Sequential(nn.Linear(4*10+11*20, 4),nn.Sigmoid())
        self.fc2=nn.Sequential(nn.Linear(50,4),nn.Sigmoid())

    def forward(self,input):
        x=self.fc1(input)
        #x=self.fc2(x)
        return x


class full_cnn(torch.nn.Module):
    def __init__(self):
        super(full_cnn,self).__init__()
        self.cnn_list = []
        for i in range(11):
            temp_net = CNN()
            self.cnn_list.append(temp_net)
        self.decision_net=Decision_maker_for_cnn()

    def forward(self,input1,input2):
        #input1 to this function should be a "list", but in the list, each element is a 1*100*150 variable
        #input2 should be a Variable which has a size of 10*1, which contain the number of hour that the player has been playing
        # in the top 10 games.
        into_decision = self.cnn_list[0].forward(input1[0])
        for i in range(1, 11):
            temp = self.cnn_list[i].forward(input1[i])
            into_decision = torch.cat((into_decision, temp), 1)

        into_decision=torch.cat((into_decision,input2),1)
        ans=self.decision_net.forward(into_decision)
        return ans[0]




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

