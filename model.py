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
        self.layer1 = nn.Sequential(nn.Linear(15000, 800), nn.Tanh())
        # self.layer2 = nn.Sequential(nn.Linear(9000, 800), nn.Tanh())
        self.layer3 = nn.Sequential(nn.Linear(800, 50), nn.Tanh())

        self.layer4 = nn.Sequential(nn.Linear(50, 800), nn.Tanh())
        #self.layer5 = nn.Sequential(nn.Linear(1000, 8000), nn.Tanh())
        self.layer6 = nn.Linear(800, 15000)

    def forward(self,input):

        x = self.layer1(input)
        # x = self.layer2(x)
        encoded = self.layer3(x)
        x = self.layer4(encoded)
        #x = self.layer5(x)
        x = self.layer6(x)
        return x, encoded


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

class RNN(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim):
        super(RNN, self).__init__()
        ######
        # 4.2 YOUR CODE HERE
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.GRU = nn.GRU(embedding_dim,hidden_size=hidden_dim)
        self.linear1 = nn.Linear(embedding_dim,1)



    def forward(self, x, lengths=None):

        ######
        # 4.2 YOUR CODE HERE
        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x,lengths)
        _, x = self.GRU(x)
        x = self.linear1(x)
        x = torch.sigmoid(x)
        return x

