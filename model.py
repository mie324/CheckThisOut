'''
    Write a model for gesture classification.
'''
import torch.nn as nn
import torch.nn.functional as F
import torch


class CNN(nn.Module):
    def __init__(self, activation, hiddensize,dropoutrate):
        super(CNN, self).__init__()

        self.stride = 1
        self.conv1 = nn.Conv1d(6, 10, 15,stride=self.stride,padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(10, 20, 11,stride=self.stride)
        self.convoutputsize = int(((((100 - 12)/self.stride)/2 - 10)/self.stride)/2) * 20 # 20 is the kernal # for last conv layer
        self.fc1 = nn.Linear(self.convoutputsize, hiddensize)
        self.fc2 = nn.Linear(hiddensize, 169)

        self.fc4 = nn.Linear(169, 98)
        # self.fc5 = nn.Linear(228, 100)
        # self.fc6 = nn.Linear(100, 60)
        self.bnconv = nn.BatchNorm1d(20)
        self.bn1 = nn.BatchNorm1d(hiddensize)
        # self.bn2 = nn.BatchNorm1d(172)
        # self.bn3 = nn.BatchNorm1d(98)

        self.fc3 = nn.Linear(98, 26)
        self.AF = activation
        self.dropout = nn.Dropout(dropoutrate)




    def forward(self, x):

        if self.AF == 'tanh':

            x = self.pool(torch.tanh(self.conv1(x)))
            x = self.pool(self.dropout(torch.tanh(self.conv2(x))))
            x = x.view(-1, self.convoutputsize)
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.bn2(self.fc2(x)))
            # x = self.fc3(x)


        elif self.AF == 'relu':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(self.dropout(F.relu(self.bnconv(self.conv2(x)))))
            x = x.view(-1, self.convoutputsize)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            # x = self.fc3(x)


        elif self.AF == 'sigmoid':
            x = self.pool(torch.sigmoid(self.conv1(x)))
            x = self.pool(self.dropout(torch.sigmoid(self.bnconv(self.conv2(x)))))
            x = x.view(-1, self.convoutputsize)
            x = torch.sigmoid(self.fc1(x))
            x = torch.sigmoid(self.fc2(x))
            # x = self.fc3(x)

        x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        # x = self.fc6(x)
        x = self.fc3(x)
        return x