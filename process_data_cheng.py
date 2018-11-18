import torch
import numpy
import numpy as np
import pandas as pd

import torchtext
from torchtext import data
import spacy

import argparse

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import timeit
from sklearn.model_selection import train_test_split

from model import *
import math

def split_data():
    file=np.load("./data/playedhours_final.npy").tolist()
    #print(len(file))
    end_train=12074
    end_val=12074+1509
    end_test=len(file)
    train_data=[]
    val_data=[]
    test_data=[]
    for i in range(end_train):
        train_data.append(file[i])
    for j in range(end_train,end_val):
        val_data.append(file[j])
    for k in range(end_val,end_test):
        test_data.append(file[k])
    return train_data,val_data,test_data

def convert_csv_to_dict(file):
    # by using the ./data/abstracts_final.csv as input, this function returns a dictionary for the games.
    ans=pd.read_csv(file)
    ans=ans.values.tolist()
    ans=dict(ans)
    return ans
