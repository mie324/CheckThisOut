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
    #file=np.load("./data/playedhours_finalv2.npy").tolist()
    file = np.load("./data/balanced_data2.npy").tolist()
    #print(len(file))
    end_train=int(len(file)*8/10)
    end_val=int(len(file)*9/10)
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

def balance_data():
    file = np.load("./data/playedhours_finalv2.npy").tolist()
    category1=2497
    category2=2741
    category3=2176
    category4=2695

    category1_list=[]
    category2_list=[]
    category3_list=[]
    category4_list=[]

    count = 0
    while len(category1_list)<2176:
        if float(file[count][-1][-1])<10:
            category1_list.append(file[count])
        count+=1

    count = 0
    while len(category2_list) < 2176:
        if float(file[count][-1][-1]) < 35 and float(file[count][-1][-1])>=10:
            category2_list.append(file[count])
        count += 1

    count = 0
    while len(category3_list) < 2176:
        if float(file[count][-1][-1]) < 85 and float(file[count][-1][-1]) >= 35:
            category3_list.append(file[count])
        count += 1

    count = 0
    while len(category4_list) < 2176:
        if float(file[count][-1][-1]) >= 85:
            category4_list.append(file[count])
        count += 1

    final_list=[]
    for i in range(2176):
        final_list.append(category1_list[i])
        final_list.append(category2_list[i])
        final_list.append(category3_list[i])
        final_list.append(category4_list[i])

    final_list=np.asarray(final_list)
    np.save('./data/balanced_data2.npy',final_list)












