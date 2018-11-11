'''
    Write a model for gesture classification.
'''
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

import torchtext
from torchtext import data
import spacy
from time import time
import argparse
import os


import torch
import torch.optim as optim
import torchtext
from torchtext import data
import spacy
import numpy as np
import pandas as pd

dataset = pd.read_csv("./data/abstract_tsv.tsv",sep='\t')



# Define Tokenizer
nlp = spacy.load('en')
def tokenizer(text): # create a tokenizer function
    tokens = []
    for i in nlp(text):
        tokens.append([i.text])
    # [tok.text for tok in nlp(text)]
    return tokens

def main():
    # Build Vocab
    abstract = data.Field(sequential=True, tokenize="spacy", include_lengths=True)
    name = data.Field(sequential=False, use_vocab=False)
    abstract_data = data.TabularDataset(path='./data/abstract_tsv.tsv', skip_header=False, format='tsv',
                                     fields=[('name', name), ('abstract', abstract)])
    abstract.build_vocab(abstract_data)
    vocab = abstract.vocab
    abstract_iter = data.BucketIterator(abstract_data, batch_size=10 ,sort_key=lambda x: len(x.abstract),sort_within_batch=True, repeat=False)
    for i, abs in enumerate(abstract_iter):
        feature,length = abs.abstract


    # while True:
    #
    #     sentence = input('Enter a sentence:')
    #     # sentence = "What once seemed creepy now just seems campy"
    #     print(sentence)
    #
    #     tokens = tokenizer(sentence)    # list of tokenized words
    #     # these were included in the tokenizer function
    #     # tokens_unsqueezed = []
    #     # for word in tokens:
    #     #     tokens_unsqueezed.append([word])
    #
    # length = 0
    # tokens_num = []
    # for tok in tokens:
    #     tokens_num.append(vocab.stoi[tok[0]])
    #     length += 1
    # tokens_num = torch.LongTensor(tokens_num).unsqueeze(1)
    # length = torch.tensor([length])
    #
    #     baseline_pred = baseline_model(tokens_num).squeeze().detach().numpy()
    #     if baseline_pred > 0.5:
    #         baseline_res = "Subjective"
    #     else:
    #         baseline_res = "Objective"
    #     print("Model Baseline: {} ({:.3f})".format(baseline_res, baseline_pred))
    #     rnn_pred = rnn_model(tokens_num, length).squeeze().detach().numpy()
    #     if rnn_pred > 0.5:
    #         rnn_res = "Subjective"
    #     else:
    #         rnn_res = "Objective"
    #     print("Model RNN: {} ({:.3f})".format(rnn_res, rnn_pred))
    #     cnn_pred = cnn_model(tokens_num).squeeze().detach().numpy()
    #     if cnn_pred > 0.5:
    #         cnn_res = "Subjective"
    #     else:
    #         cnn_res = "Objective"
    #     print("Model CNN: {} ({:.3f})".format(cnn_res, cnn_pred))

if __name__ == '__main__':
    main()