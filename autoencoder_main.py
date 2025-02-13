import torch
import numpy
import pandas as pd

import torchtext
from torchtext import data
import spacy

import argparse

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import timeit

from model import *

spacy_en = spacy.load('en')

def find_angle(target,input):
    # the input of thie function should be 15000*1, and the output will be 150*1, where the values
    # are angle between the ith vector in radian
    vector_size=100
    vector_array=[]
    for i in range(int(len(target)/vector_size)):
        temp_target=[]
        temp_input=[]
        for j in range(vector_size):
            temp_target.append(target[i*vector_size+j].tolist())
            temp_input.append(input[i*vector_size+j].tolist())
        temp_target=numpy.asarray(temp_target)
        norm_target=temp_target/numpy.linalg.norm(temp_target)
        temp_input=numpy.asarray(temp_input)
        norm_input=temp_input/numpy.linalg.norm(temp_input)
        d = numpy.dot(norm_target, norm_input)
        ans = numpy.arccos(d)
        vector_array.append(ans/3.14159265358979323846)
    ave = 0
    count = 0
    for k in range(len(vector_array)):
        if numpy.isnan(vector_array[k])==False:
            ave += vector_array[k]
            count+=1

    return vector_array,ave/count


def convert_csv_to_tsv(file,new_name):
    # given the "abstract file with 400 abstracts" in the form of csv, it converts the file into tsv, with 400 entries
    file=pd.read_csv(file)
    file=file.values.tolist()
    version_400=[]
    for i in range(400):
        version_400.append(file[i])
    file=pd.DataFrame(file)
    file.columns = ['text', 'label']
    file.to_csv(new_name,sep='\t',index=False)

def read_tsv(file):
    # this file reads a tsv file and convert it into list.
    ans=pd.read_csv(file,sep='\t')
    ans=ans.values.tolist()
    return ans

def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def adjust_length(sentence):
    # this function takes in a sentence, and adjust its length to be around "average length", that is if short-> padding, if long->cut
    # the output of this function will be a tensor
    #max=420
    #min=13
    #average=180
    target=150
    if len(sentence)<target:
        for i in range(target-len(sentence)):
            sentence.append(1)
        sentence=torch.tensor(sentence).long()
        return sentence
    short=[]
    if len(sentence)>= target:
        for j in range(target):
            short.append(sentence[j])
        short=torch.tensor(short).long()
        return short

def numeriacalize_sentence(sentence,Vocab):
    for i in range(len(sentence)):
        sentence[i]=Vocab.stoi[sentence[i]]
    return sentence

def sentence_preprocess(sentence,Vocab,embeds):
    # this function takes in a sentence straght after "read_tsv", and the output can be feed into the net directly
    tsentence = tokenizer(sentence[0])
    tsentence = numeriacalize_sentence(tsentence, Vocab)
    tsentence = adjust_length(tsentence)
    tsentence = Variable(
        embeds(tsentence).float())  # note there are considerable amount of zeros word vectors here, might be /x0
    word_0 = tsentence[0]
    for i in range(len(tsentence) - 1):
        word_0 = torch.cat((word_0, tsentence[i + 1]), 0)
    word_0=Variable(word_0)
    return word_0

def main():
    batch_size=20
    learning_rate=0.0001
    epoch=10
    TEXT = data.Field(sequential=True, include_lengths=True, tokenize='spacy')
    LABEL = data.Field(sequential=False, use_vocab=False)
    abstract_data = data.TabularDataset(path='./data/abstract_tsv.tsv', skip_header=True, format='tsv',fields=[('text',TEXT),('label',LABEL)])
    TEXT.build_vocab(abstract_data)
    Vocab=TEXT.vocab
    Vocab.load_vectors(torchtext.vocab.GloVe(name='6B',dim=100))
    #test_sentence=read_tsv('./data/abstract_tsv.tsv')
    # the above part is for text processing

    # preprocess required for each sentence

    embeds = nn.Embedding.from_pretrained(Vocab.vectors)
    #tsentence=sentence_preprocess(test_sentence[0],Vocab,embeds)

    #net=Autoencoder()
    #ans=net.forward(tsentence)

    # this is the training loop
    net=Autoencoder()
    # net.cuda()
    optimizer=torch.optim.RMSprop(net.parameters(),lr=learning_rate)
    loss_func=torch.nn.MSELoss()
    file=read_tsv('./data/abstract_tsv.tsv')

    best_loss = 1000
    for i in range(epoch):
        batch_count=0
        validation_amount=400-(int(400/batch_size)-1)*batch_size
        for j in range(int(400/batch_size)-1):
            print("Epoch#: {}, Batch#: {}".format(i+1,j+1))
            batch_loss=0
            for b in range(batch_size):
                tsentence = sentence_preprocess(file[batch_count*batch_size+b], Vocab, embeds)
                prediction, encoded = net(tsentence)
                loss=loss_func(prediction,tsentence)
                batch_loss+=loss
                angles,ave = find_angle(tsentence,prediction)
                print("Average Angle Distance: {}pi".format(ave))

            batch_loss=batch_loss/batch_size
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            batch_count+=1
            #print('we are at:'+str(j)+'th batch,'+str(i)+'th epoch')

        # this is for validation at the end of each epoch
        val_loss = 0
        for k in range((int(400/batch_size)-1)*batch_size,400):
            tsentence=sentence_preprocess(file[k],Vocab,embeds)
            prediction, encoded = net.forward(tsentence)
            loss = loss_func(prediction, tsentence)
            val_loss+=loss
        val_loss = val_loss/validation_amount
        print('epoch: '+str(i)+', training loss:'+str(batch_loss)+', validation loss:'+str(val_loss))
        if val_loss < best_loss:
            torch.save(net, 'model_{}.pt'.format('best'))
            best_loss = val_loss








if __name__=='__main__':
    #abstracts=convert_csv_to_tsv('./data/abstracts500c.csv','./data/abstract_tsv.tsv')
    #file=read_tsv('./data/abstract_tsv.tsv')

    # test
    main()

