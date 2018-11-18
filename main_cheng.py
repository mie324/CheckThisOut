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
from process_data_cheng import *

spacy_en = spacy.load('en')


def convert_csv_to_tsv(file,new_name):
    # given the "abstract file with 400 abstracts" in the form of csv, it converts the file into tsv, with 400 entries
    file=pd.read_csv(file)
    file=file.values.tolist()
    version_400=[]
    for i in range(len(file)):
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
    if len(sentence)>=target:
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
        vector_array.append(ans/3.1415926535897)
    average=0
    count=0
    for k in range(len(vector_array)):
        if numpy.isnan(vector_array[k])==False:
            average+=vector_array[k]
            count+=1
    average=average/count
    return vector_array,average




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


    embeds = nn.Embedding.from_pretrained(Vocab.vectors)

    # this is the training loop
    net=Autoencoder()
    optimizer=torch.optim.RMSprop(net.parameters(),lr=learning_rate)
    loss_func=torch.nn.MSELoss()
    file=read_tsv('./data/abstract_tsv.tsv')
    for i in range(epoch):
        batch_count=0
        validation_amount=400-(int(400/batch_size)-1)*batch_size
        for j in range(int(400/batch_size)-1):
            batch_loss=0
            for b in range(batch_size):
                tsentence = sentence_preprocess(file[batch_count*batch_size+b], Vocab, embeds)
                prediction,middle=net.forward(tsentence)
                loss=loss_func(prediction,tsentence)
                batch_loss+=loss

                angle,ave=find_angle(tsentence,prediction)
                print(angle)
                print(ave)

            batch_loss=batch_loss/batch_size
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            batch_count+=1
            #print('we are at:'+str(j)+'th batch,'+str(i)+'th epoch')
            print(j)

        # this is for validation at the end of each epoch
        val_loss = 0
        for k in range((int(400/batch_size)-1)*batch_size,400):
            tsentence=sentence_preprocess(file[k],Vocab,embeds)
            prediction,middle = net.forward(tsentence)
            loss = loss_func(prediction, tsentence)
            val_loss+=loss
        val_loss=val_loss/validation_amount
        print('epoch: '+str(i)+', training loss:'+str(batch_loss)+', validation loss:'+str(val_loss))
        


def run_fullyconnect_complete_version():
    TEXT = data.Field(sequential=True, include_lengths=True, tokenize='spacy')
    LABEL = data.Field(sequential=False, use_vocab=False)
    abstract_data = data.TabularDataset(path='./data/full_abstract_tsv.tsv', skip_header=True, format='tsv',
                                        fields=[('text', TEXT), ('label', LABEL)])
    TEXT.build_vocab(abstract_data)
    Vocab = TEXT.vocab
    Vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))

    embeds = nn.Embedding.from_pretrained(Vocab.vectors)



    # this is for testing how to preprocess the abstracts

    # this is for preprocess the abstracts
    file = read_tsv('./data/abstract_tsv.tsv')
    abstract_list=[]
    for i in range(11):
        tsentence = sentence_preprocess(file[1], Vocab, embeds)
        abstract_list.append(tsentence)

    # this is for preprocess the hours
    hours=[100,20,301,21,333,123,456,215,242,168]
    hours=Variable(torch.tensor(hours).float())

    # this is the model
    combined_net=full_mlp()
    ans=combined_net.forward(abstract_list,hours)
    print(ans)

def adjust_length_cnn(sentence):
    # this function takes in a sentence, and adjust its length to be around "average length", that is if short-> padding, if long->cut
    # the output of this function will be a tensor
    #max=420
    #min=13
    #average=180
    target=150
    if len(sentence)<target:
        for i in range(target-len(sentence)):
            sentence.append(1)
        #sentence=torch.tensor(sentence).long()
        return sentence
    short=[]
    if len(sentence)>=target:
        for j in range(target):
            short.append(sentence[j])
        #short=torch.tensor(short).long()
        return short

def sentence_preprocess_cnn(sentence,Vocab,embeds):
    # this function takes a sentence in the form of a string
    sentence=tokenizer(sentence)
    sentence=adjust_length_cnn(sentence)
    for i in range(len(sentence)):
        sentence[i]=Vocab.stoi[sentence[i]]
    sentence=embeds(torch.tensor(sentence))
    reshaped_sentence=torch.transpose(sentence,0,1)
    #sentence=torch.tensor([numpy.asarray(reshaped_sentence).tolist()])
    sentence=reshaped_sentence.unsqueeze(0)
    ans=Variable(sentence.float())
    return ans


def convert_data_cnn(data,Vocab,embeds,dictionary):
    # this function take in a data point, and convert the data point into
    # sentence list, hour list, label in the form that can be used directly by the full_cnn model.
    sentence_list=[]
    for i in range(len(data)):
        tsentence=dictionary[data[i][0]]
        tsentence=sentence_preprocess_cnn(tsentence,Vocab,embeds)
        sentence_list.append(tsentence)
    hour_list=[]
    for i in range(10):
        hour_list.append(float(data[i][1]))
    hour_list=Variable(torch.tensor(hour_list).float())
    label=Variable(torch.tensor(float(data[-1][1])).float())
    return sentence_list,hour_list,label

def check_in_diction(dictionary,data):
    count=0
    for i in range(len(data)):
        if data[i][0] in dictionary.keys():
            count+=1
    if count==11:
        return True
    if count!=11:
        return False

def correctness_cnn(prediction,label,range):
    lower=label-range
    higher=label+range
    if prediction<=higher and prediction>=lower:
        return 1
    else:
        return 0

def run_cnn_complete_version():

    learning_rate=0.001
    batch_size=50
    epoch=2
    tollerancec=10


    print('program start')
    print('start loading nlp tools')
    TEXT = data.Field(sequential=True, include_lengths=True, tokenize='spacy')
    LABEL = data.Field(sequential=False, use_vocab=False)
    abstract_data = data.TabularDataset(path='./data/full_abstract_tsv.tsv', skip_header=True, format='tsv',
                                        fields=[('text', TEXT), ('label', LABEL)])
    TEXT.build_vocab(abstract_data)
    Vocab = TEXT.vocab
    Vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    embeds = nn.Embedding.from_pretrained(Vocab.vectors)
    print('finished loading nlp tools')

    train_data, validation_data, test_data = split_data()
    print('finished loading data')

    abstract_dictionary=convert_csv_to_dict('./data/abstracts_final.csv')
    print('finished loading dictionary')

    net=full_cnn()
    optimizer=torch.optim.RMSprop(net.parameters(),lr=learning_rate)
    loss_func=torch.nn.MSELoss()

    for epoch_num in range(epoch):
        count=0
        for batch_num in range(int(len(train_data)/batch_size)):
            loss_in_batch=0
            count_in_batch=0
            accuracy_in_batch=0
            for k in range(batch_size):
                # to access the correct data, use (batch_num*batch_size+k)
                if check_in_diction(abstract_dictionary,train_data[batch_num*batch_size+k]):
                    abstract_list,hour_list,label=convert_data_cnn(train_data[batch_num*batch_size+k],Vocab,embeds,abstract_dictionary)
                    prediction=net.forward(abstract_list,hour_list)
                    loss=loss_func(prediction,label)
                    #print(prediction,label)
                    loss_in_batch+=loss
                    count_in_batch+=1
                    accuracy_in_batch+=correctness_cnn(prediction,label,tollerancec)

                #print('current batch:'+str(batch_num)+', k:'+str(k))
            if count_in_batch!=0:
                loss_in_batch=loss_in_batch/count_in_batch
                accuracy_in_batch=accuracy_in_batch/count_in_batch

                optimizer.zero_grad()
                loss_in_batch.backward()
                optimizer.step()

                #count+=1
                #if count%20==0:
                print(count_in_batch)
                print('epoch:'+str(epoch_num)+', batch:'+str(batch_num)+', loss:'+str(loss_in_batch)+', accuraccy:'+str(accuracy_in_batch))





if __name__=='__main__':
    #convert_csv_to_tsv('./data/abstracts_final_for_vocab.csv','./data/full_abstract_tsv.tsv')
    #file=read_tsv('./data/abstract_tsv.tsv')
    #main()
    #run_fullyconnect_complete_version()
    #run_cnn_complete_version()
    #ans=convert_csv_to_dict('./data/abstracts_final.csv')
    run_cnn_complete_version()











