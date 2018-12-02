import torch
import torch.optim as optim

import torchtext
# from torchtext import data
import spacy
from time import time
import argparse
import os
import pandas as pd
import numpy as np
from model import *
from torch.utils.data import DataLoader
from dataset import HoursDataset
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
seed = 12315

# abstractset = pd.read_csv("./data/abstracts_final.csv")
playedhours = np.load("./data/tophours11.npy")
feat_train, feat_temp, discard1, discard2 = train_test_split(playedhours, np.zeros((len(playedhours),1)), test_size=0.2, random_state=seed)
feat_valid, feat_test, discard1, discard2 = train_test_split(feat_temp, np.zeros((len(feat_temp),1)), test_size=0.5, random_state=seed)


# Define Tokenizer
nlp = spacy.load('en')
def tokenizer(text): # create a tokenizer function
    tokens = []
    for i in nlp(text):
        tokens.append([i.text])
    # [tok.text for tok in nlp(text)]
    return tokens

def numeriacalize_sentence(sentence,Vocab):
    res = []
    for i in range(len(sentence)):
        res.append([Vocab.stoi[sentence[i][0]]])
    return res

def adjust_length(sentence):
    # this function takes in a sentence, and adjust its length to be around "average length", that is if short-> padding, if long->cut
    # the output of this function will be a tensor
    #max=420
    #min=13
    #average=180
    target=150
    if len(sentence)<target:
        for i in range(target-len(sentence)):
            sentence.append([1])
        sentence=torch.tensor(sentence).long()
        return sentence
    short=[]
    if len(sentence)>=target:
        for j in range(target):
            short.append(sentence[j])
        short=torch.tensor(short).long()
        return short

def sentence_preprocess_rnn(sentence,Vocab):
    tsentence = tokenizer(sentence)
    tsentence = numeriacalize_sentence(tsentence, Vocab)
    tsentence = adjust_length(tsentence)
    tsentence = Variable((tsentence).float())
    return tsentence.squeeze()




def evaluate(model, valset, vocab, dict):
    valcorr = 0
    nameindex = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # predict_all = torch.Tensor([]).cuda()
    # labels_all = torch.Tensor([]).float().cuda()
    numofval = 100
    # numofval = len(valset)
    for j in range(numofval):

        features, labels = valset.__getitem__(j)
        absfeatures = []
        for l in nameindex:
            absfeatures.append(sentence_preprocess_rnn(dict[features[l]], vocab).cuda())
        for k in range(50):
            if k not in nameindex:
                absfeatures.append(torch.tensor(float(features[k])).cuda())
        predict = model(absfeatures)
        # predict_all = torch.cat((predict_all, predict), 0)
        # labels_all = torch.cat((labels_all, torch.tensor(labels.astype(float)).float().cuda()), 0)
        valcorr += correctness(predict, torch.tensor([labels.astype(float)]).cuda())

    return float(valcorr)/numofval

def load_model(lr, vocab, embsize, hiddennum):

    model = full_rnn(embsize, vocab, hiddennum)
    loss_fnc = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, loss_fnc, optimizer



def load_data():

    fakelabels = np.zeros((len(feat_valid),1))
    train_set = HoursDataset(feat_train,np.zeros((len(feat_train),1)))
    val_set = HoursDataset(feat_valid, fakelabels)
    test_set = HoursDataset(feat_test, fakelabels)

    return train_set,val_set,test_set

def convert_csv_to_dict(file):
    # by using the ./data/abstracts_final.csv as input, this function returns a dictionary for the games.
    ans=pd.read_csv(file)
    ans=ans.values.tolist()
    ans=dict(ans)
    return ans

def check_in_diction(dictionary,data):
    count=0
    for i in range(len(data)):
        if data[i] in dictionary.keys():
            count+=1
    if count==11:
        return True
    if count!=11:
        return False

def oldcorrectness(prediction,label,tolerance):
    lower=label-tolerance
    higher=label+tolerance
    if prediction<=higher and prediction>=lower:
        return 1
    else:
        return 0
def correctness(prediction,label):

    if prediction.argmax()==label.argmax():
        return 1
    else:
        return 0

def main(args):
    ######
    # 3.2 Processing of the data
    batch_size = args.batch_size
    maxepoch = args.epochs
    lear_r = args.lr
    stepsize = 10
    embdim = args.emb_dim
    # modeltype = args.model
    rnn_hidden_dim = args.rnn_hidden_dim
    # n_kernals = args.num_filt
    get_abs = convert_csv_to_dict("./data/abstracts_final.csv")
    # tolerance = 10

    # Build Vocab
    abstract = data.Field(sequential=True, tokenize="spacy", include_lengths=True)
    fakelabel = data.Field(sequential=False, use_vocab=False)
    abstract_data = data.TabularDataset(path='./data/abstract_tsv.tsv', skip_header=True, format='tsv',
                                        fields=[('abstract', abstract), ('label', fakelabel)])

    abstract_iter = data.BucketIterator(abstract_data, batch_size=10, sort_key=lambda x: len(x.abstract),
                                        sort_within_batch=True, repeat=False)

    abstract.build_vocab(abstract_data)
    glove = torchtext.vocab.GloVe(name='6B', dim=100)
    abstract.vocab.load_vectors(glove)


    model, loss_func, optimizer = load_model(lear_r, abstract.vocab, embdim, rnn_hidden_dim)

    model.cuda()

    train_accu = []
    train_loss = []
    val_accu = []
    val_loss = []

    time_elap = []
    start_time = time()
    time_now = time()
    currmaxaccu = 0.0  # for saving model


    train_set, val_set, test_set = load_data()
    nameindex = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    batch_num = int(len(feat_train)/batch_size)
    corr = 0
    epoch = 0
    for epoch in range(maxepoch):

        for i in range(batch_num):
            predict_batch = torch.Tensor([]).cuda()
            labels_batch = torch.Tensor([]).float().cuda()
            for j in range(batch_size):

                features, labels = train_set.__getitem__(i*batch_size+j)
                optimizer.zero_grad()
                # if check_in_diction(get_abs, features):

                absfeatures = []
                for l in nameindex:
                    absfeatures.append(sentence_preprocess_rnn(get_abs[features[l]], abstract.vocab).cuda())
                for k in range(50):
                    if k not in nameindex:
                        absfeatures.append(torch.tensor(float(features[k])).cuda())
                predict = model(absfeatures)
                predict_batch = torch.cat((predict_batch, predict), 0)
                labels_batch = torch.cat((labels_batch, torch.tensor(labels.astype(float)).float().cuda()), 0)
                corr += correctness(predict, torch.tensor([labels.astype(float)]).cuda())

            loss = loss_func(predict_batch, labels_batch)
            loss.backward()
            optimizer.step()



            # if val_accu[-1] > currmaxaccu:
            #     torch.save(model, 'model_{}.pt'.format(modeltype))
            #     currmaxaccu = val_accu[-1]

            train_loss.append(loss.item())
            train_accu.append(corr / batch_size)

            prev_time = time_now
            time_now = time()
            time_elap.append(time_now - prev_time)
            print('[Epoch #%d,Step #%d] | Training Loss: %.9f | Correct Predictions: %d | Training Accuracy: %f | Time Elapsed: %f | Step Time: %f' % (epoch + 1, i, train_loss[-1], corr, train_accu[-1],time_now-start_time,time_elap[-1]))
            corr = 0
            # total_train_num = 0

            if i % 30 == 29:
                val_ac = evaluate(model, val_set, abstract.vocab, get_abs)
                val_accu.append(val_ac)
                print('Validation Accuracy: {}'.format(val_ac))
                torch.save(model, 'model_{}.pt'.format(val_ac))
                # currmaxaccu = val_accu[-1]


def run_test():
    conf_mat = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    conf_mat = torch.tensor(conf_mat).cuda()
    train_set, val_set, test_set = load_data()
    model = torch.load("shuffledmodel_0.52.pt")

    testcorr = 0
    nameindex = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    numofval = len(test_set)

    abstract = data.Field(sequential=True, tokenize="spacy", include_lengths=True)
    fakelabel = data.Field(sequential=False, use_vocab=False)
    abstract_data = data.TabularDataset(path='./data/abstract_tsv.tsv', skip_header=True, format='tsv',
                                        fields=[('abstract', abstract), ('label', fakelabel)])

    abstract.build_vocab(abstract_data)
    get_abs = convert_csv_to_dict("./data/abstracts_final.csv")

    for j in range(numofval):

        features, labels = test_set.__getitem__(j)
        absfeatures = []
        for l in nameindex:
            absfeatures.append(sentence_preprocess_rnn(get_abs[features[l]], abstract.vocab).cuda())
        for k in range(50):
            if k not in nameindex:
                absfeatures.append(torch.tensor(float(features[k])).cuda())
        predict = model(absfeatures)
        conf_mat, cor = confusion_matrix(conf_mat, predict, torch.tensor([labels.astype(float)]).cuda())
        testcorr += cor
    print(conf_mat)
    print(testcorr/numofval)

    # Confusion Matrix:
    # [[212, 24, 1, 7],
    #  [33, 240, 4, 4],
    #  [7, 65, 119, 19],
    #  [26, 38, 39, 173]]
    #
    # 0.7359050445103857

    # [[177, 49, 0, 25],
    #  [83, 74, 4, 42],
    #  [19, 30, 1, 21],
    #  [14, 23, 0, 26]]
    #
    #  0.47278911564625853




def confusion_matrix(matrix, prediction, label):
    mat = matrix
    pred = prediction.argmax()
    lab = label.argmax()
    mat[lab][pred] += 1
    if lab == pred:
        yes = 1
    else:
        yes = 0
    return mat, yes




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=25)
    # parser.add_argument('--model', type=str, default='baseline',
    #                     help="Model type: baseline,rnn,cnn (Default: baseline)")
    parser.add_argument('--emb_dim', type=int, default=100)
    parser.add_argument('--rnn_hidden_dim', type=int, default=50)
    # parser.add_argument('--num_filt', type=int, default=50)

    args = parser.parse_args()

    main(args)
    # run_test()