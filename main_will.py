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
seed = 1

# abstractset = pd.read_csv("./data/abstracts_final.csv")
playedhours = np.load("./data/playedhours_final.npy")
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




def evaluate(model, valiter, loss_func):
    total_corr = 0
    total_loss = 0
    total_entries = 0

    for j, entry in enumerate(valiter):

        (features, length) = entry.text
        label = entry.Label
        val_outputs = model(features, length) # hypothetically this is batchsize x 26
        corr = (val_outputs > 0.5).squeeze().long() == label.squeeze()
        loss = loss_func(val_outputs.squeeze().float(), label.squeeze().float())

        total_entries += length.shape[0]
        total_loss += loss.item()
        total_corr += int(corr.sum())

    return float(total_corr)/total_entries, float(total_loss)/(j+1)

def load_model(lr, vocab, embsize, hiddennum):

    model = full_rnn(embsize, vocab, hiddennum)
    loss_fnc = torch.nn.MSELoss()
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

def correctness(prediction,label,tolerance):
    lower=label-tolerance
    higher=label+tolerance
    if prediction<=higher and prediction>=lower:
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
    modeltype = args.model
    rnn_hidden_dim = args.rnn_hidden_dim
    n_kernals = args.num_filt
    get_abs = convert_csv_to_dict("./data/abstracts_final.csv")
    tolerance = 10

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
    currmaxaccu = 0.0  # for saving model


    train_set, val_set, test_set = load_data()
    nameindex = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    hoursindex = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]


    batch_num = int(len(feat_train)/batch_size)

    corr = 0
    epoch = 0
    for epoch in range(maxepoch):

        for i in range(batch_num):
            predict_batch = torch.Tensor([]).cuda()
            labels_batch = torch.Tensor([]).cuda()
            for j in range(batch_size):

                features, labels = train_set.__getitem__(i*batch_size+j)
                optimizer.zero_grad()
                if check_in_diction(get_abs, features):

                    absfeatures = []
                    for l in nameindex:
                        absfeatures.append(sentence_preprocess_rnn(get_abs[features[l]], abstract.vocab).cuda())
                    for k in hoursindex:
                        absfeatures.append(torch.tensor(float(features[k])).cuda())
                    predict = model(absfeatures)
                    predict_batch = torch.cat((predict_batch, predict), 0)
                    labels_batch = torch.cat((labels_batch, torch.tensor([float(labels)]).cuda()), 0)
                    corr += correctness(predict,torch.tensor([float(labels)]).cuda(),tolerance)

            loss = loss_func(predict_batch,labels_batch)
            loss.backward()
            optimizer.step()

            # val_ac, val_los = evaluate(model, val_iter, loss_func)
            # val_accu.append(val_ac)
            # val_loss.append(val_los)

            # if val_accu[-1] > currmaxaccu:
            #     torch.save(model, 'model_{}.pt'.format(modeltype))
            #     currmaxaccu = val_accu[-1]

            train_loss.append(loss.item())
            train_accu.append(corr / batch_size)
            print('[Epoch #%d,Step #%d] | Training Loss: %.9f | Correct Predictions: %d | Training Accuracy: %f' % (epoch + 1, i, train_loss[-1], corr, train_accu[-1]))
            corr = 0
            # total_train_num = 0
            # time_elap.append(time() - start_time)







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model', type=str, default='baseline',
                        help="Model type: baseline,rnn,cnn (Default: baseline)")
    parser.add_argument('--emb_dim', type=int, default=100)
    parser.add_argument('--rnn_hidden_dim', type=int, default=50)
    parser.add_argument('--num_filt', type=int, default=50)

    args = parser.parse_args()

    main(args)