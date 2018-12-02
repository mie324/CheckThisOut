import torch
import torch.optim as optim
import torchtext
from torchtext import data
import spacy
import numpy as np
import torch.nn as nn
import pandas as pd
from torch.autograd import Variable

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

def convert_csv_to_dict(file):
    # by using the ./data/abstracts_final.csv as input, this function returns a dictionary for the games.
    ans=pd.read_csv(file)
    ans=ans.values.tolist()
    ans=dict(ans)
    return ans

def sentence_preprocess_rnn(sentence,Vocab):
    tsentence = tokenizer(sentence)
    tsentence = numeriacalize_sentence(tsentence, Vocab)
    tsentence = adjust_length(tsentence)
    tsentence = Variable((tsentence).float())
    return tsentence.squeeze()

def adjust_length_cnn(sentence):
    # this function takes in a sentence, and adjust its length to be around "average length", that is if short-> padding, if long->cut
    # the output of this function will be a tensor
    #max=420
    #min=13
    #average=180
    target=150
    if len(sentence)<target:
        for i in range(target-len(sentence)):
            sentence.append([1])
        #sentence=torch.tensor(sentence).long()
        return sentence
    short=[]
    if len(sentence)>=target:
        for j in range(target):
            short.append(sentence[j])
        #short=torch.tensor(short).long()
        return short


def convert_hour_to_onehot(list):
    ans=[]
    for i in range(len(list)):
        temp=[0,0,0,0]
        if list[i]<10:
            temp[0]=1
        if list[i]>=10 and list[i]<35:
            temp[1]=1
        if list[i]>=35 and list[i]<85:
            temp[2]=1
        if list[i]>=85:
            temp[3]=1
        for j in range(len(temp)):
            ans.append(temp[j])
    return ans

def sentence_preprocess_cnn(sentence,Vocab,embeds):
    # this function takes a sentence in the form of a string
    sentence=tokenizer(sentence)
    sentence=adjust_length_cnn(sentence)
    senttemp = []
    for i in range(len(sentence)):
        senttemp.append(Vocab.stoi[sentence[i][0]])
    sentence=embeds(torch.tensor(senttemp))
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
        #print(tsentence)
        tsentence=sentence_preprocess_cnn(tsentence,Vocab,embeds)
        sentence_list.append(tsentence)
    hour_list=[]
    for i in range(10):
        hour_list.append(float(data[i][1]))
    hour_list=convert_hour_to_onehot(hour_list)
    hour_list=Variable(torch.tensor([hour_list]).float())
    label=Variable(torch.tensor(convert_hour_to_onehot([float(data[-1][1])])).float())
    return sentence_list,hour_list,label



def subjective_bot():

    game_name_list = ['Counter-Strike Global Offensive', 'Transformice', 'Dead Island Epidemic', 'Dota 2', 'Team Fortress 2', 'War Thunder', "Garry's Mod", 'Injustice Gods Among Us Ultimate Edition', 'Loadout', 'Geometry Dash']
    hour_list = [6.0, 3.0, 2.0, 820.0, 250.0, 50.0, 36.0, 25.0, 14.0, 13.0]
    # SpeedRunners

    # game_name_list = ['Dota 2','Warframe','The Elder Scrolls V Skyrim','DayZ','DARK SOULS II','Trove','Fallout 4','Starbound','Endless Legend','Warhammer 40,000 Dawn of War II']
    # hour_list = [600.0, 300.0, 200.0, 820.0, 250.0, 500.0, 360.0, 250.0, 54.0, 130.0]
    # Endless Space

    # game_name_list = ['Dota 2' ,'Counter-Strike Global Offensive' ,'Warhammer 40,000 Dawn of War II - Chaos Rising' ,"NOBUNAGA'S AMBITION Sphere of Influence",'Endless Space','Shadowrun Hong Kong' ,'The Dark Eye Chains of Satinav','Demonicon' ,"Shadowrun Dragonfall - Director's Cut",'Total War SHOGUN 2' ]
    # hour_list = [100,100,100,100,5,20,20,5,5,10]
    # # new: The Elder Scrolls V Skyrim

    # game_name_list= ['Dota 2','Dota 2','Dota 2','Dota 2','Dota 2','Dota 2','Dota 2','Dota 2','Dota 2','Dota 2']
    # hour_list = [100, 500, 500, 500, 700, 200, 200, 500, 500, 10]

    newplayer = False
    model = torch.load('shuffledmodel_0.52.pt')
    # model.cuda()
    print('Hello There! Welcome to Check This Out!')
    print('Loading Essential Tools...')
    TEXT = data.Field(sequential=True, include_lengths=True, tokenize='spacy')
    LABEL = data.Field(sequential=False, use_vocab=False)
    abstract_data = data.TabularDataset(path='./data/abstract_tsv.tsv', skip_header=True, format='tsv',
                                        fields=[('text', TEXT), ('label', LABEL)])
    TEXT.build_vocab(abstract_data)
    Vocab = TEXT.vocab

    TEXTn = data.Field(sequential=True, include_lengths=True, tokenize='spacy')
    LABELn = data.Field(sequential=False, use_vocab=False)
    abstract_datan = data.TabularDataset(path='./data/full_abstract_tsv.tsv', skip_header=True, format='tsv',
                                        fields=[('text', TEXTn), ('label', LABELn)])
    TEXTn.build_vocab(abstract_datan)
    Vocabfull = TEXTn.vocab

    glove = torchtext.vocab.GloVe(name='6B', dim=100)
    Vocabfull.load_vectors(glove)
    embeds = nn.Embedding.from_pretrained(Vocabfull.vectors)
    abstract_dictionary = convert_csv_to_dict('./data/abstracts_final.csv')



    # game_name_list = []
    # hour_list = []
    # print('Complete!\n')
    # for i in range(10):
    #     # this is for entering the name of games
    #     name_true = 0
    #     while name_true != 1:
    #         name = input('Please enter NAME of game #{}:'.format(i+1))
    #         if name not in abstract_dictionary.keys():
    #             print('Sorry! The game is not recognized, please try again!')
    #         else:
    #             name_true = 1
    #             game_name_list.append(name)
    #
    #     # this is for entering the number of hour
    #     hour_true = 0
    #     while hour_true == 0:
    #         hour = input('Enter in HOURS, how much you have played this game:')
    #         try:
    #             float(hour)
    #             hour_list.append(float(hour))
    #             hour_true = 1
    #             print('\n')
    #         except:
    #             print('Sorry! The input is not valid, please try again!')

    while not newplayer:
        newgamelist = game_name_list[:]
        newhours = hour_list[:]
        name_true = 0
        print('\n')
        while name_true != 1:
            name = input('Please enter NAME of the NEW GAME:')
            # if name == "newplayer!":
            #     newplayer = True
            #     break
            if name not in abstract_dictionary.keys():
                print('Sorry! The game is not recognized, please try again!')
            else:
                name_true = 1
        newgamelist.append(name)
        newhours.append(0)

#==========================================================#
        # print('\n')
        print('Let us think about it!')
        temp_input = []
        for i in range(11):
            temp = [newgamelist[i], newhours[i]]
            temp_input.append(temp)

        net_cnn = torch.load('cnn_model_epoch0.pkl')
        abstract_list_cnn, hour_list_cnn, label_cnn = convert_data_cnn(temp_input, Vocabfull, embeds, abstract_dictionary)
        prediction_cnn = net_cnn.forward(abstract_list_cnn, hour_list_cnn)
        prediction_cnn = prediction_cnn.detach().numpy()
        max_cnn = prediction_cnn.argmax()

        results = ['%.3f' % elem for elem in prediction_cnn.tolist()]

        print('CNN:')
        print(results)
        # print(
        #     'the prediction of the cnn model is:' + str(prediction_cnn[0]) + ', ' + str(prediction_cnn[1]) + ', ' + str(
        #         prediction_cnn[2]) + ', ' + str(prediction_cnn[3]))
        if max_cnn == 0:
            print('I believe the player will be playing this new game for: 0 - 10 hours')
        if max_cnn == 1:
            print('I believe the player will be playing this new game for: 10 - 35 hours')
        if max_cnn == 2:
            print('I believe the player will be playing this new game for: 35 - 85 hours')
        if max_cnn == 3:
            print('I believe the player will be playing this new game for: above 85 hours')

#==========================================================#

        intomodel = []
        for i in range(11):
            intomodel.append(newgamelist[i])
            if newhours[i] < 10:
                intomodel.extend([1, 0, 0, 0])
            elif newhours[i] < 35:
                intomodel.extend([0, 1, 0, 0])
            elif newhours[i] < 85:
                intomodel.extend([0, 0, 1, 0])
            else:
                intomodel.extend([0, 0, 0, 1])
        intomodel = intomodel[:-4]
        nameindex = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        absfeatures = []
        for l in nameindex:
            absfeatures.append(sentence_preprocess_rnn(abstract_dictionary[intomodel[l]], Vocab).cuda())
        for k in range(50):
            if k not in nameindex:
                absfeatures.append(torch.tensor(float(intomodel[k])).cuda())
        predict = model(absfeatures)
        # print(predict)
        results_rnn = ['%.3f' % elem for elem in predict.detach().tolist()]
        predict = predict.argmax()
#===============================================================#
        print('RNN:')
        print(results_rnn)
        if predict == 0:
            print("Got it! I think you will play this game for less than 10 hours!")
        elif predict == 1:
            print("Got it! I think you will play this game for 10 to 35 hours!")
        elif predict == 2:
            print("Got it! I think you will play this game for 35 to 85 hours!")
        else:
            print("Got it! I think you will play this game for more than 85 hours!")


if __name__ == '__main__':
    subjective_bot()