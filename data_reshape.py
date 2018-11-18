import pandas as pd
import torch
import numpy as np

# Trimm the data to only include players with at least 11 games
# data = pd.read_csv("./data/steam_removedtype.csv")
# col = ["UserID", "Game", "Hours"]
# toberemoved = []
# tobeleftin = []
#
# j = 0
# while j < len(data[col[0]]):
#     id = data[col[0]][j]
#     numofgames = (data['UserID'] == id).sum()
#     if numofgames < 11:
#         toberemoved.append(id)
#     else:
#         tobeleftin.append(id)
#     j += numofgames
# print(len(tobeleftin), len(toberemoved))
# for i in toberemoved:
#     data = data[data['UserID'] != i]
# data.to_csv('./data/trimmed.csv', index=False)





# This block is for reshaping the trimmed data into the form of:
# every row being 11 entries representing 1 players top 11 games

def convert_csv_to_dict(file):
    # by using the ./data/abstracts_final.csv as input, this function returns a dictionary for the games.
    ans=pd.read_csv(file)
    ans=ans.values.tolist()
    ans=dict(ans)
    return ans

def check_in_diction(dictionary,data):
    count=0
    for i in range(len(data)):
        if data[i][0] in dictionary.keys():
            count+=1
    if count==11:
        return True
    if count!=11:
        return False

get_abs = convert_csv_to_dict("./data/abstracts_final.csv")
data = pd.read_csv("./data/trimmed.csv")
col = ["UserID", "Game", "Hours"]
# currid = 0
listofids = data["UserID"].unique()
# finalset = [["game1","hours1"],["game2","hours2"],["game3","hours3"],["game4","hours4"],["game5","hours5"],["game6","hours6"],["game7","hours7"],["game8","hours8"],["game9","hours9"],["game10","hours10"],["game11","hours11"]]
# finalset = np.array(finalset)
finalset = []
j = 0
for i,id in enumerate(listofids): #there are 1477 players with more than 10 games played on record # and 9873 players with less
    theirgames = []               #there are 1372 players with more than 10 games played on record # and 9978 players with less
    while data[col[0]][j] == id:
        theirgames.append([data[col[1]][j],data[col[2]][j]])
        j += 1
        if j == len(data[col[0]]):
            break
    theirgames.sort(key=lambda x: x[1],reverse=True)
    theirtopgames = np.array(theirgames[:11])
    for k in range(11):
        finalset.append(np.roll(theirtopgames, k, axis=0).tolist())
    if j == len(data[col[0]]):
        break

removethese = []
for i, entry in enumerate(finalset):
    if check_in_diction(get_abs, entry):
        pass
    else:
        removethese.append(entry)
for i in removethese:
    finalset.remove(i)

finalset = np.array(finalset)
np.save("./data/tophours11.npy", finalset)

hours = np.load("./data/tophours11.npy")
print(hours.shape)
for i in hours:
    print(i)



