import pandas as pd
import torch
import numpy as np
from random import shuffle

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





# # This block is for reshaping the trimmed data into the form of:
# # every row being 11 entries representing 1 players top 11 games
# # should produce a npy file of size (10109, 11, 5) where 10109 is the total entries (11*1372 - entries with games not picked up by webscraper)
# # 11 is the size of each entry, since it consists of the top 11 games, each game is 5 items: name ,one hot for hours
#
# def convert_csv_to_dict(file):
#     # by using the ./data/abstracts_final.csv as input, this function returns a dictionary for the games.
#     ans=pd.read_csv(file)
#     ans=ans.values.tolist()
#     ans=dict(ans)
#     return ans
#
# def check_in_diction(dictionary,data):
#     count=0
#     for i in range(len(data)):
#         if data[i][0] in dictionary.keys():
#             count+=1
#     if count==11:
#         return True
#     if count!=11:
#         return False
#
# get_abs = convert_csv_to_dict("./data/abstracts_final.csv")
# data = pd.read_csv("./data/trimmed.csv")
# col = ["UserID", "Game", "Hours"]
# # currid = 0
# listofids = data["UserID"].unique()
# # finalset = [["game1","hours1"],["game2","hours2"],["game3","hours3"],["game4","hours4"],["game5","hours5"],["game6","hours6"],["game7","hours7"],["game8","hours8"],["game9","hours9"],["game10","hours10"],["game11","hours11"]]
# # finalset = np.array(finalset)
# finalset = []
# finalnothot = []
# j = 0
# for i,id in enumerate(listofids): #there are 1477 players with >= 10 games played on record # and 9873 players with less
#     theirgames = []               #there are 1372 players with >= 11 games played on record # and 9978 players with less
#     while data[col[0]][j] == id:
#         theirgames.append([data[col[1]][j], data[col[2]][j]])
#         j += 1
#         if j == len(data[col[0]]):
#             break
#     theirgames.sort(key=lambda x: x[1], reverse=True)
#     if len(theirgames)>20:
#         theirgames = theirgames[:20]
#     shuffle(theirgames)
#
# #============================
#     # nothot = theirgames[:]
#     # topnothot = nothot[:10]
#     # for jj in range(10,len(nothot)):
#     #     tempa = topnothot[:]
#     #     tempa.append(nothot[jj])
#     #     finalnothot.append(tempa)
#     #     if jj > 20:
#     #         break
# #==================================
#
#     for gamnum, gam in enumerate(theirgames):
#         hourstocat = float(gam[1])
#         if hourstocat < 10:
#             gam[1] = 1
#             gam.extend([0, 0, 0])
#         elif hourstocat < 35:
#             gam[1] = 0
#             gam.extend([1, 0, 0])
#             # theirtopgames[gamnum] = np.concatenate((gam, np.array([1, 0, 0])))
#         elif hourstocat < 85:
#             gam[1] = 0
#             gam.extend([0, 1, 0])
#             # theirtopgames[gamnum] = np.concatenate((gam, np.array([0, 1, 0])))
#         else:
#             gam[1] = 0
#             gam.extend([0, 0, 1])
#             # theirtopgames[gamnum] = np.concatenate((gam, np.array([0, 0, 1])))
#
#     theirtopgames = theirgames[:10]
#
#     for ii in range(10,len(theirgames)):
#         tempo = theirtopgames[:]
#         tempo.append(theirgames[ii])
#         finalset.append(tempo)
#         if ii > 20:
#             break
#
#
#     # theirtopgames = np.array(theirtopgames)
#     # for k in range(11):
#     #     finalset.append(np.roll(theirtopgames, k, axis=0).tolist())
#     if j == len(data[col[0]]):  # stops the loop
#         break
#
# removethese = []
# for i, entry in enumerate(finalset):
#     if check_in_diction(get_abs, entry):
#         pass
#     else:
#         removethese.append(entry)
# for i in removethese:
#     finalset.remove(i)
#
# finalset = np.array(finalset)
# np.save("./data/tophours11.npy", finalset)

# removethese = []
# for i, entry in enumerate(finalnothot):
#     if check_in_diction(get_abs, entry):
#         pass
#     else:
#         removethese.append(entry)
# for i in removethese:
#     finalnothot.remove(i)
#
# finalnothot = np.array(finalnothot)
# np.save("./data/hoursnothotshuffled.npy", finalnothot)


# (26148, 11, 5) if we include all their games
# (8250, 11, 5) if we include top 21(or less)
# (5949, 11, 2) if top 20

hours = np.load("./data/playedhours_finalv2.npy")
print(hours.shape)
casenum = [0, 0, 0, 0]
# for i in hours:
#     casenum += i[-1, -4:].astype(int)
# print(casenum)
for i in hours:
    print(i.tolist())















#==================================================================================================#
#---------------------------------THIS BLOCK IS NO LONGER IN USE ----------------------------------#
# original purpose: modified the data set to have hours instead of one hot in the final version
#
# def convert_csv_to_dict(file):
#     # by using the ./data/abstracts_final.csv as input, this function returns a dictionary for the games.
#     ans=pd.read_csv(file)
#     ans=ans.values.tolist()
#     ans=dict(ans)
#     return ans
#
# def check_in_diction(dictionary,data):
#     count=0
#     for i in range(len(data)):
#         if data[i][0] in dictionary.keys():
#             count+=1
#     if count==11:
#         return True
#     if count!=11:
#         return False
#
# get_abs = convert_csv_to_dict("./data/abstracts_final.csv")
# data = pd.read_csv("./data/trimmed.csv")
# col = ["UserID", "Game", "Hours"]
# # currid = 0
# listofids = data["UserID"].unique()
# # finalset = [["game1","hours1"],["game2","hours2"],["game3","hours3"],["game4","hours4"],["game5","hours5"],["game6","hours6"],["game7","hours7"],["game8","hours8"],["game9","hours9"],["game10","hours10"],["game11","hours11"]]
# # finalset = np.array(finalset)
# finalset = []
# j = 0
# for i,id in enumerate(listofids): #there are 1477 players with >= 10 games played on record # and 9873 players with less
#     theirgames = []               #there are 1372 players with >= 11 games played on record # and 9978 players with less
#     while data[col[0]][j] == id:
#         theirgames.append([data[col[1]][j],data[col[2]][j]])
#         j += 1
#         if j == len(data[col[0]]):
#             break
#     theirgames.sort(key=lambda x: x[1],reverse=True)
#     theirtopgames = np.array(theirgames[:11])
#     for k in range(11):
#         finalset.append(np.roll(theirtopgames, k, axis=0).tolist())
#     if j == len(data[col[0]]):
#         break
#
# removethese = []
# for i, entry in enumerate(finalset):
#     if check_in_diction(get_abs, entry):
#         pass
#     else:
#         removethese.append(entry)
# for i in removethese:
#     finalset.remove(i)
#
# finalset = np.array(finalset)
# np.save("./data/tophours11notonehot.npy", finalset)







#==================================================================================================#
#---------------------------------THIS BLOCK IS NO LONGER IN USE ----------------------------------#
# # ROTATES 11 INSTEAD OF ADDING TO TOP 10
# # This block is for reshaping the trimmed data into the form of:
# # every row being 11 entries representing 1 players top 11 games
# # should produce a npy file of size (10109, 11, 5) where 10109 is the total entries (11*1372 - entries with games not picked up by webscraper)
# # 11 is the size of each entry, since it consists of the top 11 games, each game is 5 items: name ,one hot for hours
#
# def convert_csv_to_dict(file):
#     # by using the ./data/abstracts_final.csv as input, this function returns a dictionary for the games.
#     ans=pd.read_csv(file)
#     ans=ans.values.tolist()
#     ans=dict(ans)
#     return ans
#
# def check_in_diction(dictionary,data):
#     count=0
#     for i in range(len(data)):
#         if data[i][0] in dictionary.keys():
#             count+=1
#     if count==11:
#         return True
#     if count!=11:
#         return False
#
# get_abs = convert_csv_to_dict("./data/abstracts_final.csv")
# data = pd.read_csv("./data/trimmed.csv")
# col = ["UserID", "Game", "Hours"]
# # currid = 0
# listofids = data["UserID"].unique()
# # finalset = [["game1","hours1"],["game2","hours2"],["game3","hours3"],["game4","hours4"],["game5","hours5"],["game6","hours6"],["game7","hours7"],["game8","hours8"],["game9","hours9"],["game10","hours10"],["game11","hours11"]]
# # finalset = np.array(finalset)
# finalset = []
# j = 0
# for i,id in enumerate(listofids): #there are 1477 players with >= 10 games played on record # and 9873 players with less
#     theirgames = []               #there are 1372 players with >= 11 games played on record # and 9978 players with less
#     while data[col[0]][j] == id:
#         theirgames.append([data[col[1]][j], data[col[2]][j]])
#         j += 1
#         if j == len(data[col[0]]):
#             break
#     theirgames.sort(key=lambda x: x[1], reverse=True)
#     theirtopgames = theirgames[:11]
#
#     for gamnum, gam in enumerate(theirtopgames):
#         hourstocat = float(gam[1])
#         if hourstocat < 10:
#             gam[1] = 1
#             gam.extend([0, 0, 0])
#         elif hourstocat < 35:
#             gam[1] = 0
#             gam.extend([1, 0, 0])
#             # theirtopgames[gamnum] = np.concatenate((gam, np.array([1, 0, 0])))
#         elif hourstocat < 85:
#             gam[1] = 0
#             gam.extend([0, 1, 0])
#             # theirtopgames[gamnum] = np.concatenate((gam, np.array([0, 1, 0])))
#         else:
#             gam[1] = 0
#             gam.extend([0, 0, 1])
#             # theirtopgames[gamnum] = np.concatenate((gam, np.array([0, 0, 1])))
#     theirtopgames = np.array(theirtopgames)
#     for k in range(11):
#         finalset.append(np.roll(theirtopgames, k, axis=0).tolist())
#     if j == len(data[col[0]]):  # stops the loop
#         break
#
# removethese = []
# for i, entry in enumerate(finalset):
#     if check_in_diction(get_abs, entry):
#         pass
#     else:
#         removethese.append(entry)
# for i in removethese:
#     finalset.remove(i)
#
# finalset = np.array(finalset)
# np.save("./data/tophours11.npy", finalset)
#
#
# hours = np.load("./data/playedhours_finalv2_hot_new.npy")
# print(hours.shape)
# # casenum = [0, 0, 0, 0]
# # for i in hours:
# #     casenum += i[-1, -4:].astype(int)
# # print(casenum)
# for i in hours:
#     print (i)
