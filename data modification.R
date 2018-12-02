
library(utf8)
library(readxl)
library(dplyr)
library(tidyr)
library(ggplot2)


# these were for the first dataset

abstracts <- read.csv("Abstracts_Final.csv");
abstracts_trimmed <- abstracts[!abstracts$Abstract == "", ]
abstracts_trimmed <- abstracts_trimmed[-c(1)]
write.csv(abstracts_trimmed,'abstracts.csv',row.names=False)


steam <- read.csv("steam-200k.csv");
steam_trimmed <- steam[!steam$Type == "purchase",]


# removed_pur <- dat[grep("purchase", dat['Type'])]
steam_trimmed <- steam_trimmed[-c(3,5)]
steam_trimmed <- steam_trimmed[order(steam_trimmed$UserID),]
write.csv(steam_trimmed,'steam_trimmed.csv',row.names=FALSE)


final <- steam_trimmed%>%group_by(UserID)%>%summarise(totalhours = sum(Hours))


gamenames <- abstracts_trimmed[,1]
playerids <- final[,1]


gamenames <-gamenames[1:10]

for (game in gamenames){
  final <- cbind(final,game[1]=0)
}
  
# # to get the most played list of games
# gamehours<-steam%>%group_by(Game)%>%summarise(totalhours = sum(Hours))
# write.csv(gamehours,'mostplayedgames.csv')


