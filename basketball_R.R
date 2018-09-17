#@Author Adi
#@Created on Mon Jul 18 07:32:24 2018
#@Purpose: Pre-Process and visualize the NBA shot data
#Random Forest implementation in R is slower and requires 
#more computation than the python version



library(randomForest)
library(ggplot2)


df<-read.csv("shot.csv",header=T)
train<-df[,c(-2,-4,-5,-17,-18)]



#this step used to convert shot into numeric data and convert into factor
train$SHOT_RESULT <- as.numeric(train$SHOT_RESULT)
train$SHOT_RESULT[train$SHOT_RESULT == 2]<-0 #made missed into 0 factor

#visualization of the frequency of game_clock
train$GAME_CLOCK <- as.numeric(train$GAME_CLOCK)
train$GAME_CLOCK<-train$GAME_CLOCK/60 #making a time 
ggplot(train, aes(SHOT_RESULT,GAME_CLOCK)) +
geom_bin2d(bins = 20,binwidth = c(0.4, 0.4))+
scale_x_continuous( breaks = seq(0, 1, 1),minor_breaks = seq(0 , 1, 1))+
scale_fill_gradientn(colours=c("yellow","red"),name = "Frequency", na.value=NA)

#making shot results into factors for random forest
train$SHOT_RESULT <- as.character(train$SHOT_RESULT)
train$SHOT_RESULT <- as.factor(train$SHOT_RESULT) 


train$LOCATION <- as.numeric(train$LOCATION)

#moving objective variable to the last column
SHOT_RESULT<-train$SHOT_RESULT
train<-train$SHOT_RESULT
train<-subset(train, select=-SHOT_RESULT)
train<-cbind(train,SHOT_RESULT) 


write.csv(train,"shot_pre.csv",row.names=FALSE,quote=FALSE)

#breaking up shots into three pointer and two pointer
three<-train[train$PTS_TYPE == 3,]
two<-train[train$PTS_TYPE == 2,]

#random forest for all shots
fit<-randomForest(SHOT_RESULT ~.,data =train,importance=TRUE,ntree=10,
na.action=na.omit,proximity=TRUE,mtry = 3)

#random forest for two pointers
two_v2<-subset(two, select=-PTS_TYPE)
fit2<-randomForest(SHOT_RESULT ~.,data=  two_v2,importance=TRUE,ntree=2000,
na.action=na.omit,proximity=TRUE,mtry = 3)

#random forest for three pointers
three_v2<-subset(three, select=-PTS_TYPE) #removing PTS_TYPE variable
fit3<-randomForest(SHOT_RESULT ~.,data = three_v2,importance=TRUE,ntree=2000,
na.action=na.omit,proximity=TRUE,mtry = 3)

#prediction for all shots
pred <- predict(fit, test,type='class')

varImpPlot(fit2)#get feature importance for 2pts shot
varImpPlot(fit3)



