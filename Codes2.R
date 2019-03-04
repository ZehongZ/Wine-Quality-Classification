#Import data set
red=read.csv("winequality-red.csv",header = T,sep = ";")
head(red)
dim(red)
white=read.csv("winequality-white.csv",header=T,sep=";")
head(white)
dim(white)

#Install library
library(e1071)
library(kernlab)
#Put two dataset together
myd=data.frame(rbind(red,white))
head(myd)
dim(myd)

#Detect missing values
library(Amelia)
library(mlbench)
missmap(myd,col=c("black","grey",legend=FALSE))
missmap(dataFilt, col=c("block","grey",legend=FALSE))
##No missing values

#Boxplot
library(caret)
dim(myd)
x=myd[,1:11]
y=myd[,12]
boxplot(myd$fixed.acidity)#Outliers
boxplot(myd$volatile.acidity)#Outliers
boxplot(myd$citric.acid)#Outliers
boxplot(myd$residual.sugar)#Outliers
boxplot(myd$chlorides)#Outliers
boxplot(myd$free.sulfur.dioxide)#Outliers
boxplot(myd$total.sulfur.dioxide)#Outliers
boxplot(myd$density)#2 outliers
boxplot(myd$pH)#Outliers
boxplot(myd$sulphates)#Outliers
boxplot(myd$alcohol)#3 Outliers

#Detect outliers outside 3 standard deviation
findOutlier=function(myd,cutoff=3){
  sds=apply(myd,2,sd,na.rm=TRUE)
  result=mapply(function(d,s){
    which(d>cutoff*s)
  }, myd,sds)
  result
}
outliers=findOutlier(myd)
outliers
length(outliers)
#Remove Outliers
removeOutliers=function(myd,outliers){
  result=mapply(function(d,o){
    res=d
    res[0]=NA
    return(res)
  }, myd,outliers)
  return(as.data.frame(result))
}
dataFilt=removeOutliers(myd,outliers)
dim(dataFilt)
dim(myd)
##Only 12 outliers

#Remove outlier method
mydc=myd
vars=c("fixed.acidity","volatile.acidity","citric.acid","residual.sugar","chlorides","free.sulfur.dioxide","total.sulfur.dioxide","density","pH","sulphates","alcohol")
Outliers=c()
for (i in vars){
  max=quantile(mydc[,i],0.75,na.rm = TRUE)+(IQR(mydc[,i],na.rm = TRUE))
  min=quantile(mydc[,i],0.25,na.rm = TRUE)-(IQR(mydc[,i],na.rm= TRUE))
  idx=which(mydc[,i]<min  | mydc[,i]>max)
  print(paste(i,length(idx),sep = ''))
  Outliers=c(Outliers,idx)
}
Outliers=sort(Outliers)
mydc=mydc[-Outliers,]
dim(mydc)

#Normalize
mydc[,1:11]=scale(mydc[,1:11])
head(mydc)

#Split data into training and testing set
trainIndex=createDataPartition(mydc$quality,p=0.66, list=FALSE)
train=mydc[trainIndex,]
test=mydc[-trainIndex,]

#Train individual class
three=train
three$quality=as.character(three$quality)
three$quality[three$quality!="3"]="0"
three$quality[three$quality=="3"]='1'
three$quality=as.factor(three$quality)
tune.three=tune.svm(quality~., data=train, gamma=10^(-6:1), cost=10^(-1:1))
summary(tune.three)
model.three=svm(quality~., data=three, kernel="radial",gamma=1,cost=10,scale=TRUE,probabilities=TRUE, na.action = na.omit)
summary(model.three)
predict.three=predict(model.three, test[,-12])
table.three=table(predict.three, test[,12])
table.three

four=train
four$quality=as.character(four$quality)
four$quality[four$quality!="4"]="0"
four$quality[four$quality=="4"]='1'
four$quality=as.factor(four$quality)
model.four=svm(quality~., data=three, kernel="radial",gamma=1,cost=10,scale=TRUE,probabilities=TRUE, na.action = na.omit)
summary(model.four)
predict.four=predict(model.four, test[,-12])
table.four=table(predict.four, test[,12])
table.four

five=train
five$quality=as.character(five$quality)
five$quality[five$quality!="5"]="0"
five$quality[four$quality=="5"]='1'
five$quality=as.factor(five$quality)
model.five=svm(quality~., data=five, kernel="radial",gamma=1,cost=10,scale=TRUE,probabilities=TRUE, na.action = na.omit)
summary(model.five)
predict.five=predict(model.five, test[,-12])
table.five=table(predict.five, test[,12])
table.five

six=train
six$quality=as.character(six$quality)
six$quality[six$quality!="6"]="0"
six$quality[six$quality=="6"]='1'
six$quality=as.factor(six$quality)
model.six=svm(quality~., data=six, kernel="radial",gamma=1,cost=10,scale=TRUE,probabilities=TRUE, na.action = na.omit)
summary(model.six)
predict.six=predict(model.six, test[,-12])
table.six=table(predict.six, test[,12])
table.six

seven=train
seven$quality=as.character(seven$quality)
seven$quality[seven$quality!="7"]="0"
seven$quality[seven$quality=="7"]='1'
seven$quality=as.factor(seven$quality)
model.seven=svm(quality~., data=seven, kernel="radial",gamma=1,cost=10,scale=TRUE,probabilities=TRUE, na.action = na.omit)
summary(model.seven)
predict.seven=predict(model.seven, test[,-12])
table.seven=table(predict.seven, test[,12])
table.seven

eight=train
eight$quality=as.character(eight$quality)
eight$quality[eight$quality!="8"]="0"
eight$quality[eight$quality=="8"]='1'
eight$quality=as.factor(eight$quality)
model.eight=svm(quality~., data=eight, kernel="radial",gamma=1,cost=10,scale=TRUE,probabilities=TRUE, na.action = na.omit)
summary(model.eight)
predict.eight=predict(model.eight, test[,-12])
table.eight=table(predict.eight, test[,12])
table.eight

nine=train
nine$quality=as.character(nine$quality)
nine$quality[nine$quality!="9"]="0"
nine$quality[nine$quality=="9"]='1'
nine$quality=as.factor(nine$quality)
model.nine=svm(quality~., data=nine, kernel="radial",gamma=1,cost=10,scale=TRUE,probabilities=TRUE, na.action = na.omit)
summary(model.nine)
predict.nine=predict(model.nine, test[,-12])
table.nine=table(predict.nine, test[,12])
table.nine

bind=cbind(predict.three,predict.four,predict.five,predict.six,predict.seven,predict.eight,predict.nine)
classnames=c('three','four','five','six','seven','eight','nine')
a=apply(bind,1,classnames[which.max])

