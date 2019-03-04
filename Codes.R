#Import data set
red=read.csv("winequality-red.csv",header = T,sep = ";")
head(red)
dim(red)
white=read.csv("winequality-white.csv",header=T,sep=";")
head(white)
dim(white)

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
length(myd)
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

#Perform multiclass SVM
library(e1071)
attach(mydc)
mydc$quality=as.factor(mydc$quality)
x=subset(mydc, select=-quality)
y=quality
model=svm(x,y,kernel="radial",cost=1,gamma=1,probability = TRUE)
pred=predict(model,x,decision.values = TRUE, probability = TRUE)
attr(pred, "probabilities")
table(pred, mydc$quality)


#Plot
plot(cmdscale(dist(mydc[,-12])),col=as.integer(mydc[,12]),pch=c("o","+")[1:150%in% model$index+1])
library(pROC)
prediction=as.numeric(pred)
plot(pred)

#ROC analysis
roc=multiclass.roc(mydc$quality,prediction)
auc(roc)
