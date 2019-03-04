#Import dataset
myd=read.csv("Wine.csv",header = T)
myd$quality=as.factor(myd$quality)
head(myd)
levels(myd$quality)
typeof(myd)
typeof(mydc)

#Install library
library(kernlab)
library(e1071)
library(caret)
library(ROCR)
library(pROC)
library(gmum)
#Check the dataset
dim(myd)
levels(myd$quality)
boxplot(myd[,1:11])

#Remove outlier method
mydc=data.frame(myd)
mydc$quality=as.factor(mydc$quality)
vars=c("fixedacidity","volatileacidity","citricacid","residualsugar","chlorides","freesulfurdioxide","totalsulfurdioxide","density","pH","sulphates","alcohol")
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
head(mydc$quality)

#Split data into training and testing set
trainIndex=createDataPartition(mydc$quality,p=0.66, list=FALSE)
train=mydc[trainIndex,]
test=mydc[-trainIndex,]
train[["quality"]]=factor(train[["quality"]])

#Set and Tune parameters
grid=expand.grid(C=c(0.1,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5))
trctrl=trainControl(method="cv", number=10)
grid_radial=expand.grid(sigma=c(0.1,0.2,0.3,0.4,0.5,1,1.1,1.2,1.3,1.4),C=c(0.1,0.2,0.3,0.4,0.5,1,2,3,4,5))
linear=tune.svm(quality~., data=train, cost=2^(1:5),kernel="linear")
nonlinear=tune.svm(quality~., data=train, cost=2^(1:5),gamma=2^(0.1:1), kernel="radial")
nonlinear

##Cross Validation
#Linear
start.time=Sys.time()
cv.lsvm=svm(as.factor(quality)~., data=train, kernel="linear", trControl=trctrl, preProcess=c("center","scale"), tuneGrid=grid, tuneLength=10)
end.time=Sys.time()
time.taken=end.time-start.time
time.taken
cv.lsvm
train_cv.lsvm=predict(cv.lsvm, newdata=train)
confusionMatrix(train_cv.lsvm, train$quality)
test_cv.lsvm=predict(cv.lsvm,newdata=test)
confusionMatrix(test_cv.lsvm, test$quality)
##ROC Analysis
perf_cv.lsvm=prediction(as.numeric(train_cv.lsvm),as.numeric(train$quality))
pred_cv.lsvm=performance(perf_cv.lsvm, "tpr","fpr")
plot(pred_cv.lsvm,colorize=TRUE,main="ROC Curve")
auc_cv.lsvm=performance(perf_cv.lsvm,measure = "auc")
auc_cv.lsvm@y.values[[1]]
#Non-linear
start.time=Sys.time()
cv.nlsvm=svm(quality~.,data=train, kernel="radial",trControl=trctrl, preProcess=c("center","scale"), tuneGrid=grid_radial, tuneLength=10)
end.time=Sys.time()
time.taken=end.time-start.time
time.taken
train_cv.nlsvm=predict(cv.nlsvm,data=train)
confusionMatrix(train_cv.nlsvm, train$quality)
test_cv.nlsvm=predict(cv.nlsvm,newdata=test)
confusionMatrix(test_cv.nlsvm,test$quality)
##ROC Analysis
perf_cv.nlsvm=prediction(as.numeric(train_cv.nlsvm),as.numeric(train$quality))
pred_cv.nlsvm=performance(perf_cv.nlsvm, "tpr","fpr")
plot(pred_cv.nlsvm,colorize=TRUE,main="ROC Curve")
auc_cv.nlsvm=performance(perf_cv.nlsvm,measure = "auc")
auc_cv.nlsvm@y.values[[1]]


##Repeat Cross Validation
#Linear
trctrl=trainControl(method="repeatedcv", number=10,repeats = 3)
time.start=Sys.time()
rcv.lsvm=svm(quality~., data=train, kernel="linear", trControl=trctrl, preProcess=c("center","scale"), tuneGrid=grid, tuneLength=10)
time.end=Sys.time()
time.taken=time.end-time.start
time.taken
rcv.lsvm
train_rcv.lsvm=predict(rcv.lsvm,newdata=train)
confusionMatrix(train_rcv.lsvm, train$quality)
test_rcv.lsvm=predict(rcv.lsvm, newdata=test)
confusionMatrix(test_cv.lsvm, test$quality)
##ROC Analysis
perf_rcv.lsvm=prediction(as.numeric(train_rcv.lsvm),as.numeric(train$quality))
pred_rcv.lsvm=performance(perf_rcv.lsvm, "tpr","fpr")
plot(pred_rcv.lsvm,colorize=TRUE,main="ROC Curve")
auc_rcv.lsvm=performance(perf_rcv.lsvm,measure = "auc")
auc_rcv.lsvm@y.values[[1]]

#Non-linear
time.start=Sys.time()
rcv.nlsvm=svm(quality~., data=train, kernel="radial", trControl=trctrl, preProcess=c("center","scale"), tuneGrid=grid_radial, tuneLength=10)
time.end=Sys.time()
time.taken=time.end-time.start
time.taken
rcv.nlsvm
train_rcv.nlsvm=predict(rcv.nlsvm, newdata=train)
confusionMatrix(train_rcv.nlsvm, train$quality)
test_rcv.nlsvm=predict(rcv.nlsvm, newdata=test)
confusionMatrix(test_rcv.nlsvm, test$quality)
##ROC Analysis
perf_rcv.nlsvm=prediction(as.numeric(train_rcv.nlsvm),as.numeric(train$quality))
pred_rcv.nlsvm=performance(perf_rcv.nlsvm, "tpr","fpr")
plot(pred_rcv.nlsvm,colorize=TRUE,main="ROC Curve")
auc_rcv.nlsvm=performance(perf_rcv.nlsvm,measure = "auc")
auc_rcv.nlsvm@y.values[[1]]

##Leave-one-out Cross Validation
#Linear
trctrl=trainControl(method="LOOCV")
time.start=Sys.time()
loocv.lsvm=svm(quality~., data=train, kernel="linear", trControl=trctrl, preProcess=c("center","scale"), tuneGrid=grid, tuneLength=10)
time.end=Sys.time()
time.taken=time.end-time.start
time.taken
loocv.lsvm
train_loocv.lsvm=predict(loocv.lsvm, newdata=train)
confusionMatrix(train_loocv.lsvm, train$quality)
test_loocv.lsvm=predict(loocv.lsvm,newdata=test)
confusionMatrix(test_loocv.lsvm, test$quality)
##ROC Analysis
perf_loocv.lsvm=prediction(as.numeric(train_loocv.lsvm),as.numeric(train$quality))
pred_loocv.lsvm=performance(perf_loocv.lsvm, "tpr","fpr")
plot(pred_loocv.lsvm,colorize=TRUE,main="ROC Curve")
auc_loocv.lsvm=performance(perf_loocv.lsvm,measure = "auc")
auc_loocv.lsvm@y.values[[1]]
#Non-linear
time.start=Sys.time()
loocv.nlsvm=svm(quality~., data=train, kernel="radial", trControl=trctrl, preProcess=c("center","scale"), tuneGrid=grid_radial, tuneLength=10)
time.end=Sys.time()
time.taken=time.end-time.start
time.taken
loocv.nlsvm
train_loocv.nlsvm=predict(loocv.nlsvm, newdata=train)
confusionMatrix(train_loocv.nlsvm,train$quality)
test_loocv.nlsvm=predict(loocv.nlsvm,newdata=test)
confusionMatrix(test_loocv.nlsvm,test$quality)
##ROC Analysis
perf_loocv.nlsvm=prediction(as.numeric(train_loocv.nlsvm),as.numeric(train$quality))
pred_loocv.nlsvm=performance(perf_loocv.nlsvm, "tpr","fpr")
plot(pred_loocv.nlsvm,colorize=TRUE,main="ROC Curve")
auc_loocv.nlsvm=performance(perf_loocv.nlsvm,measure = "auc")
auc_loocv.nlsvm@y.values[[1]]

#Visualization
plot(cv.lsvm, formula=fixedacidity~alcohol,data=train)
plot(cv.nlsvm, formula=fixedacidity~alcohol,data=train)
plot(rcv.lsvm, formula=fixedacidity~alcohol,data=train)
plot(rcv.nlsvm, formula=fixedacidity~alcohol,data=train)
plot(loocv.lsvm, formula=fixedacidity~alcohol,data=train)
plot(loocv.nlsvm, formula=fixedacidity~alcohol,data=train)

