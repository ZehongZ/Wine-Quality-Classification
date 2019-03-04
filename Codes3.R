#Import dataset
myd=read.csv("Wine.csv",header = T)
myd$quality=as.factor(myd$quality)
head(myd)
levels(myd$quality)

#Install library
library(kernlab)
library(caret)
library(ROCR)
library(e1071)

#Check the dataset
dim(myd)
levels(myd$quality)
boxplot(myd[,1:11])

#Remove outlier method
mydc=myd
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
head(mydc)

#Split data into training and testing set
trainIndex=createDataPartition(mydc$quality,p=0.66, list=FALSE)
train=mydc[trainIndex,]
test=mydc[-trainIndex,]
train[["quality"]]=factor(train[["quality"]])

#Set and Tune parameters
grid=expand.grid(C=c(0.1,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5))
trctrl=trainControl(method="cv", number=10)
grid_radial=expand.grid(sigma=c(0.1,0.2,0.3,0.4,0.5,1,1.1,1.2,1.3,1.4),C=c(0.1,0.2,0.3,0.4,0.5,1,2,3,4,5))

#Train Cross-validation, linear SVM
start.time=Sys.time()
cv.svm.l=train(quality~., data=train, method="svmLinear", trControl=trctrl, preProcess=c("center","scale"), tuneGrid=grid, tuneLength=10 )
end.time=Sys.time()
time.taken=end.time-start.time
time.taken
cv.svm.l
train_cv.svm.l=predict(cv.svm.l, newdata=train)
confusionMatrix(train_cv.svm.l, train$quality)
plot(train_cv.svm.l)
plot(cv.svm.l)
#Testing Cross-validation, Linear SVM
test_cv.svm.l=predict(cv.svm.l, newdata=test)
confusionMatrix(test_cv.svm.l, test$quality)
plot(test_cv.svm.l)
#ROC analysis
library(pROC)
library(ROCR)
perf_cv.svm.l=prediction(as.numeric(test_cv.svm.l),as.numeric(test$quality))
pred_cv.svm.l=performance(perf_cv.svm.l, "tpr","fpr")
plot(pred_cv.svm.l,colorize=TRUE,main="ROC Curve")
auc_cv.svm.l=performance(perf_cv.svm.l,measure = "auc")
auc_cv.svm.l@y.values[[1]]


#Train Cross-validation, Non-Linear SVM
start.time=Sys.time()
cv.svm.nl=train(quality~., data=train, method="svmRadial", trControl=trctrl, preProcess=c("center","scale"), tuneGrid=grid_radial, tuneLength=10)
end.time=Sys.time()
time.taken=end.time-start.time
time.taken
cv.svm.nl
plot(cv.svm.nl)
train_cv.svm.nl=predict(cv.svm.nl, newdata=train)
confusionMatrix(train_cv.svm.nl,train$quality)
#Test Cross-validation, Non-linear SVM
test_cv.svm.nl=predict(cv.svm.nl, newdata=test)
confusionMatrix(test_cv.svm.nl, test$quality)
#ROC analysis
library(pROC)
library(ROCR)
perf_cv.svm.nl=prediction(as.numeric(test_cv.svm.nl),as.numeric(test$quality))
pred_cv.svm.nl=performance(perf_cv.svm.nl, "tpr","fpr")
plot(pred_cv.svm.nl,colorize=TRUE,main="ROC Curve")
auc_cv.svm.nl=performance(perf_cv.svm.nl,measure = "auc")
auc_cv.svm.nl@y.values[[1]]

#Train Repeated Cross-validation, linear SVM
trctrl=trainControl(method="repeatedcv", number=10,repeats = 3)
start.time=Sys.time()
rcv.svm.l=train(quality~., data=train, method="svmLinear", trControl=trctrl, preProcess=c("center","scale"), tuneGrid=grid, tuneLength=10 )
end.time=Sys.time()
time.taken=end.time-start.time
time.taken
rcv.svm.l
train_rcv.svm.l=predict(rcv.svm.l, newdata=train)
confusionMatrix(train_rcv.svm.l, train$quality)
plot(rcv.svm.l)
#Testing Repeated Cross-validation, Linear SVM
test_rcv.svm.l=predict(rcv.svm.l, newdata=test)
confusionMatrix(test_rcv.svm.l, test$quality)
plot(test_rcv.svm.l)
#ROC analysis
library(pROC)
library(ROCR)
perf_rcv.svm.l=prediction(as.numeric(test_rcv.svm.l),as.numeric(test$quality))
pred_rcv.svm.l=performance(perf_rcv.svm.l, "tpr","fpr")
plot(pred_rcv.svm.l,colorize=TRUE,main="ROC Curve")
auc_rcv.svm.l=performance(perf_rcv.svm.l,measure = "auc")
auc_rcv.svm.l@y.values[[1]]

#Train Repeated Cross-validation, Non-linear SVM
start.time=Sys.time()
rcv.svm.nl=train(quality~., data=train, method="svmRadial", trControl=trctrl, preProcess=c("center","scale"), tuneGrid=grid_radial, tuneLength=10)
end.time=Sys.time()
time.taken=end.time-start.time
time.taken
rcv.svm.nl
plot(rcv.svm.nl)
train_rcv.svm.nl=predict(rcv.svm.nl,newdata=train)
confusionMatrix(train_rcv.svm.nl, train$quality)
#Test Cross-validation, Non-linear SVM
test_rcv.svm.nl=predict(rcv.svm.nl, newdata=test)
confusionMatrix(test_rcv.svm.nl, test$quality)
#ROC analysis
library(pROC)
library(ROCR)
perf_rcv.svm.nl=prediction(as.numeric(test_rcv.svm.nl),as.numeric(test$quality))
pred_rcv.svm.nl=performance(perf_rcv.svm.nl, "tpr","fpr")
plot(pred_rcv.svm.nl,colorize=TRUE,main="ROC Curve")
auc_rcv.svm.nl=performance(perf_rcv.svm.nl,measure = "auc")
auc_rcv.svm.nl@y.values[[1]]

#Train Leave-one-out Cross-validation, Linear
trctrl=trainControl(method="LOOCV")
start.time=Sys.time()
loocv.svm.l=train(quality~., data=train, method="svmLinear", trControl=trctrl, preProcess=c("center","scale"), tuneGrid=grid, tuneLength=10)
end.time=Sys.time()
time.taken=end.time-start.time
time.taken

