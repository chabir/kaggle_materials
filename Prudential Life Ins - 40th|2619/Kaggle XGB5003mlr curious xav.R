# R XGBOOST starter script

#devtools::install_github('dmlc/xgboost', subdir='R-package')

# The readr library is the best way to read and write CSV files in R

setwd("D:/NYC Data Science Academy/Kaggle/Prudential")

library(readr)
library(reshape2)
library(xgboost)
library(data.table)
library(caret)
library(Metrics)
library(mlr)

# The competition datafiles are in the directory ../input
# Read competition data files:
train <- read_csv("train_feat_feat_6.csv")
test  <- read_csv("test_feat_feat_6.csv")

staking = 0


qbmic <- 0.8
qbmic2 <- 0.9
#Hand engineered features. Found by EDA (especially added variable plots), some parameters optimized
#using cross validation. Nonlinear dependence on BMI and its interaction with age make intuitive sense.
custom_var_1 <- as.numeric(train$Medical_History_15 < 10.0)
custom_var_1[is.na(custom_var_1)] <- 0.0 #impute these NAs with 0s, note that they were not median-imputed
custom_var_3 <- as.numeric(train$Product_Info_4 < 0.075)
custom_var_4 <- as.numeric(train$Product_Info_4 == 1)
custom_var_6 <- (train$BMI + 1.0)**2.0
custom_var_7 <- (train$BMI)**0.8
custom_var_8 <- train$Ins_Age**8.5
custom_var_9 <- (train$BMI*train$Ins_Age)**2.5
BMI_cutoff <- quantile(train$BMI, qbmic)
custom_var_10 <- as.numeric(train$BMI > BMI_cutoff)
custom_var_11 <- (train$BMI*train$Product_Info_4)**0.9
ageBMI_cutoff <- quantile(train$Ins_Age*train$BMI, qbmic2)
custom_var_12 <- as.numeric(train$Ins_Age*train$BMI > ageBMI_cutoff)
#custom_var_13 <- (train$BMI*train$Medical_Keyword_3 + 0.5)**3.0
#Add the custom variables to the important variable dataframe
train <- cbind(train, custom_var_1, custom_var_3, custom_var_4, custom_var_6, custom_var_7, custom_var_8, custom_var_9, custom_var_10, custom_var_11, custom_var_12)

#Same features as above
custom_var_1 <- as.numeric(test$Medical_History_15 < 10.0)
custom_var_3 <- as.numeric(test$Product_Info_4 < 0.075)
custom_var_4 <- as.numeric(test$Product_Info_4 == 1)
custom_var_1[is.na(custom_var_1)] <- 0.0
custom_var_6 <- (test$BMI + 1.0)**2.0
custom_var_7 <- (test$BMI)**0.8
custom_var_8 <- test$Ins_Age**8.5
custom_var_9 <- (test$BMI*test$Ins_Age)**2.5
custom_var_10 <- as.numeric(test$BMI > BMI_cutoff)
custom_var_11 <- (test$BMI*test$Product_Info_4)**0.9
custom_var_12 <- as.numeric(test$Ins_Age*test$BMI > ageBMI_cutoff)	
#custom_var_13 <- (test$BMI*test$Medical_Keyword_3 + 0.5)**3.0
#Make important variable dataframe for test as well
test <- cbind(test, custom_var_1, custom_var_3, custom_var_4, custom_var_6, custom_var_7, custom_var_8, custom_var_9, custom_var_10, custom_var_11, custom_var_12)



feature.names <- names(train)[2:(ncol(train)-1)]


for(i in 1:ncol(train)){
  train[is.na(train[,i]), 'Employment_Info_1'] <- 0
}
for(i in 1:ncol(test)){
  test[is.na(test[,i]), 'Employment_Info_1'] <- 0
}

for(i in 1:ncol(train)){
  train[is.na(train[,i]), i] <- -99999
}
for(i in 1:ncol(test)){
  test[is.na(test[,i]), i] <- -99999
}

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

train = data.frame(train)
test = data.frame(test)


logregobj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  preds <- 1/(1 + exp(-preds))
  grad <- (preds - labels)
  hess <- preds * (1 - preds)
  return(list(grad = grad, hess = hess))
}

#https://www.kaggle.com/scirpus/prudential-life-insurance-assessment/xgb-test
kapparegobj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  x = (preds - labels)
  grad = 2*x*exp(-(x**2))*(exp(x**2)+x**2+1)
  hess = 2*exp(-(x**2))*(exp(x**2)-2*(x**4)+5*(x**2)-1)
  return(list(grad = grad, hess = hess))
}




evalkappa <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- ScoreQuadraticWeightedKappa(as.numeric(labels),as.numeric(round(preds)))
  return(list(metric = "kappa", value = err))
}

SQWKfun = function(x = seq(1.5, 7.5, by = 1), pred) { 
  preds = pred$predict 
  true = pred$Response 
  cuts = c(min(preds), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(preds)) 
  preds = as.numeric(Hmisc::cut2(preds, cuts)) 
  err = Metrics::ScoreQuadraticWeightedKappa(preds, true, 1, 8) 
  return(-err) }




X = data.table(train)
X.test = data.table(test)


# extract id
id.test <- X.test$Id
id.train <- X$Id
X.test$Id <- NULL
X$Id <- NULL
n <- nrow(X)

# extarct target
train_y <- X$Response

#train_y[train_y==3] <- 2
#train_y[train_y==4] <- 2

X = subset(X,select = -Response)

feature.names <- names(X)

train_x <- as.matrix(X)
test_x <- as.matrix(X.test)

# train & tune --skipped--
# 0.03, 2000, 10, 0, 3, .65, .6 

logfile <- data.frame(eta=        c(0.05, 0.02,.017, 0.01),
                      depth =     c(   8,   15, 17, 16),
                      alpha =    c(0.0, 0, 0.0, 0),
                      min.child = c(   3, 145, 145, 150),#### try 1,3,6,9
                      colsample.bytree = c(.9,.85,.9,.92),
                      subsample = c(.8,.85,.85,.9)#,
                      #maxdeltastep = c(2,4,10,20)
                      )

runs = data.frame(Id = id.test)
valsetpredavg   = rep(0,9000)
trainsetpredavg = rep(0,nrow(X))
testsetpredavg  = rep(0,nrow(X.test))

inTrain = 9001:length(train_y) ## /!\ change the seeds

valsubmit = data.frame(Id = id.train[-inTrain])
testsubmit = data.frame(Id = id.test)
trainsubmit = data.frame(Id = id.train)



# generate final prediction -- bag of 50 models --
models <- 1
repeats <- 1
yhat.test  <- rep(0,nrow(X.test))
l=1
for (i in 1:models){
    for (j in 1:repeats) {

    #set.seed(12)

    set.seed(j*1000 + i*100)   ############## fix seed 
    
    param <- list(#num_class = 9,
                  eta = logfile$eta[i],
                  max.depth = logfile$depth[i], 
                  min.child.weight= logfile$min.child[i],
                  colsample_bytree=logfile$colsample.bytree[i],
                  subsample=logfile$subsample[i],
                  #gamma=logfile$gamma[i], 
                  #num_class=9,
                  objective = kapparegobj

                  )
    
    # Set xgboost test and training and validation datasets
    xgtest <- xgb.DMatrix(data = test_x)
    xgtrain <- xgb.DMatrix(data = train_x[inTrain,], label= train_y[inTrain])
    xgval <-  xgb.DMatrix(data = train_x[-inTrain,], label= train_y[-inTrain])
    
    # setup watchlist to enable train and validation, validation must be first for early stopping
    watchlist <- list(val=xgval, train=xgtrain)
    # to train with watchlist, use xgb.train, which contains more advanced features
    
    #this will use default evaluation metric = rmse which we want to minimise
    bstval <- xgb.train(params = param, 
                      data = xgtrain, 
                      nround=7000,
                      print.every.n = 50, 
                      nthread = 30, 
                      watchlist=watchlist, 
                      early.stop.round = 300 ,
                      eval_metric="rmse",
                      maximize = FALSE,
                      missing=-999999)
    
    
    xgtrainfull <- xgb.DMatrix(data = train_x, label= train_y)
    
    
        bstcv <- xgb.cv(params = param, 
                          data = xgtrainfull, 
                          nfold = 3,
                          nround=bstval$bestInd+100,
                          print.every.n = 10, 
                          nthread = 30, 
                          feval=evalkappa,
                          maximize = TRUE,
                          missing=-999999) 
    
    
    if (staking==TRUE){
    
    # for purpose of staking
    
    split = floor(length(train_y)/2)
    xgtrainst_stack1 <- xgb.DMatrix(data = train_x[1:split,], label= train_y[1:split])
        
    bstfin1 <- xgb.train(params = param, 
                      data = xgtrainst_stack1, 
                      nround=bstval$bestInd+200,
                      print.every.n = 50, 
                      nthread = 30,
                      eval_metric="rmse",
                      maximize = FALSE,
                      missing=-999999)
    
    trainsetpred_s2 = predict(bstfin1,xgtrainst_stack2,missing=-999999,ntreelimit=bstval$bestInd)
    
    xgtrainst_stack2 <- xgb.DMatrix(data = train_x[(split+1):length(train_y),], label= train_y[(split+1):length(train_y)])
    
    bstfin2 <- xgb.train(params = param, 
                         data = xgtrainst_stack2, 
                         nround=bstval$bestInd+200,
                         print.every.n = 50, 
                         nthread = 30,
                         eval_metrics="rmse",
                         maximize = FALSE,
                         missing=-999999)
    
    trainsetpred_s1 = predict(bstfin2,xgtrainst_stack1,missing=-999999,ntreelimit=bstval$bestInd)
    
    
    
    
    
    cat("\n",paste0("run #",l),"\n")
    l=l+1
    
    
    valsetpred = predict(bstval,xgval,missing=-999999,ntreelimit=bstval$bestInd)
    
    xgtrainst_full <- xgb.DMatrix(data = train_x, label= train_y)
    
    bstfinfull <- xgb.train(params = param, 
                            data = xgtrainst_full, 
                            nround=bstval$bestInd+300,
                            print.every.n = 50, 
                            nthread = 30,
                            eval_metrics="rmse",
                            maximize = FALSE,
                            missing=-999999)
    
    

    testsetpred= predict(bstfinfull,xgtest,missing=-999999,ntreelimit=bstval$bestInd)
    
    
    } else {
      
      
      #normal algo
      
      xgtrainst_full <- xgb.DMatrix(data = train_x, label= train_y)
      
      bstfinfull <- xgb.train(params = param, 
                           data = xgtrainst_full, 
                           nround=bstval$bestInd+300,
                           print.every.n = 50, 
                           nthread = 30,
                           eval_metrics="rmse",
                           maximize = FALSE,
                           missing=-999999)
      
      
      
      
      
      
      cat("\n",paste0("run #",l),"\n")
      l=l+1
      
      
      valsetpred = predict(bstval,xgval,missing=-999999,ntreelimit=bstval$bestInd)
      
      trainsetpred = predict(bstfinfull,xgtrainst_full,missing=-999999,ntreelimit=bstval$bestInd)

      testsetpred= predict(bstfinfull,xgtest,missing=-999999,ntreelimit=bstval$bestInd)      
      
      
      valsubmit[paste0("c",l)] = valsetpred
      trainsubmit[paste0("c",l)] = trainsetpred
      testsubmit[paste0("c",l)] = testsetpred
      
      valsetpredavg   = valsetpredavg+valsetpred
      trainsetpredavg = trainsetpredavg+trainsetpred
      testsetpredavg  = testsetpredavg+testsetpred
      
      
      responseval = train_y[-inTrain]
      errorrmse = sum((valsetpred-responseval)^2)/length(responseval)
      errorrmse
      
      kappas = ScoreQuadraticWeightedKappa(as.numeric(responseval),round(valsetpred))
      kappas
      cat("kappa ",kappas,"  rmse ",errorrmse)
      
    }
    
    
    #runs[paste0("s",l)]<- probs1a
   }
}

responseval = train_y[-inTrain]
errorrmse = sum((valsetpred-responseval)^2)/length(responseval)
errorrmse

kappas = ScoreQuadraticWeightedKappa(as.numeric(responseval),round(valsetpred))
kappas



testresp = testsetpred
trainresp = trainsetpred

predtrain = data.frame(Id=id.train, Response=train$Response, predict=trainresp) 
optCuts = optim(seq(1.5, 7.5, by = 1), SQWKfun, pred = predtrain) 
print(optCuts) 

testresp2 = as.numeric(Hmisc::cut2(testresp, c(-Inf, optCuts$par, Inf))) 
print(table(testresp2))

submission = data.frame(Id = id.test)
submission$Response <- as.integer(testresp2)

write.table(submission, file = 'xgboost_05002- stack2 poisson 026.csv', row.names = F, col.names = T, sep = ",", quote = F)




write.table(valsubmit, file = 'xgboost_05-curious 20 runs poisson feat 4 val 001 .csv', row.names = F, col.names = T, sep = ",", quote = F)
write.table(testsubmit, file = 'xgboost_05-curious 20 runs poisson feat 4 test 001.csv', row.names = F, col.names = T, sep = ",", quote = F)
write.table(trainsubmit, file = 'xgboost_05-curious 20 runs poisson feat 4 train 001.csv', row.names = F, col.names = T, sep = ",", quote = F)

predtrain = data.frame(Id=train_y[-inTrain], Response=train_y[-inTrain], predict=valsetpred) 
optCuts = optim(seq(1.5, 7.5, by = 1), SQWKfun, pred = predtrain) 
print(optCuts) 
testresp2 = as.numeric(Hmisc::cut2(testresp, c(-Inf, optCuts$par, Inf))) 
print(table(testresp2))
submission = data.frame(Id = id.test)
submission$Response <- as.integer(testresp2)

write.table(submission, file = 'xgboost_05002- stack2 poisson 025.csv', row.names = F, col.names = T, sep = ",", quote = F)




#toto = predict(bst1, test_x,missing="NAN") 

# yhat.test <-  yhat.test/(models*repeats)

submission = data.frame(Id = id.test)
submission$Response <- as.integer(testresp2)

write.table(submission, file = 'xgboost_05002- stack2 poisson 004.csv', row.names = F, col.names = T, sep = ",", quote = F)


#submission[submission$Response<1, "Response"] <- 1
#submission[submission$Response>8, "Response"] <- 8
#submission[submission$Response==3,"Response"] <- 2


trainstack = train
trainstack$Response2 = trainsetpred
teststack = test
teststack$Response2 = testsetpred

write.table(trainstack, file = 'train poisson nlog stack01 001.csv', row.names = F, col.names = T, sep = ",", quote = F)
write.table(teststack, file = 'test poisson nlog stack01 003.csv', row.names = F, col.names = T, sep = ",", quote = F)






write.table(testresp2, file = 'xgboost_05-curious stack2 poisson.cs, row.names = F, col.names = T, sep = ",", quote = F)
