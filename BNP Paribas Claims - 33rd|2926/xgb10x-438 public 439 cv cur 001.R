


library(data.table) #Faster reading
library(xgboost)

setwd("D:/NYC Data Science Academy/Kaggle/BNP")

# Start the clock!
start_time <- Sys.time()

na.roughfix2 <- function (object, ...) {
  res <- lapply(object, roughfix)
  structure(res, class = "data.frame", row.names = seq_len(nrow(object)))
}

roughfix <- function(x) {
  missing <- is.na(x)
  if (!any(missing)) return(x)
  
  if (is.numeric(x)) {
    x[missing] <- median.default(x[!missing])
  } else if (is.factor(x)) {
    freq <- table(x)
    x[missing] <- names(freq)[which.max(freq)]
  } else {
    stop("na.roughfix only works for numeric or factor")
  }
  x
}

# Set a seed for reproducibility
set.seed(2016)

cat("reading the train and test data\n")
# Read train and test
train_raw <- fread("train_feat_4_2e.csv", stringsAsFactors=TRUE)
print(dim(train_raw))
print(sapply(train_raw, class))

y <- train_raw$target
train_raw$target <- NULL
train_raw$ID <- NULL
n <- nrow(train_raw)

test_raw <- fread("test_feat_4_2e.csv", stringsAsFactors=TRUE)
test_id <- test_raw$ID
test_raw$ID <- NULL
print(dim(test_raw))
print(sapply(test_raw, class))
cat("Data read ")
print(difftime( Sys.time(), start_time, units = 'sec'))

# Preprocess data
# Find factor variables and translate to numeric
cat("Preprocess data\n")
all_data <- rbind(train_raw,test_raw)
all_data <- as.data.frame(all_data) # Convert data table to data frame

# NonCorrRemovals <- c("v76","v50","v51","v6","v70","v120","v69",
#                       "v102","v84","v121","v85","v119","v23","v2","v87",
#                       "v82","v5","v128","v59","v124","v25","v8","v2","v87",
#                       "v93","v28","v44","v4","v18","v90","v98","v99","v121","v84",
#                       "v131","v7","v27","v94","v95","v102")
# 
# all_data <- all_data[,which(names(all_data) %in% NonCorrRemovals)]



# Small feature addition - Count NA percentage
N <- ncol(all_data)
all_data$NACount_N <- rowSums(is.na(all_data)) / N 

feature.names <- names(all_data)

# make feature of counts of zeros factor
all_data$ZeroCount <- rowSums(all_data[,feature.names]== 0) / N
#all_data$Below0Count <- rowSums(all_data[,feature.names] < 0) / N
# 


# cat("Remove highly correlated features\n")
# # highCorrRemovals <- c("v50")
# highCorrRemovals <- c("v76","v50","v51","v6","v70","v120","v69",
#                       "v102","v84","v121","v85","v119","v23","v2","v87",
#                       "v82","v5","v128","v59","v124","v25","v8","v2","v87",
#                       "v93","v28","v44","v4","v18","v90","v98","v99","v121","v84",
#                       "v131","v7","v27","v94","v95","v102")
# 
# highCorrRemovals <- c("v25","v36","v37","v46",
#                       "v51","v53","v54","v63","v73","v81",
#                       "v82","v89","v92","v95","v105","v107",
#                       "v108","v109","v116","v117","v118",
#                       "v119","v123","v124","v128")
#  all_data <- all_data[,-which(names(all_data) %in% highCorrRemovals)]

cat("assuming text variables are categorical & replacing them with numeric ids\n")
cat("re-factor categorical vars & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(all_data[[f]])=="character" || class(all_data[[f]])=="factor") {
    all_data[[f]] <- as.integer(factor(all_data[[f]]))
  }
}

NonCorrRemovals <- c("v107","v71","v110","v31")

all_data <- all_data[,-which(names(all_data) %in% NonCorrRemovals)]



round_df <- function(df, digits) {
  nums <- vapply(df, is.numeric, FUN.VALUE = logical(1))
  
  df[,nums] <- round(df[,nums], digits = digits)
  
  (df)
}

all_data = round_df(all_data, digits=5)


train <- all_data[1:n,]
test <- all_data[(n+1):nrow(all_data),] 

print(dim(train))
#summary(train)tr
print(dim(test))
#summary(test)




#all_data <- na.roughfix2(all_data)

train <- all_data[1:n,]
test <- all_data[(n+1):nrow(all_data),] 

xgtrain = xgb.DMatrix(as.matrix(train), label = y, missing=NA)
xgtest = xgb.DMatrix(as.matrix(test), missing=NA)

# Do cross-validation with xgboost - xgb.cv
docv <- function(param0, iter) {
  model_cv = xgb.cv(
    params = param0
    , nrounds = iter
    , nfold = 5
    , data = xgtrain
    , early.stop.round = 50
    , print.every.n = 10
    , maximize = FALSE
    , nthread = 7
  )
  gc()
  best <- min(model_cv$test.logloss.mean)
  bestIter <- which(model_cv$test.logloss.mean==best)
  
  cat("\n",best, bestIter,"\n")
  print(model_cv[bestIter])
  
  bestIter-1
}

doTest <- function(param0, iter) {
  watchlist <- list('train' = xgtrain)
  model = xgb.train(
    nrounds = iter
    , params = param0
    , data = xgtrain
    , watchlist = watchlist
    , print.every.n = 20
    , nthread = 8
  )
  p <- predict(model, xgtest)
  rm(model)
  gc()
  p
}

param0 <- list(
  # some generic, non specific params
  "objective"  = "binary:logistic"
  , "eval_metric" = "logloss"
  , "eta" = 0.01
  , num_feature=10000
  , "subsample" = .9
  , "colsample_bytree" = .65
  , "min_child_weight" = 1
  , "max_depth" = 10
  , min_child_weight = 1
  , alpha = 1
  , lambda = 1
  , gamma = 1
)

#print(proc.time() - start_time)
print( difftime( Sys.time(), start_time, units = 'sec'))
cat("Training a XGBoost classifier with cross-validation\n")
set.seed(2018)
cv <- docv(param0, 15000) 
# Show the clock
print( difftime( Sys.time(), start_time, units = 'sec'))










#############################################################



# sample submission total analysis
submission <- read.csv("sample_submission.csv")
ensemble <- rep(0, nrow(submission))

cv <- round(720*1.2)
cat("Calculated rounds:", cv, " Starting ensemble\n")

# Bagging of single xgboost for ensembling
# change to e.g. 1:10 to get quite good results
for (i in 1:1) {
  print(i)
  set.seed(i + 2017)
  p <- doTest(param0, cv) 
  # use 40% to 50% more than the best iter rounds from your cross-fold number.
  # as you have another 50% training data now, which gives longer optimal training time
  ensemble <- ensemble + p
}

# Finalise prediction of the ensemble
cat("Making predictions\n")
submission$PredictedProb <- ensemble/i

# Prepare submission
write.csv(submission, "bnp-xgb-013.csv", row.names=F, quote=F)
summary(submission$PredictedProb)

# Stop the clock
#print(proc.time() - start_time)
print( difftime( Sys.time(), start_time, units = 'min'))
