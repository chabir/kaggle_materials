Santander competition: 128th/4591 - silver medal - august 2018
 
A. No Leak part:

several method were used: the final feature engineering that works best is to use top 45 xgb features and statistics such as mean, std, median on rows for all the remaining columns. Several Lightgbm and xgboost using CV 5 to 10 folds were used.


B. Leak part:
Inspired by many public kernels (leak_piece notebook), I found 90 sets of ordered columns among which 70 of 40 columns and 20 of >40 columns (the max numbers of possible sets seems to be 107) using diverse methods: manual, semi automatic using VBA macros (!) based on unique values and position of unique values, and a kernel Jaccard+graph


Note: I also tried pseudo labeling method using train + leaked test set and built some artificial training set based on the property of the train/test set with leak but it didn't give me any better result unfortunately.
