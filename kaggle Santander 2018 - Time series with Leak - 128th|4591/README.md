Santander competition: 120th/4591 - silver medal - august 2018
 
the training set is composed of 4459 rows / 4993 cols and test set (49342rows , 4992cols). the dataset is fully numerical and no indication of the nature of the variable is provided. 
 
A. Tradition ML part:

several method were used: the final feature engineering that works best is to use top 45 xgb important features and statistics such as mean, std, median on rows for all the remaining columns. Several Lightgbm and xgboost using CV 5 to 10 folds were used and a straight average was calculated.


B. Leak part:

After one month of competition, it appeared that even if the data was randomized, 85% train set can be ordered by cols and rows and the target can be deduced from another row. The same can be done for 25% of the test data after some work.

Inspired by many public kernels compiled in the leak_piece notebook, I found 90 sets of ordered columns among which 70 of 40 columns and 20 of >40 columns covering about 3800 cols out of 4991 original columns (the max numbers of possible sets seems to be 107) using diverse methods: manual, semi automatic using VBA macros (!) based on unique values and position of unique values, and a kernel Jaccard+graph. After filtering the wrong sets, the script is able to find 7885 leaked values in the test set.


Note: I also tried pseudo labeling method using train + leaked test set and built some artificial training set based on the property of the train/test set with leak but it didn't give me any better result unfortunately.
