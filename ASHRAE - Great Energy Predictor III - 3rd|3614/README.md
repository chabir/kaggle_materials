Hello!

A summary of the solution is available on the Kaggle forum [here](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/124984)

Below is an outline of how to reproduce my solution for the ASHRAE - Great Energy Predictor III competition.

(for all notebooks except catboost) . 
#HARDWARE & SOFTWARE :  
Cloud Computing Services: AWS . 
Instance Type: m5a.8xlarge (32 vCPU, 128MB) . 
AMI ID: Deep Learning AMI (Ubuntu) Version 18.0 (ami-07b18f799864a106a) . 
Python: Python 3.6.8 :: Anaconda, Inc. (python packages are detailed separately in `requirements.txt`):  

#DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed) . 
below are the shell commands used in each step, as run from the top level directory . 
kaggle competitions download -c ashrae-energy-prediction

#DATA PROCESSING . 
Generate clean data : 4-4.5 hours total (a single groupby takes 3.5 hours to run) . 
Run notebooks:  
1. generate_datasets.ipynb . 
2. generate_leak_data.ipynb . 

#TRAINING AND PREDICTIONS:  
Lightgbm: between 30 mins and 1 hour for training / 5 mins for prediction . 
Dense NN: 1 hour training / 3 min for prediction . 
CNN: 2.5h+ for training / 50min for prediction . 
Catboost: 1.5 hour training / 45 mins for prediction . 

1. Run notebooks:  
level1--submission_multimeter003.ipynb . 
level1--submission_multimeter004_nobuild.ipynb . 
level1--submission_nn001-DenseNN.ipynb . 
level1--submission_nn007lofo--CNN.ipynb . 
level1--submission_whatsyourcv3_0052_trncl.ipynb . 
level1--submission_withoutleak001.ipynb . 

2. Run Catboost notebook on a GPU instance and copy the prediction csv.gz file (here) in the main folder . 

3. Run notebook:
level2--ensembling_model.ipynb . 

(for Catboost):  
#HARDWARE & SOFTWARE :  
Cloud Computing Services: AWS  
Instance Type: p3.2xlarge (8 vCPU, 61MB) . 
AMI ID: Deep Learning AMI (Ubuntu 16.04) Version 26.0 (ami-0e30cdd8359d89531)   
Python: Python 3.6.6 :: Anaconda, Inc. (python packages are detailed separately in `requirements.txt`):  





