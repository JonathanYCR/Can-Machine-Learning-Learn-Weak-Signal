# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 09:43:13 2024

@author: Zhouyu Shen
"""

# Macro2_scale.txt

from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd 
import math
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
# use parallel processing to calculate the mse and r2
from joblib import Parallel, delayed
from scipy.io import loadmat
from scipy.linalg import pinv
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from scipy.io import loadmat
from scipy.linalg import pinv
from sklearn.linear_model import RidgeCV, LassoCV
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import random
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import random
import argparse
parser  = argparse.ArgumentParser()
parser.add_argument('integer', metavar='N', type=int, help='an integer for the accumulator')
arg = parser.parse_args()
an=int(arg.integer)



Macro2 = loadmat("weak/GrowthData.mat")
data = Macro2["data"]
data = np.column_stack((np.ones(data.shape[0]), data))

p = data.shape[1] - 1
n = data.shape[0]
y = data[:, p]
X = data[:, :p]
X = X[:, 1:]

def fit_models(X, y, index_train, index_pred):
    np.random.seed(an)
    random.seed(an)
    y_train = y[index_train]
    X_train = X[index_train, :]
    y_pred = y[index_pred]
    X_pred = X[index_pred, :]
    
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    y_mean = y_train.mean()
    X_train = (X_train - X_mean) / X_std
    X_pred = (X_pred - X_mean) / X_std
    
    # Define fixed parameters
    fixed_params = {
        'n_estimators': 500,
        'random_state': an,
        'n_jobs':-1
    }


    # Define parameters to search
    param_grid = {
        'max_depth': range(1,31,5),
        'max_features':[1,3,5,10,20,30,40,50,60],
        'max_samples':[0.5, 1.0]
    }

    # Define the model with fixed parameters
    rfr = RandomForestRegressor(**fixed_params, bootstrap=True)

    # Manually implement Cross-Validation
    def manual_grid_search(X_train, y_train, param_grid, n_splits=10):
        # Define the k-fold cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=an)
    
        # Initialize variables to store the best results
        best_score = float('inf')
        best_params = None
    
        # Iterate over all combinations of hyperparameters in the grid
        for max_depth in param_grid['max_depth']:
            for max_features in param_grid['max_features']:
                for max_samples in param_grid['max_samples']:
                    mse_scores = []
    
                    # Perform cross-validation
                    for train_index, val_index in kf.split(X_train):
                        X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
                        y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]
    
                        # Initialize the model with the current hyperparameters
                        rfr.set_params(max_depth=max_depth, max_features=max_features, max_samples=max_samples)
    
                        # Fit the model
                        rfr.fit(X_train_cv, y_train_cv)
    
                        # Make predictions
                        y_val_pred = rfr.predict(X_val_cv)
    
                        # Calculate mean squared error for validation set
                        mse = mean_squared_error(y_val_cv, y_val_pred)
                        mse_scores.append(mse)
    
                    # Calculate the average MSE for the current combination of hyperparameters
                    avg_mse = np.mean(mse_scores)
    
                    # Update the best parameters if the current MSE is lower
                    if avg_mse < best_score:
                        best_score = avg_mse
                        best_params = {
                            'max_depth': max_depth,
                            'max_features': max_features,
                            'max_samples': max_samples
                        }

        return best_params, best_score


    # Perform manual grid search
    best_params, best_score = manual_grid_search(X_train, y_train - y_mean, param_grid)
    
    # Train the model with the best parameters
    rfr.set_params(**best_params)
    rfr.fit(X_train, y_train - y_mean)
    
    
    # Predict on the test set
    y_predhat = rfr.predict(X_pred)

    f=open("weak/macro2_rf_result_revised.txt","a")
    f.write(f"{an} {((y_pred - y_mean-y_predhat) ** 2).mean()} { ((y_pred - y_mean) ** 2).mean()} {best_params['max_depth']} {best_params['max_features']} {best_params['max_samples']}"+'\n')
    return 1

#an:1-100
results = []
f = open("weak/Macro2_index.txt", "r")
lines = f.readlines()
lines1 = lines[an][:-1]
line = np.array(lines1.split(' '))
# find the index where the value is -1000
train_endindex = np.where(line == '-1000')[0][0]
train_index = line[2:train_endindex].astype(int) -1
test_endindex = np.where(line == '-2000')[0][0]
test_index = line[train_endindex+1:test_endindex].astype(int) -1
fit_models(X, y, train_index, test_index)





'''
file_path = 'macro2_rf_result_revised.txt'  
df = pd.read_csv(file_path, sep=' ', header=None)
column_means = df.mean()
print(column_means)
col_1 = df.iloc[:, 1]
col_2 = df.iloc[:, 2]

result = 1 - (col_1 / col_2)

mean_value = result.mean()

print(mean_value)


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(file_path, sep='\s+',  header=None)
# Assuming df is already loaded with your data
col_1 = df.iloc[:, -2]  # Second last column
col_2 = df.iloc[:, -1]  # Last column
col_3= df.iloc[:, -3]
# Creating a figure with 2 subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

# Plotting histogram for second last column
axes[0].hist(col_1, bins=20, color='blue', alpha=0.7)
axes[0].set_title('Histogram of Second Last Column')
axes[0].set_xlabel('Values')
axes[0].set_ylabel('Frequency')

# Plotting histogram for last column
axes[1].hist(col_2, bins=20, color='green', alpha=0.7)
axes[1].set_title('Histogram of Last Column')
axes[1].set_xlabel('Values')
axes[1].set_ylabel('Frequency')

# Plotting histogram for last column
axes[2].hist(col_3, bins=20, color='green', alpha=0.7)
axes[2].set_title('Histogram of Last Column')
axes[2].set_xlabel('Values')
axes[2].set_ylabel('Frequency')
# Display the plot
plt.tight_layout()
plt.show()
'''