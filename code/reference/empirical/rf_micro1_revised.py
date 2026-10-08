# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 16:59:29 2025

@author: Zhouyu Shen
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 12:50:15 2024

@author: 42577
"""
import os
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np
from sklearn.model_selection import KFold
from scipy.io import loadmat
import argparse

# Command-line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('integer', metavar='N', type=int, help='an integer for the accumulator')
arg = parser.parse_args()
an = int(arg.integer)

# Load data
Micro1 = loadmat("weak/Abortion_data_u13.mat")
TDums = Micro1["TDums"]
Dxmurd = Micro1["Dxmurd"]
Zmurd = Micro1["Zmurd"]
Dymurd = Micro1["Dymurd"]
data = np.column_stack((np.ones(TDums.shape[0]), TDums, Dxmurd, Zmurd, Dymurd))

# Prepare data
n = data.shape[0]
p = data.shape[1] - 1
X = data[:, :p]
y = data[:, p]
X = X[:, 1:]


# Define the fit_models function
def fit_models(X, y, index_train, index_pred, num2):
    y_train = y[index_train]
    X_train = X[index_train, :]
    y_pred = y[index_pred]
    X_pred = X[index_pred, :]

    # Standardize data
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    y_mean = y_train.mean()
    X_train = (X_train - X_mean) / X_std
    X_pred = (X_pred - X_mean) / X_std


    # Define fixed parameters
    fixed_params = {
        'n_estimators': 500,
        'random_state': an,  # Replace `an` with a valid random seed, such as 42
        'n_jobs': -1  # Use all available CPU cores
    }
    
    # Define parameters to search
    param_grid = {
        'max_depth': np.arange(1, 21, 5),
        'max_features': np.arange(1, 21, 5),
        'max_samples': [0.005,0.01, 0.05,0.1,0.5]
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
    
    # Write results to file
    with open("weak/micro1_rf_result_revised.txt", "a") as f:
        # Calculate the residuals (mean squared error for predictions)
        mse_test = ((y_pred - y_mean - y_predhat) ** 2).mean()
        mse_true = ((y_pred - y_mean) ** 2).mean()
    
        # Save additional results (including best parameters)
        f.write(f"{num2} {mse_test} {mse_true} "
                f"{best_params['max_depth']} {best_params['max_features']} "
                f"{best_params['max_samples']}\n")

    return 1

# 1-104
with open("weak/Micro1_index.txt", "r") as f:
    lines = f.readlines()

num1 = num2 = 0
for i in range(1, len(lines)):
    lines1 = lines[i][:-1]
    line = np.array(lines1.split(' '))
    train_endindex = np.where(line == '-1000')[0][0]
    train_index = line[3:train_endindex].astype(int) - 1
    test_endindex = np.where(line == '-2000')[0][0]
    test_index = line[train_endindex + 1:test_endindex].astype(int) - 1
    
    if an == i:
        final = fit_models(X, y, train_index, test_index, num2)
    
    if num1 == 7:
        num2 += 1
        num1 = 0
    else:
        num1 += 1




'''
file_path = 'temp.txt'
df = pd.read_csv(file_path, sep='\s+', usecols=[0, 1, 2], header=None)
df.columns = ['i', 'nn', 'mean']

final_temp = pd.DataFrame(np.zeros((1, 1)))
for i in range(13):  
    temp = df[df.iloc[:,0] == i].mean()[1:]
    temp[:-1] = 1 - temp[:-1]/temp[-1]
    final_temp.iloc[0] += temp[:-1].values/13
print(final_temp)

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