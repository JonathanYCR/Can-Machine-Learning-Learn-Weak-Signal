import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd
import math
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.io import loadmat
import random
from scipy.linalg import pinv
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from multiprocessing import Pool, cpu_count
import argparse
parser  = argparse.ArgumentParser()
parser.add_argument('integer', metavar='N', type=int, help='an integer for the accumulator')
arg = parser.parse_args()
an=int(arg.integer)

Macro1 = loadmat("weak/FredMDlargeHor1.mat")
X = Macro1["X"]
y = Macro1["Y"]
n = len(y)
p = X.shape[1]



def fit_models(i):
    filepath='weak/Macro1data/' 
    data = loadmat(filepath+f"Macro1_{an}.mat")
    X_train = data['X_train']
    X_pred = data['X_pred']
    X_train = np.delete(X_train, 5, axis=1)
    X_pred = np.delete(X_pred, 5, axis=1)
    y_train = data['y_train']
    y_pred = data['y_pred']
    W_train = data['W_train']
    W_pred = data['W_pred']

    gammahat = np.linalg.solve(W_train.T @ W_train, W_train.T @ y_train)
    y_predb= W_pred @ gammahat
    
    Mw=np.eye(W_train.shape[0]) - np.dot(np.dot(W_train, np.linalg.inv(np.dot(W_train.T, W_train))), W_train.T)
    X_pred=X_pred-W_pred@ np.linalg.inv(W_train.T @ W_train)@ W_train.T @ X_train
    X_train=Mw@X_train
    y_train=Mw@y_train
    
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_train = (X_train - X_mean) / X_std
    X_pred = (X_pred - X_mean) / X_std

    param_grid = {
        'max_depth': range(5,51,10),
        'max_features':range(5,101,10), 
        'max_samples':[0.5, 1.0]
    }
    
    rfr = RandomForestRegressor(n_estimators=500, random_state=an,bootstrap=True,n_jobs=-1)

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
    best_params, best_score = manual_grid_search(X_train, y_train, param_grid)
    
    # Train the model with the best parameters
    rfr.set_params(**best_params)
    rfr.fit(X_train, y_train)
    
    
    # Predict on the test set
    y_predhat = rfr.predict(X_pred)



    f=open("weak/macro1b_rf_result_revised.txt","a")
    f.write(f"{an} {((y_pred - y_predb-y_predhat) ** 2).mean()} { ((y_pred - y_predb) ** 2).mean()} {best_params['max_depth']} {best_params['max_features']} {best_params['max_samples']}"+'\n')
    return 1
#an:1-45
fit_models(an)



'''

file_path = 'macro1b_rf_result_revised.txt'  
df = pd.read_csv(file_path, sep=' ', header=None)
column_means = df.mean()
print(column_means)
col_1 = df.iloc[:, 1]
col_2 = df.iloc[:, 2]


print(1-col_1.mean()/col_2.mean())
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
