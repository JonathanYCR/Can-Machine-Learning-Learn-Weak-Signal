import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.linalg import sqrtm, orth
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
import argparse
from sklearn.model_selection import KFold
from sklearn.utils import resample

# Command-line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('integer', metavar='N', type=int, help='an integer for the accumulator')
arg = parser.parse_args()
an = int(arg.integer)

n = 500
R2=0.05
tau_n = R2/(1-R2)
p = 300
sigmaep = 1
sigmabeta = 1
n_valid=10000
sparsity = 0.2

np.random.seed(1)
rho_1 = 0.5
Sigma_1 = np.array([[rho_1 ** abs(i - j) for j in range(n)] for i in range(n)])
sqrtSigma_1 = sqrtm(Sigma_1)

U2 = orth(np.random.rand(p, p))

eigen_Sigma_ep = 0.5 + np.random.rand(n)
sqrtSigma_ep = np.diag(np.sqrt(eigen_Sigma_ep))
eigen_Sigma_2 = 0.5 + np.random.rand(p)
Sigma_2 = U2 @ np.diag(eigen_Sigma_2) @ U2.T
sqrtSigma_2 = U2 @ np.diag(np.sqrt(eigen_Sigma_2)) @ U2.T

def generatedata(n, p, func):
    zero_loc = (np.random.uniform(size=p) > 1 - sparsity).astype(int)
    beta = np.sqrt(1/sparsity) * sigmabeta * np.random.randn(p) / np.sqrt(p * (1 / tau_n)) * zero_loc
    beta = beta.reshape(p, 1)
    ep = sqrtSigma_ep @ np.random.randn(n).reshape(n, 1) * sigmaep
    
    X = sqrtSigma_1 @ np.random.randn(n * p).reshape(n, p) @ sqrtSigma_2
    y = X @ beta + ep
    n_oos = n_valid
    ep_test = np.random.randn(n_oos, 1) * sigmaep
    X_test = np.random.randn(n_oos, p) @ sqrtSigma_2
    y_test = X_test @ beta + ep_test
    if func == 'tan':
        return np.arctan(X), y, np.arctan(X_test), y_test
    if func == 'cubic':
        return np.cbrt(X), y, np.cbrt(X_test), y_test
    if func == 'sinh':
        return np.arcsinh(X), y, np.arcsinh(X_test), y_test
    if func == 'linear':
        return X, y, X_test, y_test

def run_simulation(i):
    np.random.seed(i)
    
    x_train, y_train, x_test, y_test = generatedata(n, p, 'linear')
    

    fixed_params = {
        'n_estimators': 5000,
        'bootstrap': True,
        'random_state': i,
        'n_jobs': -1
    }


    param_grid = {
        'max_depth': [5,10,20],
        'max_features': [10,50,300],
        'max_samples':[0.1,0.2]
    }

    rfr = RandomForestRegressor(**fixed_params)
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
    best_params, best_score = manual_grid_search(x_train, np.ravel(y_train), param_grid)
    
    # Train the model with the best parameters
    rfr.set_params(**best_params)
    rfr.fit(x_train,  np.ravel(y_train))
    
    
    # Perform GridSearchCV

    x_test_sample, y_test_sample = resample(x_test, y_test, n_samples=2000, random_state=0)

    rf_importance = permutation_importance(rfr, x_test_sample, y_test_sample, n_repeats=20, random_state=0, n_jobs=-1)
    
    non_zero_count = np.count_nonzero(rf_importance.importances_mean)


    y_pred = rfr.predict(x_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mse_zero = mean_squared_error(y_test, np.zeros((n_valid, 1)))
    depth=best_params['max_depth']
    features=best_params['max_features']
    samples=best_params['max_samples']
    result = p * n**(-1) * tau_n**(-2) * (mse - mse_zero)
    output_string = f"{i} {mse} {mse_zero} {result} {depth} {features} {samples} {non_zero_count}\n"
    f = open( 'weaksimu/rf_linear.txt', "a")
    f.write(output_string)
    f.close()
    return 1
for i in range((an-1)*5+1,(an)*5+1):
    result = run_simulation(i)    

'''
file_path = 'rf_linear.txt'

df = pd.read_csv(file_path, sep='\s+',  header=None)
# Assuming df is already loaded with your data
col_1 = df.iloc[:, -2]  # Second last column
col_2 = df.iloc[:, -3]  # Last column
col_3= df.iloc[:, -4]
col_4=df.iloc[:,-5]
# Creating a figure with 2 subplots
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 5))

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
# Plotting histogram for last column
axes[3].hist(col_4, bins=20, color='green', alpha=0.7)
axes[3].set_title('Histogram of Last Column')
axes[3].set_xlabel('Values')
axes[3].set_ylabel('Frequency')
axes[3].axvline(x=0, color='r', linestyle='--')
axes[3].set_xlim([-3, 3])
# Display the plot
plt.tight_layout()
plt.show()
'''