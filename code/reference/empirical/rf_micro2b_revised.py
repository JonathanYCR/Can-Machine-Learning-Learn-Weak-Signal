from sklearn.metrics import mean_squared_error
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from scipy.io import loadmat
import random
from sklearn.model_selection import KFold
import argparse
parser  = argparse.ArgumentParser()
parser.add_argument('integer', metavar='N', type=int, help='an integer for the accumulator')
arg = parser.parse_args()
an=int(arg.integer)


def fit_models(num1, num2):
    np.random.seed(an)
    random.seed(an)    
    tf.random.set_seed(an)  
    
    filepath='weak/Micro2data/'
    data = loadmat(filepath+f"Micro2_{num2+1}_{num1+1}.mat")
    X_train = data['X_train']
    X_pred = data['X_pred']
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
    
    fixed_params = {
        'n_estimators': 500,
        'random_state': an,
        'n_jobs':-1
    }

    
    param_grid = {
        'max_depth': np.arange(1, 51, 5),
        'max_features':  np.arange(1, 4, 1),
        'max_samples':[0.5, 1.0]
    }

    rfr = RandomForestRegressor(**fixed_params,bootstrap=True)
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

    f=open("weak/micro2b_rf_result_revised.txt","a")
    f.write(f"{num2} {((y_pred- y_predb-y_predhat) ** 2).mean()} { ((y_pred - y_predb) ** 2).mean()} {best_params['max_depth']} {best_params['max_features']} {best_params['max_samples']}"+'\n')
    return 1

num1 = 0
num2 = 0
results = []
#num1=j num2=i
#an 1-100
for i in range(1, 101):
    if i==an:
        fit_models(num1,num2)
    if(num1 == 19):
      num2+=1
      num1 = 0
    else: 
      num1+=1

'''
file_path = 'micro2b_rf_result_revised.txt'
df = pd.read_csv(file_path, sep='\s+', usecols=[0, 1, 2], header=None)
df.columns = ['i', 'nn', 'mean']

final_temp = pd.DataFrame(np.zeros((1, 1)))
for i in range(5):  
    temp = df[df.iloc[:,0] == i].mean()[1:]
    temp[:-1] = 1 - temp[:-1]/temp[-1]
    final_temp.iloc[0] += temp[:-1].values/5
print(final_temp)

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