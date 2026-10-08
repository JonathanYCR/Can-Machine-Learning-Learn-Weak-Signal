import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import xgboost as xgb 
from xgboost import XGBRegressor
import random
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
from sklearn.inspection import permutation_importance
import math
import argparse
parser  = argparse.ArgumentParser()
parser.add_argument('integer', metavar='N', type=int, help='an integer for the accumulator')
arg = parser.parse_args()
an=int(arg.integer)

import time

filepath='weaksimu/'

n = 500
R2=0.05
tau_n = R2/(1-R2)
p = 300
sigmaep = 1
sigmabeta = 1
n_valid=10000
sparsity=0.2

from scipy.linalg import sqrtm, eig, orth



np.random.seed(1)
random.seed(1)

rho_1 = 0.5
Sigma_1 = np.array([[rho_1 ** abs(i - j) for j in range(n)] for i in range(n)])
sqrtSigma_1 = sqrtm(Sigma_1)

# Create random orthogonal matrix U2
U2 = orth(np.random.rand(p, p))

# Create eigenvalues for Sigma_ep and Sigma_2, and then create the matrices
eigen_Sigma_ep = 0.5 + np.random.rand(n)
sqrtSigma_ep = np.diag(np.sqrt(eigen_Sigma_ep))
eigen_Sigma_2 = 0.5 + np.random.rand(p)
Sigma_2 = U2 @ np.diag(eigen_Sigma_2) @ U2.T
sqrtSigma_2 = U2 @ np.diag(np.sqrt(eigen_Sigma_2)) @ U2.T

def generatedata(n,p,func):
    zero_loc = (np.random.uniform(size=p) > 1-sparsity).astype(int)
    beta = np.sqrt(1/sparsity) * sigmabeta * np.random.randn(p) / np.sqrt(p * (1 / tau_n)) * zero_loc
    beta = beta.reshape(p, 1)
    ep = sqrtSigma_ep @ np.random.randn(n).reshape(n, 1) * sigmaep
    
    # Generate the design matrix X and the response variable y
    X = sqrtSigma_1 @ np.random.randn(n * p).reshape(n, p) @ sqrtSigma_2
    y = X @ beta + ep
    n_oos = n_valid
    ep_test = np.random.randn(n_oos, 1) * sigmaep
    X_test = np.random.randn(n_oos, p) @ sqrtSigma_2
    y_test = X_test @ beta + ep_test
    if func=='tan':
        return np.arctan(X),y,np.arctan(X_test),y_test
    if func=='cubic':
        return np.cbrt(X),y,np.cbrt(X_test),y_test
    if func=='sinh':
        return np.arcsinh(X), y , np.arcsinh(X_test), y_test
    if func=='linear':
        return X,y,X_test,y_test





def main(i):
    np.random.seed(i)
    random.seed(i)
    
    x_train, y_train,x_test,y_test = generatedata(n, p, 'linear')
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    
    
    param_grid = {
        'max_n_estimators':100,
        'max_depth': [1, 2, 3, 4, 5,6],
        'learning_rate': [0.001,0.01,0.1,0.2,0.5]
    }
    

    best_score = float("inf")
    best_params = None
    max_n_estimators = param_grid['max_n_estimators']
    
    for max_depth in param_grid['max_depth']:
        for learning_rate in param_grid['learning_rate']:
            params = {
                'objective': 'reg:squarederror',
                'reg_alpha': 0,
                'reg_lambda': 0,
                'booster': 'gbtree',
                'n_jobs': -1,
                'seed': i,
                'tree_method': 'hist',
                #'device': "cuda",
                'max_depth': max_depth,
                'learning_rate': learning_rate
            }
            

            cv_results = xgb.cv(params, dtrain, num_boost_round=max_n_estimators, nfold=10, metrics='rmse')
            
            min_rmse_index = np.argmin(cv_results['test-rmse-mean'])
            mean_rmse = cv_results['test-rmse-mean'][min_rmse_index]
            if mean_rmse < best_score:
                best_score = mean_rmse
                best_params = params.copy()
                best_params['n_estimators'] = min_rmse_index + 1  
    xgb_model = XGBRegressor(**best_params)
    
    xgb_model.fit(x_train, y_train)
    
    # model_dir = filepath+'/xgb_model/'
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)    
        
    # model_save_path= os.path.join(model_dir, f"xgb_model_{i}.json")
    # xgb_model.save_model(model_save_path)
    # xgb_model = xgb.XGBRegressor()
    # xgb_model.load_model(model_save_path)

    x_test_sample, y_test_sample = resample(x_test, y_test, n_samples=2000, random_state=0)

    xgb_importance = permutation_importance(xgb_model, x_test_sample, y_test_sample, n_repeats=20, random_state=0, n_jobs=-1)
    
    non_zero_count = np.count_nonzero(xgb_importance.importances_mean)

    y_pred = xgb_model.predict(x_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mse_zero = mean_squared_error(y_test, np.zeros((n_valid, 1)))
    result = p * n**(-1) * tau_n**(-2) * (mse - mse_zero)

    lr = best_params['learning_rate']
    md = best_params['max_depth']
    btn = best_params['n_estimators']
    output_string = f"{i} {mse} {mse_zero} {result} {lr} {md} {btn} {non_zero_count}\n"
    f = open(filepath+"xgb_linear.txt", "a")
    f.write(output_string)
    f.close()

for i in range((an-1)*10+1,(an)*10+1):
    main(i)






