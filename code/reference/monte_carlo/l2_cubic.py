
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 18:50:54 2023

@author: Zhouyu Shen
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from keras.models import Sequential,load_model
from keras.layers import Dense
import random
from keras.optimizers import SGD 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras import regularizers
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from keras.optimizers import SGD
from keras.regularizers import l1,l2
from keras.initializers import VarianceScaling
import math
import argparse
parser  = argparse.ArgumentParser()
parser.add_argument('integer', metavar='N', type=int, help='an integer for the accumulator')
arg = parser.parse_args()
an=int(arg.integer)
import tensorflow as tf




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


def cubic_l2(i):
    np.random.seed(i)
    random.seed(i)
    
    x_train, y_train,x_test,y_test = generatedata(n, p,'cubic')
    
    def create_model(lambda_value, lr):
        model = Sequential()
        model.add(Dense(16, input_dim=p, kernel_regularizer=l2(lambda_value), kernel_initializer=VarianceScaling(scale=1), activation='relu'))
        model.add(Dense(1, kernel_regularizer=l2(lambda_value), use_bias=False, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=lr))
        return model


    # Define the grid search parameters
#    epochs = [50,100,200]
    lrs=[0.005,0.007,0.003,0.001,0.009]
    k = np.arange(-0.6, 0.2, 0.1)
    #lambda_values = [0.1,0.3,0.5,0.7]
    #lambda_values = [2.1]
    lambda_values = [10 ** i for i in k]
#    lambda_values= np.arange(0.5,0.7,0.02)
    mse_cv_min = 100000000
    best_lambda = -1
    best_epoch = -1
    best_lr = -1
    for lbd in lambda_values:
        for lr in lrs:
                epoch = int(0.5/lr)
                mse_cv = [] 
                for ii in range(10):
                    model = create_model(lbd, lr)
                    start = ii/10 * n
                    end = (ii+1)/10 * n
                    x_train1 = np.concatenate((x_train[0:int(start)], x_train[int(end):n]), axis=0)
                    y_train1 = np.concatenate((y_train[0:int(start)], y_train[int(end):n]), axis=0)
                    x_valid = x_train[int(start):int(end)]
                    y_valid = y_train[int(start):int(end)]
                    model.fit(x_train1, y_train1, epochs=epoch, batch_size=100, verbose=0)
                    y_pred = model.predict(x_valid)
                    mse=mean_squared_error(y_valid, y_pred)
                    mse_zero=np.mean(np.square(y_valid))
                    result=p/n*tau_n**(-2)*(mse-mse_zero)
                    mse_cv.append(result)
                mse_cv_mean = np.mean(mse_cv)
                if mse_cv_mean < mse_cv_min:
                    mse_cv_min = mse_cv_mean
                    best_lambda = lbd
                    best_epoch = epoch
                    best_lr = lr
    
    model = create_model(best_lambda, best_lr)
    model.fit(x_train, y_train, epochs=best_epoch, batch_size=100, verbose=0)
    
    model_dir = f'weaksimu/nn_model/cubic_l2'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, f'model_{i}.h5')
    model.save(model_path)
    model_file_path = os.path.join(model_dir, f'model_{i}.h5')
    model = load_model(model_file_path)
    os.remove(os.path.join(model_dir, f'model_{i}.h5'))
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_zero = mean_squared_error(y_test, np.zeros((n_valid, 1)))
    result = p/n * np.power(tau_n,-2) *(mse-mse_zero)


    f = open("weaksimu/nn_cubic_l2.txt", "a")
    f.write(str(i)+' '+ str(mse) + ' ' + str(mse_zero) + ' ' + str(result) + ' ' + str(best_lr) + ' ' + str(best_lambda) + '\n')
    f.close()
    

for i in range((an-1)*5+1,(an)*5+1):
    cubic_l2(i)
