from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import os
from sklearn.ensemble import RandomForestRegressor
import random 
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import pickle
import random

np.random.seed(1000)

import argparse
parser  = argparse.ArgumentParser()
parser.add_argument('integer', metavar='N', type=int, help='an integer for the accumulator')
arg = parser.parse_args()
an=int(arg.integer)

# pool_size = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
feature_names = pd.read_csv("data1957-2022/data.csv", nrows=0).columns.tolist()
RFm_df=pd.DataFrame(columns=feature_names)


def myfunc(year):
    np.random.seed(year*1000)
    random.seed(year*1000)
    data = pd.read_csv("data1957-2022/data.csv",delimiter=',')
    ret = pd.read_csv("data1957-2022/ret.csv",delimiter=',').values[:,0]
    date = pd.read_csv("data1957-2022/date.csv",delimiter=',').values[:,0]
    sic2_x = pd.read_csv("data1957-2022/sic2_x.csv",delimiter=',').values  
    rn = year
    train_len=18
    test_len=12
    oos_len=1 ### 30 Periods
    t00=19570300
      
      
    t0=19570000+oos_len*rn*10000
    t1=t0+train_len*10000
    t2=t1+test_len*10000
    t3=t2+oos_len*10000
    
    ind=(date<=t1)*(date>=t00)
    xtrain=data[ind].values
    ytrain=ret[ind]
    #wtrain=weight[ind]
    #trainper=per[ind]
    traindate=date[ind]
    sic2_xtrain=sic2_x[ind,:]
      
    ind=(date<=t2)*(date>=t1)
    xtest=data[ind].values
    ytest=ret[ind]
      #wtest=weight[ind]
      #testper=per[ind]
    testdate=date[ind]
    sic2_xtest=sic2_x[ind,:]
      
    ind=(date>=t2)*(date<=t3)
    xoos=data[ind].values
    yoos=ret[ind]
      
    del data
      
      #woos=weight[ind]
      #oosper=per[ind]
    oosdate=date[ind]
    sic2_xoos=sic2_x[ind,:]
      
      # w1=np.zeros(len(traindate))
      # u=np.unique(traindate)
      # for i in range(len(u)):
      #     ind=traindate==u[i]
      #     w1[ind]=1.0/np.sum(ind)
      # w1=w1/np.sum(w1)
      # wtrain0=w1+0.0
      # #wtrain0=np.ones(wtrain)/1.0/len(wtrain)
      # wtrain=wtrain/np.sum(wtrain)
      
      # w1=np.zeros(len(testdate))
      # u=np.unique(testdate)
      # for i in range(len(u)):
      #     ind=testdate==u[i]
      #     w1[ind]=1.0/np.sum(ind)
      # w1=w1/np.sum(w1)
      # wtest0=w1+0.0
      # #wtest0=np.ones(wtest)/1.0/len(wtest)
      # wtest=wtest/np.sum(wtest)
      
      # mtrain=np.sum(wtrain0*ytrain)
      # mtest=np.sum(wtest0*ytest)
      
    mtrain=np.mean(ytrain)
    mtest=np.mean(ytest)
  ### Times All Y_t ###
    ts=pd.read_csv('data1957-2022/tspredictors_1950.csv',delimiter=',')
    d=ts['date'].values
  #  yscale=np.array([0.33,0.33,1,50,10,50,5,50])

    n1=xtrain.shape[0]
    n2=xtrain.shape[1]
    ynum=range(1,13)
    ynum=[1,2,3,4,5,6,7,8]
    xtrain=np.hstack((xtrain,np.zeros((n1,n2*len(ynum))),sic2_xtrain))
    ad=np.unique(traindate)
    for i in range(len(ad)):
      ind=traindate==ad[i]
      weizhi=(np.arange(len(d)))[d==np.floor(ad[i]/100)]-1
      for j in range(len(ynum)):
          yt=ts.iloc[weizhi,ynum[j]].values[0]#*yscale[j]
          xtrain[ind,(n2*(j+1)):(n2*(j+2))]=xtrain[ind,0:n2]*yt

    n1=xtest.shape[0]
    n2=xtest.shape[1]
    ynum=range(1,13)
    ynum=[1,2,3,4,5,6,7,8]
    xtest=np.hstack((xtest,np.zeros((n1,n2*len(ynum))),sic2_xtest))
    ad=np.unique(testdate)
    for i in range(len(ad)):
      ind=testdate==ad[i]
      weizhi=(np.arange(len(d)))[d==np.floor(ad[i]/100)]-1
      for j in range(len(ynum)):
          yt=ts.iloc[weizhi,ynum[j]].values[0]#*yscale[j]
          xtest[ind,(n2*(j+1)):(n2*(j+2))]=xtest[ind,0:n2]*yt

    n1=xoos.shape[0]
    n2=xoos.shape[1]
    ynum=range(1,13)
    ynum=[1,2,3,4,5,6,7,8]
    xoos=np.hstack((xoos,np.zeros((n1,n2*len(ynum))),sic2_xoos))
    ad=np.unique(oosdate)
    for i in range(len(ad)):
      ind=oosdate==ad[i]
      weizhi=(np.arange(len(d)))[d==np.floor(ad[i]/100)]-1
      for j in range(len(ynum)):
          yt=ts.iloc[weizhi,ynum[j]].values[0]
          xoos[ind,(n2*(j+1)):(n2*(j+2))]=xoos[ind,0:n2]*yt
    

    del ret
    del date
    del sic2_x
    del ind
    del testdate
    del traindate
    del yt
    xmean=np.vstack((xtrain,xtest)).mean(axis=0)
    sd=np.vstack((xtrain,xtest)).std(axis=0)
    for i in range(xoos.shape[1]):
      s=sd[i]
      m=xmean[i]
      xtrain[:,i]=xtrain[:,i]-m
      xtest[:,i]=xtest[:,i]-m
      xoos[:,i]=xoos[:,i]-m
      if s>1e-4:
          xtrain[:,i]=xtrain[:,i]/s
          xtest[:,i]=xtest[:,i]/s
          xoos[:,i]=xoos[:,i]/s
    print('start train')
    
    xtrain=np.vstack((xtrain,xtest))
    ytrain=np.hstack((ytrain,ytest))
    del xtest
    del ytest
    ymean=ytrain.mean()
    ytrain=ytrain-ymean
        # Define the parameter grid
    

    # # Define fixed parameters
    fixed_params = {
        'n_estimators': 500,
        'random_state': an,
        'n_jobs':-1
    }


    # Define parameters to search
    param_grid = {
        'max_depth': [2,4,8,12],
        'max_features': [1,2,3,5],
        'max_samples':[0.5,1.0]
    }


    # Define the model with fixed parameters
    rfr = RandomForestRegressor(**fixed_params, bootstrap=True)

    
    kf = KFold(n_splits=2)
    # Initialize variables to store the best results
    best_score = float('inf')
    best_params = None

    # Iterate over all combinations of hyperparameters in the grid
    for max_depth in param_grid['max_depth']:
        for max_features in param_grid['max_features']:
            for max_samples in param_grid['max_samples']:
                mse_scores = []

                # Perform cross-validation
                for train_index, val_index in kf.split(xtrain):
                    X_train_cv, X_val_cv = xtrain[train_index], xtrain[val_index]
                    y_train_cv, y_val_cv = ytrain[train_index], ytrain[val_index]

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
                #print(f"max_depth: {max_depth}, max_features: {max_features}, max_samples: {max_samples} -> MSE: {avg_mse}")
                # Update the best parameters if the current MSE is lower
                if avg_mse < best_score:
                    best_score = avg_mse
                    best_params = {
                        'max_depth': max_depth,
                        'max_features': max_features,
                        'max_samples': max_samples
                    }
   

    # Train the model with the best parameters
    rfr.set_params(**best_params)
    rfr.fit(xtrain,ytrain)
    rf_pred = rfr.predict(xoos)+ymean
    
    
    del xtrain, ytrain
    r2_RFm=1-np.sum(np.power(rf_pred-yoos,2))/np.sum(np.power(yoos,2))
    vip_scores_RFm={}
    for i, feature in enumerate(feature_names):
      xoos_mod = xoos.copy()
      feature_index = feature_names.index(feature)
      xoos_mod[:, feature_index] = 0
      for j in range(1, 9):
          interaction_index = n2 * j + feature_index
          xoos_mod[:, interaction_index] = 0
      yhat_RFm_mod = rfr.predict(xoos_mod) +ymean
      r2_RFm_mod=1-np.sum(np.power(yhat_RFm_mod-yoos,2))/np.sum(np.power(yoos,2))
      vip_RFm=r2_RFm_mod-r2_RFm
      vip_scores_RFm[feature] = vip_RFm
      del xoos_mod,yhat_RFm_mod 
    RFm_df.loc[year] = vip_scores_RFm
    filename='weak/Finance2_VIP/RFm_df_revised.csv'
    if os.path.exists(filename):
      existing_df=pd.read_csv(filename, index_col=0)
      existing_df.loc[year]=RFm_df.loc[year]
      existing_df.to_csv(filename)
    else:
       RFm_df.loc[[year]].to_csv(filename)


    combined_data=np.column_stack((oosdate,rf_pred,yoos))
    filename1 = './weak/Finance2_rf_pred_revised.txt'
    with open(filename1, 'ab') as f:
        np.savetxt(f, combined_data, fmt='%.16f', delimiter=" ", header="oosdate, nn_pred, y_pred")

    opt_tuning = [best_params['max_depth'],best_params['max_features'], best_params['max_samples']]

    temp=[year,
        ((np.squeeze(np.asarray(yoos)) -ymean- rfr.predict(xoos)) ** 2).mean(),
        np.mean(np.power(yoos,2))]
    with open("weak/finance2_rf_result_revised.txt", "ab") as f:
        f.write(b"\n")
        np.savetxt(f,temp+opt_tuning,newline=" ")
    return(1)


# an from 0-34

myfunc(an)

'''
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Read the file
# Replace 'sep' with the appropriate separator if the file is not a CSV
df = pd.read_csv('finance2_rf_result.txt', sep=' ', header=None, usecols=[0, 1, 2, 3, 4])
column_means = df.mean()

# Print the means
print(column_means)
# Step 2: Select the last three columns
last_three_columns = df.iloc[:, -2:]

# Step 3: Plot histograms
for column in last_three_columns:
    plt.figure()
    df[column].hist(bins=40)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.show()
'''
