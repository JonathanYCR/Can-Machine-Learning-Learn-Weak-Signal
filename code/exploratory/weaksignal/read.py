# -*- coding: utf-8 -*-
"""
Logistic Ridge rolling prediction
"""

import numpy as np
from scipy.io import loadmat

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss


# =========================
# 如果你想改 rolling step，在这里改
# =========================
i = 0      # ← 在 Spyder 里直接改这个数字即可


# =========================
# Load data
# =========================
Macro1 = loadmat(r"C:\Users\jonat\Desktop\论文\weaksignal\weaksignal\Macro1_construction\FredMD/FredMDlargeHor1.mat") 
#Micro2 = loadmat(r'C:\Users\jonat\Desktop\论文\weaksignal\weaksignal\ecmadata\data\Micro2\Data1stStageEminentDomain_u80.mat')
#Micro1 = loadmat(r"C:\Users\jonat\Desktop\论文\weaksignal\weaksignal\ecmadata\data\Micro1\Abortion_data_u13.mat")
#Finance1 = loadmat(r"C:\Users\jonat\Desktop\论文\weaksignal\weaksignal\Finance1_construction\Goyal_monthly.mat")
X = Macro1["X"]
X = np.delete(X, 5, axis=1)
y = 100*Macro1["Y"].ravel()
n = len(y)
p = X.shape[1]

print(X.shape)
print((y>0).mean())
