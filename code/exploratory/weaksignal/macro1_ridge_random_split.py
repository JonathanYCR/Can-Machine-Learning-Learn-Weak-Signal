# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 19:30:08 2026

@author: jonat
"""

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

# ==============================
# 固定参数
# ==============================
MAT_PATH = r"C:\Users\jonat\Desktop\论文\weaksignal\weaksignal\Macro1_construction\FredMD/FredMDlargeHor1.mat"
OUTPUT_PATH = "macro1_realrecession_logistic_ridge_randomsplit.txt"

N_SPLITS = 100
TRAIN_RATIO = 0.5
RANDOM_SEED = 42
CV_FOLDS = 10

CS_GRID = np.logspace(-3, 1, 30)

np.random.seed(RANDOM_SEED)

# ==============================
# 读取数据
# ==============================
data = loadmat(MAT_PATH)
X = data["X"]
X = np.delete(X, 5, axis=1)
y_raw = data["Y"].ravel()

# recession 转 0/1
y = (y_raw > 0).astype(int)

n = len(y) - 24
X = X[:n]
y = y[:n]

p = X.shape[1]

print(f"Sample size: {n}")
print(f"Dimension p: {p}")
print(f"Recession rate: {y.mean():.4f}")

# ==============================
# 100 次 Random Split
# ==============================
rows = []

splitter = StratifiedShuffleSplit(
    n_splits=N_SPLITS,
    train_size=TRAIN_RATIO,
    random_state=RANDOM_SEED
)

split_id = 0

for train_idx, test_idx in splitter.split(X, y):

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # ======================
    # Baseline: null model
    # ======================
    pi0 = float(y_train.mean())
    eps = 1e-6
    pi0 = min(max(pi0, eps), 1 - eps)

    baseline_prob = np.full_like(y_test, pi0, dtype=float)
    baseline_ll = log_loss(y_test, baseline_prob, labels=[0, 1])

    # ======================
    # Logistic Ridge + CV
    # ======================
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegressionCV(
            Cs=CS_GRID,
            cv=cv,
            penalty="l2",
            solver="lbfgs",
            scoring="neg_log_loss",
            max_iter=5000,
            n_jobs=-1,
            refit=True
        ))
    ])

    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1]
    model_ll = log_loss(y_test, prob, labels=[0, 1])

    best_C = float(model.named_steps["lr"].C_[0])

    excess_vs_null = model_ll - baseline_ll

    rows.append([
        split_id,
        model_ll,
        baseline_ll,
        excess_vs_null,
        pi0,
        best_C
    ])

    print(f"Split {split_id}: "
          f"excess_vs_null={excess_vs_null:+.6f}, "
          f"pi0={pi0:.3f}, best_C={best_C:.4f}")

    split_id += 1

rows = np.array(rows, dtype=float)

# ==============================
# 保存结果
# ==============================
np.savetxt(
    OUTPUT_PATH,
    rows,
    fmt="%.8f",
    header="split_id model_logloss baseline_logloss excess_vs_null pi0 best_C"
)

print("\nFinished.")
print("Saved to:", OUTPUT_PATH)
print("Average excess_vs_null:", rows[:, 3].mean())
print("Std     excess_vs_null:", rows[:, 3].std())
print("Median  excess_vs_null:", np.median(rows[:, 3]))
print("Share (excess<0):", np.mean(rows[:, 3] < 0))