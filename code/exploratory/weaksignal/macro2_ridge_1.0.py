# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 22:28:00 2026

@author: jonat
"""

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

# ==============================
# 固定参数
# ==============================
MAT_PATH = r"C:\Users\jonat\Desktop\论文\weaksignal\weaksignal\ecmadata\data\Macro2\GrowthData.mat"
OUTPUT_PATH = "macro2_growth_logistic_ridge_randomsplit_100reps.txt"

REPS = 100
TEST_SIZE = 0.5
RANDOM_SEED = 42
CV_FOLDS = 10
CS_GRID = np.logspace(-3, 1, 30)

np.random.seed(RANDOM_SEED)

# ==============================
# 读取数据（按你给的方式）
# ==============================
Macro2 = loadmat(MAT_PATH)
data = Macro2["data"]
data = np.column_stack((np.ones(data.shape[0]), data))

p = data.shape[1] - 1
n = data.shape[0]
y = data[:, p]
X = data[:, :p]
X = X[:, 1:]  # 去掉手动加的那列 1（不作为特征）

print(f"Sample size n: {n}")
print(f"Feature dimension p: {X.shape[1]}")

# ==============================
# y 转成 0/1（如果需要）
# ==============================
y_unique = np.unique(y[~np.isnan(y)])
if not np.array_equal(y_unique, [0, 1]) and not np.array_equal(y_unique, [0]) and not np.array_equal(y_unique, [1]):
    thr = np.nanmedian(y)
    y = (y > thr).astype(int)
    print(f"[Info] y is not binary. Converted to 0/1 by median threshold = {thr:.6g}")
else:
    y = y.astype(int)

print(f"Positive rate: {y.mean():.4f}")
print(X.shape)
print(y.shape)

# ==============================
# Random Split × 100
# ==============================
rows = []

for rep in range(REPS):
    # 50/50 stratified random split
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED + rep
    )
    train_idx, test_idx = next(splitter.split(X, y))

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # 如果训练集只有一个类别，跳过（理论上 StratifiedShuffleSplit 很少发生）
    if len(np.unique(y_train)) < 2:
        print(f"Rep {rep+1:03d}: skipped (only one class in training).")
        continue

    # ===== Baseline: null model =====
    pi0 = float(y_train.mean())
    eps = 1e-6
    pi0 = min(max(pi0, eps), 1 - eps)

    baseline_prob = np.full_like(y_test, pi0, dtype=float)
    baseline_ll = log_loss(y_test, baseline_prob, labels=[0, 1])

    baseline_pred = np.full_like(y_test, 1 if pi0 >= 0.5 else 0, dtype=int)
    baseline_acc = accuracy_score(y_test, baseline_pred)

    # ===== Logistic Ridge + inner CV =====
    inner_cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED + rep
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegressionCV(
            Cs=CS_GRID,
            cv=inner_cv,
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

    model_pred = (prob >= 0.5).astype(int)
    model_acc = accuracy_score(y_test, model_pred)

    best_C = float(np.ravel(model.named_steps["lr"].C_)[0])

    excess_vs_null = model_ll - baseline_ll
    acc_improve = model_acc - baseline_acc

    rows.append([
        rep + 1,
        model_ll,
        baseline_ll,
        excess_vs_null,
        model_acc,
        baseline_acc,
        acc_improve,
        pi0,
        best_C
    ])

    print(
        f"Rep {rep+1:03d}: "
        f"excess_vs_null={excess_vs_null:+.6f}, "
        f"acc_improve={acc_improve:+.4f}, "
        f"acc_model={model_acc:.4f}, acc_base={baseline_acc:.4f}, "
        f"pi0={pi0:.3f}, best_C={best_C:.4f}"
    )

if len(rows) == 0:
    raise RuntimeError("No valid repetitions were produced (unexpected).")

rows = np.array(rows, dtype=float)

# ==============================
# 保存结果
# ==============================
np.savetxt(
    OUTPUT_PATH,
    rows,
    fmt="%.8f",
    header="rep model_logloss baseline_logloss excess_vs_null "
           "model_acc baseline_acc acc_improve pi0 best_C"
)

print("\nFinished.")
print("Saved to:", OUTPUT_PATH)

print("\n=== Summary (across reps) ===")
print("Average excess_vs_null:", rows[:, 3].mean())
print("Median  excess_vs_null:", np.median(rows[:, 3]))
print("Share (excess<0):", np.mean(rows[:, 3] < 0))

print("Average acc_improve:", rows[:, 6].mean())
print("Median  acc_improve:", np.median(rows[:, 6]))
print("Share (acc_improve>0):", np.mean(rows[:, 6] > 0))
print("Share (acc_improve<0):", np.mean(rows[:, 6] < 0))