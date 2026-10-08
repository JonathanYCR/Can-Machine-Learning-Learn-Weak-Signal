# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 00:36:36 2026

@author: jonat
"""

import os
import random
import numpy as np
from scipy.io import loadmat

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss


# ==============================
# 参数
# ==============================
MAT_PATH = r"C:\Users\jonat\Desktop\论文\weaksignal\weaksignal\Macro1_construction\FredMD/FredMDlargeHor1.mat"

INITIAL_TRAIN = 179
STEP = 1
CV_FOLDS = 5
CS_GRID = np.logspace(-3, 1, 30)

SEED = 0


# ==============================
# 输出路径
# ==============================
def script_dir():
    if "__file__" in globals():
        return os.path.dirname(os.path.abspath(__file__))
    return os.getcwd()

OUT_PATH = os.path.join(script_dir(), "macro1_logistic_lasso_step1_new.txt")


# ==============================
# 工具函数
# ==============================
def safe_logloss(y_true, p_hat):
    eps = 1e-15
    p_hat = np.clip(p_hat, eps, 1 - eps)
    return log_loss(y_true, p_hat)


# ==============================
# 主程序
# ==============================
def main():

    print("=" * 80)
    print("Loading data...")

    data = loadmat(MAT_PATH)
    print("Keys in mat:", list(data.keys()))

    X = data["X"]
    X = np.delete(X, 5, axis=1)
    y_raw = data["Y"].ravel()

    # 二值化
    y = (y_raw >= 0).astype(int)

    n = len(y)

    print(f"Sample size n = {n}, p = {X.shape[1]}")
    print("=" * 80)

    np.random.seed(SEED)
    random.seed(SEED)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    y_true_all = []
    p_model_all = []
    p_null_all = []

    t = INITIAL_TRAIN

    while t < n:

        train_idx = np.arange(0, t)
        test_end = min(t + STEP, n)
        test_idx = np.arange(t, test_end)

        X_train = X[train_idx]
        y_train = y[train_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]

        print("-" * 60)
        print(f"Train: 0 ~ {t-1} (size={len(train_idx)})")
        print(f"Predicting: {t} ~ {test_end-1}")

        # rolling null probability
        p_null_t = np.mean(y_train)

        if len(np.unique(y_train)) < 2:
            print("WARNING: single class in training set")
            p_hat_block = np.full(len(test_idx), p_null_t)
        else:
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegressionCV(
                    Cs=CS_GRID,
                    cv=cv,
                    penalty="l1",
                    solver="saga",          # L1 必须用 saga 或 liblinear（CV+多类更建议 saga）
                    scoring="neg_log_loss",
                    max_iter=10000,         # saga 可能需要更大迭代
                    n_jobs=-1,
                    refit=True
                ))
            ])

            print("Fitting LASSO logistic (L1)...")
            clf.fit(X_train, y_train)

            p_hat_block = clf.predict_proba(X_test)[:, 1]
            bestC = clf.named_steps["lr"].C_[0]
            print("Best C:", bestC)

            # 可选：看稀疏度（非零系数数量）
            coef = clf.named_steps["lr"].coef_.ravel()
            nnz = int(np.sum(np.abs(coef) > 1e-12))
            print("Nonzero coefficients:", nnz)

        y_true_all.extend(y_test)
        p_model_all.extend(p_hat_block)
        p_null_all.extend(np.full(len(test_idx), p_null_t))

        t += STEP

    # ==============================
    # 统一评估
    # ==============================
    y_true_all = np.array(y_true_all)
    p_model_all = np.array(p_model_all)
    p_null_all = np.array(p_null_all)

    # Log-loss
    model_ll = safe_logloss(y_true_all, p_model_all)
    null_ll = safe_logloss(y_true_all, p_null_all)
    logloss_improve = (1 - model_ll / null_ll) * 100   # >0 better

    # Brier score
    brier_model = np.mean((y_true_all - p_model_all) ** 2)
    brier_null = np.mean((y_true_all - p_null_all) ** 2)
    brier_improve = (1 - brier_model / brier_null) * 100  # >0 better

    print("=" * 80)
    print("FINAL SUMMARY (LASSO logistic)")
    print("n_pred:", len(y_true_all))
    print("model_ll:", model_ll)
    print("null_ll :", null_ll)
    print("logloss_improve (null - model):", logloss_improve)
    print("brier_model:", brier_model)
    print("brier_null :", brier_null)
    print("brier_improve (null - model):", brier_improve)
    print("=" * 80)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(f"model_ll {model_ll}\n")
        f.write(f"null_ll {null_ll}\n")
        f.write(f"logloss_improve {logloss_improve}\n")
        f.write(f"brier_model {brier_model}\n")
        f.write(f"brier_null {brier_null}\n")
        f.write(f"brier_improve {brier_improve}\n")

    print("Saved to:", OUT_PATH)


if __name__ == "__main__":
    main()