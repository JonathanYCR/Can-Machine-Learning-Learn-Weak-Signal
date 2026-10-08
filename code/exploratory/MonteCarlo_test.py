# -*- coding: utf-8 -*-
"""
Reproduce Figure 5 + Table 1 from Shen & Xiu (2024) Monte Carlo (Section 3.1)
Port of the provided R code to Python.

Dependencies:
  numpy, scipy, scikit-learn, pandas, matplotlib, joblib

Install:
  pip install numpy scipy scikit-learn pandas matplotlib joblib
"""

from __future__ import annotations
import os
import math
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV


# -----------------------------
# Utilities: random orthogonal
# -----------------------------
def rand_ortho(p: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random orthonormal matrix U (p x p) using QR decomposition."""
    A = rng.standard_normal((p, p))
    Q, R = np.linalg.qr(A)
    # Make Q uniform over O(p): fix sign ambiguity via diag(R)
    d = np.sign(np.diag(R))
    d[d == 0] = 1.0
    Q = Q * d
    return Q


def make_cov_mats(n: int, p: int, rho_1: float, seed: int = 1000):
    """
    Build Sigma_1, sqrtSigma_1, sqrtSigma_ep, Sigma_2, sqrtSigma_2
    exactly like your R code (fixed across MC reps).
    """
    rng = np.random.default_rng(seed)

    # Sigma_1[i,j] = rho_1^(abs(i-j))
    idx = np.arange(n)
    Sigma_1 = rho_1 ** np.abs(idx[:, None] - idx[None, :])
    sqrtSigma_1 = sqrtm(Sigma_1).real  # should be PSD, take real part

    # sqrtSigma_ep: diag(sqrt(eigen_Sigma_ep)), eigen_Sigma_ep ~ U(0.5, 1.5)
    eigen_Sigma_ep = 0.5 + rng.random(n)
    sqrtSigma_ep = np.diag(np.sqrt(eigen_Sigma_ep))

    # Sigma_2: U diag(eigen) U^T, eigen ~ U(0.5, 1.5), U random orthogonal
    U2 = rand_ortho(p, rng)
    eigen_Sigma_2 = 0.5 + rng.random(p)
    Sigma_2 = U2 @ np.diag(eigen_Sigma_2) @ U2.T
    sqrtSigma_2 = U2 @ np.diag(np.sqrt(eigen_Sigma_2)) @ U2.T

    return sqrtSigma_1, sqrtSigma_ep, sqrtSigma_2


# -----------------------------
# Core single replication
# -----------------------------
def one_rep(
    ii: int,
    R2: float,
    sparsity: float,
    n: int,
    p: int,
    tau_n: float,
    sigmabeta: float,
    sigmaep: float,
    sqrtSigma_1: np.ndarray,
    sqrtSigma_ep: np.ndarray,
    sqrtSigma_2: np.ndarray,
    n_oos: int,
    cv_folds: int,
    alpha_grid_lasso: np.ndarray,
    alpha_grid_ridge: np.ndarray,
):
    """
    One Monte Carlo repetition; matches your R code structure.
    Returns: (R2, sparsity, R2oos_lasso, R2oos_ridge, DeltaLasso, DeltaRidge)
    """
    rng = np.random.default_rng(1000 + ii)

    # zero_loc = (runif(p) > 1 - sparsity) + 0
    zero_loc = (rng.random(p) > (1.0 - sparsity)).astype(float)

    # beta = sqrt(1/sparsity)*sigmabeta*rnorm(p)/sqrt(p*tau_n^(-1))*zero_loc
    # tau_n^(-1) = 1/tau_n, so sqrt(p * (1/tau_n)) = sqrt(p/tau_n)
    beta = (math.sqrt(1.0 / sparsity) * sigmabeta *
            rng.standard_normal(p) / math.sqrt(p / tau_n) *
            zero_loc).reshape(-1, 1)

    # ep = sqrtSigma_ep %*% rnorm(n, sd=sigmaep)
    ep = (sqrtSigma_ep @ (rng.standard_normal((n, 1)) * sigmaep))

    # X = sqrtSigma_1 %*% N(0,1)_{n x p} %*% sqrtSigma_2
    X = sqrtSigma_1 @ rng.standard_normal((n, p)) @ sqrtSigma_2
    y = X @ beta + ep
    y = y.ravel()

    # OOS:
    ep_oos = rng.standard_normal((n_oos, 1)) * sigmaep
    X_oos = rng.standard_normal((n_oos, p)) @ sqrtSigma_2
    y_oos = (X_oos @ beta + ep_oos).ravel()

    # --- CV fits ---
    # Match glmnet: intercept=FALSE, but standardize=TRUE by default
    lasso = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", LassoCV(
            alphas=alpha_grid_lasso,
            cv=cv_folds,
            fit_intercept=False,
            max_iter=20000,
            n_jobs=1,
            random_state=1000 + ii,
        ))
    ])
    lasso.fit(X, y)
    betahat_lasso = lasso.named_steps["model"].coef_.reshape(-1, 1)

    # RidgeCV: choose alpha minimizing CV MSE
    ridge = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", RidgeCV(
            alphas=alpha_grid_ridge,
            cv=cv_folds,
            fit_intercept=False,
            scoring="neg_mean_squared_error",
        ))
    ])
    ridge.fit(X, y)
    betahat_ridge = ridge.named_steps["model"].coef_.reshape(-1, 1)

    # --- Delta ---
    # Delta = p * n^(-1) * tau_n^(-2) * ( ||sqrtSigma2*(betahat-beta)||^2 - ||sqrtSigma2*beta||^2 )
    err_l = sqrtSigma_2 @ (betahat_lasso - beta)
    err_r = sqrtSigma_2 @ (betahat_ridge - beta)
    base = sqrtSigma_2 @ beta

    DeltaLasso = (p / n) * (tau_n ** -2) * (float((err_l.T @ err_l)) - float((base.T @ base)))
    DeltaRidge = (p / n) * (tau_n ** -2) * (float((err_r.T @ err_r)) - float((base.T @ base)))

    # --- R2oos ---
    pred_l = lasso.predict(X_oos)
    pred_r = ridge.predict(X_oos)
    R2oos_lasso = 1.0 - np.mean((y_oos - pred_l) ** 2) / np.mean(y_oos ** 2)
    R2oos_ridge = 1.0 - np.mean((y_oos - pred_r) ** 2) / np.mean(y_oos ** 2)

    return (R2, sparsity, R2oos_lasso, R2oos_ridge, DeltaLasso, DeltaRidge)


# -----------------------------
# Main driver (like your R main)
# -----------------------------
def main(
    R2: float,
    sparsity: float,
    n: int = 500,
    p: int = 300,
    mc_reps: int = 1000,
    n_oos: int = 10000,
    cv_folds: int = 10,
    n_jobs: int = -1,
    out_dir: str = "weaksimu_revised_py",
):
    os.makedirs(out_dir, exist_ok=True)

    tau_n = R2 / (1.0 - R2)
    sigmaep = 1.0
    sigmabeta = 1.0
    rho_1 = 0.5

    # Fixed matrices across reps (like your R code)
    sqrtSigma_1, sqrtSigma_ep, sqrtSigma_2 = make_cov_mats(n, p, rho_1, seed=1000)

    # Alpha grids (tune if you want closer numeric match to glmnet lambda path)
    # sklearn alphas are in the objective: (1/(2n))*||y-Xb||^2 + alpha*||b||_1
    alpha_grid_lasso = np.logspace(-4, 1, 120)  # broad
    alpha_grid_ridge = np.logspace(-4, 6, 160)  # ridge needs huge in weak signals

    rows = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(one_rep)(
            ii=ii,
            R2=R2,
            sparsity=sparsity,
            n=n,
            p=p,
            tau_n=tau_n,
            sigmabeta=sigmabeta,
            sigmaep=sigmaep,
            sqrtSigma_1=sqrtSigma_1,
            sqrtSigma_ep=sqrtSigma_ep,
            sqrtSigma_2=sqrtSigma_2,
            n_oos=n_oos,
            cv_folds=cv_folds,
            alpha_grid_lasso=alpha_grid_lasso,
            alpha_grid_ridge=alpha_grid_ridge,
        )
        for ii in range(1, mc_reps + 1)
    )

    df = pd.DataFrame(rows, columns=["R2", "q", "R2oos_lasso", "R2oos_ridge", "DeltaLasso", "DeltaRidge"])
    out_path = os.path.join(out_dir, "exam_main.csv")
    df.to_csv(out_path, index=False)
    return df


# -----------------------------
# Reproduce Figure 5 + Table 1
# -----------------------------
def make_figure_and_table(dfs: dict, out_dir: str = "weaksimu_revised_py"):
    """
    dfs: dict keyed by (R2,q) -> DataFrame from main()
    We plot the three setups shown in your screenshot:
      (R2=0.05,q=0.2), (R2=0.05,q=0.8), (R2=0.5,q=0.2)
    Top row: Ridge Delta histogram
    Bottom row: Lasso Delta histogram
    """
    os.makedirs(out_dir, exist_ok=True)

    setups = [(0.05, 0.2), (0.05, 0.8), (0.5, 0.2)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=False)

    for j, key in enumerate(setups):
        df = dfs[key]
        dr = df["DeltaRidge"].to_numpy()
        dl = df["DeltaLasso"].to_numpy()

        # Ridge (top)
        ax = axes[0, j]
        ax.hist(dr, bins=60, density=True)
        ax.axvline(0.0, linestyle="--")
        ax.set_title(rf"$R^2={int(key[0]*100)}\%,\, q={key[1]}$")
        ax.set_xlim(-2, 2)

        # Lasso (bottom)
        ax2 = axes[1, j]
        ax2.hist(dl, bins=60, density=True)
        ax2.axvline(0.0, linestyle="--")
        ax2.set_title(rf"$R^2={int(key[0]*100)}\%,\, q={key[1]}$")
        ax2.set_xlim(-2, 2)

    fig.suptitle("Simulation Results for Ridge and Lasso in Linear DGPs (Python)")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig_path = os.path.join(out_dir, "figure5_python.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    # Table 1 summary
    table_rows = []
    for (R2, q) in setups:
        df = dfs[(R2, q)]
        for model in ["Lasso", "Ridge"]:
            d = df["DeltaLasso"] if model == "Lasso" else df["DeltaRidge"]
            d = d.to_numpy()
            q1, q2, q3 = np.quantile(d, [0.25, 0.50, 0.75])
            n_zero = int(np.sum(np.isclose(d, 0.0, atol=1e-12)))
            table_rows.append({
                "q": q,
                "R2(%)": int(R2 * 100),
                "Model": model,
                "Q1": q1,
                "Q2": q2,
                "Q3": q3,
                "#Zero": n_zero
            })

    tab = pd.DataFrame(table_rows)
    tab_path = os.path.join(out_dir, "table1_python.csv")
    tab.to_csv(tab_path, index=False)

    return fig_path, tab_path, tab


if __name__ == "__main__":
    # Run the three setups shown in the screenshot:
    dfs = {}
    for (R2, q) in [(0.05, 0.2), (0.05, 0.8), (0.5, 0.2)]:
        print(f"Running: R2={R2}, q={q}")
        df = main(R2=R2, sparsity=q, n=500, p=300, mc_reps=1000, n_oos=10000, cv_folds=10, n_jobs=-1)
        dfs[(R2, q)] = df

    fig_path, tab_path, tab = make_figure_and_table(dfs)
    print("Saved:", fig_path)
    print("Saved:", tab_path)
    print(tab)
