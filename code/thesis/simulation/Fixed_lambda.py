# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 00:48:35 2026

@author: jonat
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.linalg import toeplitz, cholesky
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def sigmoid(z):
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def rand_orthogonal(p: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(p, p))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def build_matrices(n: int, p: int, rho_1: float, seed: int = 1000):
    rng = default_rng(seed)

    Sigma_1 = toeplitz(rho_1 ** np.arange(n))
    sqrtSigma_1 = cholesky(Sigma_1, lower=True)

    U2 = rand_orthogonal(p, rng)
    eigen_Sigma_2 = 0.5 + rng.random(p)
    sqrtSigma_2 = U2 @ np.diag(np.sqrt(eigen_Sigma_2)) @ U2.T

    return sqrtSigma_1, sqrtSigma_2


def gen_beta(p: int, sparsity: float, tau_n: float, sigmabeta: float, rng: np.random.Generator):
    q = sparsity
    mask = (rng.random(p) < q).astype(float)
    beta = (np.sqrt(1.0 / q) * sigmabeta * rng.normal(size=p) * np.sqrt(tau_n / p)) * mask
    return beta.reshape(-1, 1)


def fit_logistic_ridge_paper(X, y, lam_n):
    """
    Objective (paper-style):
      (1/n)*logloss + (p*lam_n/n)*||beta||^2

    sklearn LogisticRegression uses:
      (1/n)*logloss + 0.5*(1/C)*||beta||^2

    Match:
      0.5*(1/C) = (p*lam_n/n)  ->  C = n/(2 p lam_n)
    """
    n, p = X.shape
    C = n / (2.0 * p * lam_n)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            penalty="l2",
            C=C,
            fit_intercept=False,
            solver="lbfgs",
            max_iter=6000
        ))
    ])
    model.fit(X, y.ravel())
    return model


def fit_logistic_lasso_paper(X, y, lam_n):
    """
    Paper logistic lasso:
      (1/n)*sum logloss + (lam_n/sqrt(n))*||beta||_1

    Multiply by n:
      sum logloss + (lam_n*sqrt(n))*||beta||_1

    sklearn (saga, L1) approximately:
      sum logloss + (1/C)*||beta||_1

    Match:
      1/C = lam_n*sqrt(n)  =>  C = 1/(lam_n*sqrt(n))
    """
    n, _p = X.shape
    C = 1.0 / (lam_n * np.sqrt(n))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            penalty="l1",
            C=C,
            fit_intercept=False,
            solver="saga",
            max_iter=20000,
            tol=1e-4
        ))
    ])
    model.fit(X, y.ravel())
    return model



def logloss_from_proba(p1, y):
    eps = 1e-12
    p1 = np.clip(p1, eps, 1 - eps)
    y = y.ravel()
    return -np.mean(y * np.log(p1) + (1 - y) * np.log(1 - p1))


def plot_and_save_boxplot(data, lambdas, title, xlabel, ylabel, filename):
    plt.figure()
    plt.boxplot(data, labels=[str(x) for x in lambdas])
    plt.axhline(0.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved: {filename}")


def print_quartiles(name, data, lambdas):
    print(f"\n{name} quartiles (Q1, Median, Q3):")
    for j, lam in enumerate(lambdas):
        q1 = np.percentile(data[:, j], 25)
        q2 = np.percentile(data[:, j], 50)
        q3 = np.percentile(data[:, j], 75)
        print(f"lambda={lam:<7g}  Q1={q1:.6f}  Median={q2:.6f}  Q3={q3:.6f}")


def main_logistic_experiment3(
    tau_n: float,
    sparsity: float,
    n: int,
    p: int,
    lambdas=(500,1000,1500,2000,2500,3000),
    REPS: int = 200,
    n_oos: int = 10000,
    rho_1: float = 0.5,
    sigmabeta: float = 1.0,
    seed_global: int = 1000,
    out_prefix: str = "exp3",
):
    sqrtSigma_1, sqrtSigma_2 = build_matrices(n, p, rho_1=rho_1, seed=seed_global)
    lambdas = np.array(list(lambdas), dtype=float)

    # Experiment 3 metric:
    # scaled = (p/(n*tau_n^2))*(logloss_oos - log2)
    scaled_ridge = np.zeros((REPS, len(lambdas)))
    scaled_lasso = np.zeros((REPS, len(lambdas)))

    for ii in range(1, REPS + 1):
        rng = default_rng(seed_global + ii)

        beta = gen_beta(p, sparsity=sparsity, tau_n=tau_n, sigmabeta=sigmabeta, rng=rng)

        # Train: has Sigma_1
        Z = rng.normal(size=(n, p))
        X = (sqrtSigma_1 @ Z) @ sqrtSigma_2

        # OOS: no Sigma_1
        Z_oos = rng.normal(size=(n_oos, p))
        X_oos = Z_oos @ sqrtSigma_2

        # Logistic responses
        pi = sigmoid(X @ beta).ravel()
        y = rng.binomial(1, pi).reshape(-1, 1)

        pi_oos = sigmoid(X_oos @ beta).ravel()
        y_oos = rng.binomial(1, pi_oos).reshape(-1, 1)

        for j, lam in enumerate(lambdas):
            # keep your tuning rate choice:
            lam_n = lam / tau_n

            # ridge
            model_r = fit_logistic_ridge_paper(X, y, lam_n=lam_n)
            p1_oos_r = model_r.predict_proba(X_oos)[:, 1]
            ll_oos_r = logloss_from_proba(p1_oos_r, y_oos)
            scaled_ridge[ii - 1, j] = (p / (n * (tau_n ** 2))) * (ll_oos_r - np.log(2))

            # lasso
            model_l = fit_logistic_lasso_paper(X, y, lam_n=lam_n)
            p1_oos_l = model_l.predict_proba(X_oos)[:, 1]
            ll_oos_l = logloss_from_proba(p1_oos_l, y_oos)
            scaled_lasso[ii - 1, j] = (p / (n * (tau_n ** 2))) * (ll_oos_l - np.log(2))

        if ii % 20 == 0:
            print(f"(tau={tau_n}, q={sparsity}) finished {ii}/{REPS}")

    # ---- Save plots ----
    base = f"{out_prefix}_tau{tau_n}_q{sparsity}_n{n}_p{p}_REPS{REPS}"

    plot_and_save_boxplot(
        scaled_ridge, lambdas,
        title=f"Logistic Ridge Scaled Excess (tau={tau_n}, q={sparsity}, n={n}, p={p})",
        xlabel="lambda (constant), with lambda_n = lambda / tau",
        ylabel=r"Scaled excess  $\tilde{\Delta} = \frac{p}{n\tau_n^2}\,(\mathrm{logloss}-\log 2)$",
        filename=f"{base}_RIDGE.png"
    )

    plot_and_save_boxplot(
        scaled_lasso, lambdas,
        title=f"Logistic Lasso Scaled Excess (tau={tau_n}, q={sparsity}, n={n}, p={p})",
        xlabel="lambda (constant), with lambda_n = lambda / tau",
        ylabel=r"Scaled excess  $\tilde{\Delta} = \frac{p}{n\tau_n^2}\,(\mathrm{logloss}-\log 2)$",
        filename=f"{base}_LASSO.png"
    )

    # ---- Print quartiles ----
    print_quartiles("RIDGE", scaled_ridge, lambdas)
    print_quartiles("LASSO", scaled_lasso, lambdas)

    return lambdas, scaled_ridge, scaled_lasso


if __name__ == "__main__":
    main_logistic_experiment3(tau_n=0.025, sparsity=0.2, n=2500, p=1500, REPS=200)
