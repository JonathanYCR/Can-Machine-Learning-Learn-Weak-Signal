# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 00:02:38 2026

@author: jonat
"""

import os
import numpy as np
from scipy.linalg import toeplitz, cholesky
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# =========================
# 1) Utilities
# =========================
def rand_orthogonal(p: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(p, p))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q

def setup_design_matrices(n: int, p: int, rho: float = 0.5, seed: int = 1000):
    """
    Fixed across all reps, matching your R:
      set.seed(1000) done before foreach.
    """
    rng = np.random.default_rng(seed)

    # Sigma_1: Toeplitz AR(1): rho^{|i-j|}
    c = rho ** np.arange(n)
    Sigma1 = toeplitz(c)
    sqrtSigma1 = cholesky(Sigma1, lower=True)

    # Sigma_eps: diagonal U(0.5, 1.5)
    diag_eps = 0.5 + rng.random(n)
    sqrtSigma_eps = np.diag(np.sqrt(diag_eps))

    # Sigma_2: U diag(eigs) U'
    U = rand_orthogonal(p, rng)
    eigs = 0.5 + rng.random(p)
    sqrtSigma2 = U @ np.diag(np.sqrt(eigs)) @ U.T

    return sqrtSigma1, sqrtSigma_eps, sqrtSigma2

def generate_beta(p: int, tau: float, q: float, rng: np.random.Generator) -> np.ndarray:
    """
    Spike-and-slab:
      beta0 = sqrt(tau/p) * b0
      b0_j = 0 w.p. 1-q;  N(0, q^{-1}) w.p. q
    """
    zero_loc = (rng.random(p) < q).astype(float)
    slab = np.sqrt(1.0 / q) * rng.normal(size=p)
    beta = (slab * zero_loc) * np.sqrt(tau / p)
    return beta.reshape(-1, 1)

def coef_back_to_original_scale(pipe: Pipeline) -> np.ndarray:
    """
    Pipeline: StandardScaler -> (LassoCV or RidgeCV)
    Return coefficients on original X scale.
    """
    scaler = pipe.named_steps["scaler"]
    model = pipe.named_steps["model"]
    coef_scaled = model.coef_.reshape(-1, 1)
    # If X_scaled = (X - mean)/scale, then beta_original = beta_scaled / scale
    return coef_scaled / scaler.scale_.reshape(-1, 1)

# =========================
# 2) One replication
# =========================
def simulate_one_rep(
    n: int,
    p: int,
    R2: float,
    q: float,
    sqrtSigma1: np.ndarray,
    sqrtSigma_eps: np.ndarray,
    sqrtSigma2: np.ndarray,
    rep_seed: int,
    n_oos: int = 10000,
):
    rng = np.random.default_rng(rep_seed)
    tau = R2 / (1.0 - R2)

    beta = generate_beta(p, tau, q, rng)

    # Train
    Z = rng.normal(size=(n, p))
    X = (sqrtSigma1 @ Z) @ sqrtSigma2
    eps = (sqrtSigma_eps @ rng.normal(size=(n, 1)))
    y = (X @ beta + eps).ravel()

    # glmnet-like: standardize TRUE, intercept FALSE
    lasso = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", LassoCV(cv=10, fit_intercept=False, n_alphas=100, max_iter=8000, random_state=rep_seed)),
        ]
    )
    ridge = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", RidgeCV(alphas=np.logspace(-4, 4, 120), cv=10, fit_intercept=False)),
        ]
    )

    lasso.fit(X, y)
    ridge.fit(X, y)

    bhat_lasso = coef_back_to_original_scale(lasso)
    bhat_ridge = coef_back_to_original_scale(ridge)

    # Delta (eq 8)
    def Delta(bhat: np.ndarray) -> float:
        term1 = np.sum((sqrtSigma2 @ (bhat - beta)) ** 2)
        term0 = np.sum((sqrtSigma2 @ beta) ** 2)
        return (p / n) * (tau ** -2) * (term1 - term0)

    DeltaLasso = float(Delta(bhat_lasso))
    DeltaRidge = float(Delta(bhat_ridge))

    # OOS (matches your R: X_oos = N(0,1) %*% sqrtSigma_2; eps ~ N(0,1))
    Z_oos = rng.normal(size=(n_oos, p))
    X_oos = Z_oos @ sqrtSigma2
    y_oos = (X_oos @ beta + rng.normal(size=(n_oos, 1))).ravel()

    yhat_lasso = lasso.predict(X_oos)
    yhat_ridge = ridge.predict(X_oos)

    R2oos_lasso = 1.0 - np.mean((y_oos - yhat_lasso) ** 2) / np.mean(y_oos**2)
    R2oos_ridge = 1.0 - np.mean((y_oos - yhat_ridge) ** 2) / np.mean(y_oos**2)

    return R2oos_lasso, R2oos_ridge, DeltaLasso, DeltaRidge

# =========================
# 3) Main: run all settings + append to exam_main.txt
# =========================
def run_and_save(
    out_path: str = "exam_main.txt",
    REPS: int = 1000,
    n: int = 500,
    p: int = 300,
    seed_design: int = 1000,
):
    # Ensure directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Fixed design matrices (same as your R: set.seed(1000) before foreach)
    sqrtSigma1, sqrtSigma_eps, sqrtSigma2 = setup_design_matrices(n, p, rho=0.5, seed=seed_design)

    # Same calls as your R
    settings = [
        #(0.05, 0.05),
        (0.05, 0.2),
        (0.05, 0.8),
        #(0.5, 0.05),
        #(0.5, 0.2),
        #(0.5, 0.8),
    ]

    # Append mode, line-per-rep, 6 columns:
    # R2, sparsity(q), R2oos_lasso, R2oos_ridge, DeltaLasso, DeltaRidge
    with open(out_path, "a", encoding="utf-8") as f:
        for (R2, q) in settings:
            for ii in range(1, REPS + 1):
                rep_seed = seed_design + ii  # matches R: set.seed(1000+ii)
                R2oos_l, R2oos_r, dL, dR = simulate_one_rep(
                    n=n,
                    p=p,
                    R2=R2,
                    q=q,
                    sqrtSigma1=sqrtSigma1,
                    sqrtSigma_eps=sqrtSigma_eps,
                    sqrtSigma2=sqrtSigma2,
                    rep_seed=rep_seed,
                    n_oos=10000,
                )

                # Similar to R write.table(t(temp), append=TRUE)
                f.write(f"{R2} {q} {R2oos_l} {R2oos_r} {dL} {dR}\n")

                print(f"(R2={R2}, q={q}) finished {ii}/{REPS}")

    print(f"Done. Results appended to: {out_path}")

if __name__ == "__main__":
    run_and_save(out_path="exam_main.txt", REPS=100, n=500, p=300, seed_design=1000)
