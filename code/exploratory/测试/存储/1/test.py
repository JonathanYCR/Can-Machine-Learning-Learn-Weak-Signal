# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 16:07:00 2026

@author: jonat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import toeplitz, cholesky
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# -----------------------
# 1) Utilities: random orthogonal, covariances, etc.
# -----------------------
def rand_orthogonal(p: int, rng: np.random.Generator) -> np.ndarray:
    """Random orthogonal matrix via QR of N(0,1) matrix (det positive)."""
    A = rng.normal(size=(p, p))
    Q, R = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def setup_design_matrices(n: int, p: int, rho: float = 0.5, seed: int = 1000):
    """
    Matches paper/R code:
      Sigma_1(i,j) = rho^{|i-j|} (rho=1/2)
      Sigma_eps diagonal entries ~ U(0.5,1.5)
      Sigma_2 = U diag(eigs) U' with eigs ~ U(0.5,1.5), U random orthogonal
    Fixed once per (R2,q) experiment (same as your R code: set.seed(1000) outside foreach).
    """
    rng = np.random.default_rng(seed)

    # Sigma_1 (AR(1) / Toeplitz)
    c = rho ** np.arange(n)
    Sigma1 = toeplitz(c)
    sqrtSigma1 = cholesky(Sigma1, lower=True)  # Sigma1^{1/2}

    # Sigma_eps (diagonal heteroskedastic)
    diag_eps = 0.5 + rng.random(n)
    sqrtSigma_eps = np.diag(np.sqrt(diag_eps))  # Sigma_eps^{1/2}

    # Sigma_2 (random spectrum + random eigenvectors)
    U = rand_orthogonal(p, rng)
    eigs = 0.5 + rng.random(p)
    sqrtSigma2 = U @ np.diag(np.sqrt(eigs)) @ U.T  # Sigma_2^{1/2}

    return sqrtSigma1, sqrtSigma_eps, sqrtSigma2


# -----------------------
# 2) DGP and metrics
# -----------------------
def generate_beta(p: int, tau: float, q: float, rng: np.random.Generator) -> np.ndarray:
    """
    Spike-and-slab:
      b0 ~ (1-q) * delta_0 + q * N(0, q^{-1} * sigma_beta^2), sigma_beta^2 = 1
      beta0 = sqrt(tau/p) * b0
    Code implementation matches your R:
      zero_loc ~ Bernoulli(q)
      beta = sqrt(1/q)*N(0,1) * zero_loc * sqrt(tau/p)
    """
    zero_loc = (rng.random(p) < q).astype(float)  # indicator, P= q
    slab = np.sqrt(1.0 / q) * rng.normal(size=p)  # N(0, q^{-1})
    beta = (slab * zero_loc) * np.sqrt(tau / p)
    return beta.reshape(-1, 1)


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
    """
    One Monte Carlo replication:
      X = Sigma1^{1/2} Z Sigma2^{1/2}, Z~N(0,1)
      eps = Sigma_eps^{1/2} e, e~N(0,1)
      y = X beta + eps
    Then 10-fold CV ridge/lasso, compute Delta (eq 8) and R2_oos (optional).
    """
    rng = np.random.default_rng(rep_seed)

    tau = R2 / (1.0 - R2)  # calibration used in your R code

    beta = generate_beta(p, tau, q, rng)

    # Train data
    Z = rng.normal(size=(n, p))
    X = (sqrtSigma1 @ Z) @ sqrtSigma2
    eps = (sqrtSigma_eps @ rng.normal(size=(n, 1)))
    y = (X @ beta + eps).ravel()

    # Models: mimic glmnet standardize=TRUE, intercept=FALSE
    # - Standardize features
    # - fit_intercept=False (paper/code uses intercept=FALSE)
    lasso_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", LassoCV(cv=10, fit_intercept=False, n_alphas=100, max_iter=8000, random_state=rep_seed)),
        ]
    )

    ridge_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", RidgeCV(alphas=np.logspace(-4, 4, 120), cv=10, fit_intercept=False)),
        ]
    )

    lasso_pipe.fit(X, y)
    ridge_pipe.fit(X, y)

    # Extract coefficients on ORIGINAL X scale:
    # If X_scaled = (X - mean)/scale, and y = X_scaled*w, then coef_original = w/scale
    # (mean doesn't matter because fit_intercept=False and X mean ~ 0 in DGP)
    lasso_scaler = lasso_pipe.named_steps["scaler"]
    ridge_scaler = ridge_pipe.named_steps["scaler"]

    coef_lasso_scaled = lasso_pipe.named_steps["model"].coef_
    coef_ridge_scaled = ridge_pipe.named_steps["model"].coef_

    coef_lasso = (coef_lasso_scaled / lasso_scaler.scale_).reshape(-1, 1)
    coef_ridge = (coef_ridge_scaled / ridge_scaler.scale_).reshape(-1, 1)

    # Delta (Equation (8))
    def Delta(bhat: np.ndarray) -> float:
        term1 = np.sum((sqrtSigma2 @ (bhat - beta)) ** 2)
        term0 = np.sum((sqrtSigma2 @ beta) ** 2)
        return (p / n) * (tau ** -2) * (term1 - term0)

    delta_lasso = float(Delta(coef_lasso))
    delta_ridge = float(Delta(coef_ridge))

    # #Zero: all coefficients ~ 0  <=> Delta == 0 theoretically
    zero_lasso = int(np.all(np.abs(coef_lasso) < 1e-12))
    zero_ridge = int(np.all(np.abs(coef_ridge) < 1e-12))

    # Optional: out-of-sample R^2 (not required for Fig5/Table1 but your R code computed it)
    # Use i.i.d. test X_oos ~ N(0, Sigma2), eps ~ N(0,1)
    Z_oos = rng.normal(size=(n_oos, p))
    X_oos = Z_oos @ sqrtSigma2
    y_oos = (X_oos @ beta + rng.normal(size=(n_oos, 1))).ravel()

    yhat_lasso = lasso_pipe.predict(X_oos)
    yhat_ridge = ridge_pipe.predict(X_oos)
    R2oos_lasso = 1.0 - np.mean((y_oos - yhat_lasso) ** 2) / np.mean(y_oos**2)
    R2oos_ridge = 1.0 - np.mean((y_oos - yhat_ridge) ** 2) / np.mean(y_oos**2)

    return delta_lasso, delta_ridge, zero_lasso, zero_ridge, R2oos_lasso, R2oos_ridge


# -----------------------
# 3) Run experiments + make Figure 5 and Table 1
# -----------------------
def summarize_deltas(deltas: np.ndarray, zeros: int):
    q1, q2, q3 = np.quantile(deltas, [0.25, 0.5, 0.75])
    return float(q1), float(q2), float(q3), int(zeros)


def main(
    REPS: int = 1000,
    n: int = 500,
    p: int = 300,
    seed_design: int = 1000,
    out_prefix: str = "repro",
):
    # Figure 5 uses three setups in the paper:
    setups = [
        (0.05, 0.2),  # R^2=5%, q=0.2
        (0.05, 0.8),  # R^2=5%, q=0.8
        (0.50, 0.2),  # R^2=50%, q=0.2
    ]

    # Fixed design matrices (same across reps)
    sqrtSigma1, sqrtSigma_eps, sqrtSigma2 = setup_design_matrices(n, p, rho=0.5, seed=seed_design)

    # Storage
    all_results = {}  # (R2,q) -> dict with lasso/ridge deltas, zeros
    for (R2, q) in setups:
        deltas_lasso = []
        deltas_ridge = []
        zeros_lasso = 0
        zeros_ridge = 0

        for ii in range(1, REPS + 1):
            rep_seed = seed_design + ii  # matches your R: set.seed(1000+ii)
            dl, dr, zl, zr, _, _ = simulate_one_rep(
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
            deltas_lasso.append(dl)
            deltas_ridge.append(dr)
            zeros_lasso += zl
            zeros_ridge += zr

            if ii % max(1, REPS // 10) == 0:
                print(f"[{out_prefix}] (R2={R2}, q={q}) finished {ii}/{REPS}")

        all_results[(R2, q)] = {
            "lasso": np.array(deltas_lasso),
            "ridge": np.array(deltas_ridge),
            "zeros_lasso": zeros_lasso,
            "zeros_ridge": zeros_ridge,
        }

    # ---- Table 1 ----
    rows = []
    for (R2, q) in setups:
        dl = all_results[(R2, q)]["lasso"]
        dr = all_results[(R2, q)]["ridge"]
        zL = all_results[(R2, q)]["zeros_lasso"]
        zR = all_results[(R2, q)]["zeros_ridge"]

        L_q1, L_q2, L_q3, L_zero = summarize_deltas(dl, zL)
        R_q1, R_q2, R_q3, R_zero = summarize_deltas(dr, zR)

        rows.append(
            {
                "q": q,
                "R2(%)": int(R2 * 100),
                "Lasso Q1": L_q1,
                "Lasso Q2": L_q2,
                "Lasso Q3": L_q3,
                "Lasso #Zero": L_zero,
                "Ridge Q1": R_q1,
                "Ridge Q2": R_q2,
                "Ridge Q3": R_q3,
                "Ridge #Zero": R_zero,
            }
        )

    table = pd.DataFrame(rows)
    table_path = f"{out_prefix}_table1.csv"
    table.to_csv(table_path, index=False)
    print(f"Saved Table 1 to: {table_path}")
    print(table)

    # ---- Figure 5 (2 x 3 histograms) ----
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2), constrained_layout=True)

    for j, (R2, q) in enumerate(setups):
        # top row: Ridge
        dr = all_results[(R2, q)]["ridge"]
        ax = axes[0, j]
        ax.hist(dr, bins=200, density=True)
        ax.axvline(0.0, linestyle="--")          # red dashed baseline in paper
        ax.axvline(np.median(dr), linestyle="-") # blue solid ~ median (paper’s solid line)
        ax.set_title(rf"$R^2={int(R2*100)}\%,\, q={q}$")
        ax.set_xlim([-2, 2])
        ax.set_ylim([0, 2.0])

        # bottom row: Lasso
        dl = all_results[(R2, q)]["lasso"]
        ax = axes[1, j]
        ax.hist(dl, bins=200, density=True)
        ax.axvline(0.0, linestyle="--")
        ax.axvline(np.median(dl), linestyle="-")
        ax.set_title(rf"$R^2={int(R2*100)}\%,\, q={q}$")
        ax.set_xlim([-2, 2])
        ax.set_ylim([0, 2.0])

    axes[0, 0].set_ylabel("Density (Ridge)")
    axes[1, 0].set_ylabel("Density (Lasso)")

    fig.suptitle("Figure 5: Simulation Results for Ridge and Lasso in Linear DGPs", fontsize=12)

    fig_path = f"{out_prefix}_figure5.png"
    fig.savefig(fig_path, dpi=200)
    print(f"Saved Figure 5 to: {fig_path}")

    return table_path, fig_path


if __name__ == "__main__":
    # 先用 REPS=100 或 200 验证；想对齐论文就 REPS=1000
    main(REPS=1000, n=500, p=300, seed_design=1000, out_prefix="repro")
