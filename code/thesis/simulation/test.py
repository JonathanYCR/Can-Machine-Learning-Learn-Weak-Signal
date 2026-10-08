import os
from pathlib import Path
import numpy as np
from numpy.random import default_rng
from scipy.linalg import toeplitz, cholesky
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss


def sigmoid(z):
    # stable sigmoid
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def rand_orthogonal(p: int, rng: np.random.Generator) -> np.ndarray:
    """Random orthogonal matrix via QR; det(Q) > 0."""
    A = rng.normal(size=(p, p))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def build_covariances(n: int, p: int, rho_1: float, rng: np.random.Generator):
    # Sigma_1: AR(1) Toeplitz
    Sigma_1 = toeplitz(rho_1 ** np.arange(n))
    sqrtSigma_1 = cholesky(Sigma_1, lower=True)

    # Sigma_2: random orthogonal eigenvectors + eigenvalues in [0.5, 1.5]
    U2 = rand_orthogonal(p, rng)
    eigen_Sigma_2 = 0.5 + rng.random(p)
    sqrtSigma_2 = U2 @ np.diag(np.sqrt(eigen_Sigma_2)) @ U2.T

    return sqrtSigma_1, sqrtSigma_2


def fit_logistic_cv(X, y, penalty: str, seed: int):
    """
    Logistic Ridge / Logistic Lasso via CV.

    glmnet-like choice:
    - standardize features (no centering) + fit_intercept=False
    """
    scaler = StandardScaler(with_mean=False, with_std=True)

    # Cs is inverse regularization strength in sklearn: C = 1/lambda
    Cs = np.logspace(-4, 4, 30)

    if penalty == "l2":
        # lbfgs is stable for L2
        model = LogisticRegressionCV(
            Cs=Cs,
            cv=5,
            penalty="l2",
            solver="lbfgs",
            scoring="neg_log_loss",
            fit_intercept=False,
            max_iter=5000,
            n_jobs=-1,
            random_state=seed,
        )
    elif penalty == "l1":
        # saga supports L1
        model = LogisticRegressionCV(
            Cs=Cs,
            cv=5,
            penalty="l1",
            solver="saga",
            scoring="neg_log_loss",
            fit_intercept=False,
            max_iter=8000,
            n_jobs=-1,
            random_state=seed,
        )
    else:
        raise ValueError("penalty must be 'l1' or 'l2'")

    pipe = Pipeline([("scaler", scaler), ("model", model)])
    pipe.fit(X, y)

    # Recover coefficients on original X-scale
    scale_ = pipe.named_steps["scaler"].scale_.copy()
    scale_[scale_ == 0.0] = 1.0
    beta_scaled = pipe.named_steps["model"].coef_.reshape(-1)  # shape (p,)
    beta_hat = (beta_scaled / scale_).reshape(-1, 1)
    return beta_hat, pipe


def one_rep(ii, R2, sparsity, n, p, sqrtSigma_1, sqrtSigma_2):
    """
    One Monte Carlo repetition under logistic DGP:
      y | X ~ Bernoulli(sigmoid(X beta))
    """
    seed = 1000 + ii
    rng = default_rng(seed)

    # keep the same R2->tau mapping as in the linear code, as a "signal knob"
    # (in logistic it's not literal R^2, but it's a convenient weak-signal control)
    tau_n = R2 / (1.0 - R2)

    # Assumption 3 style sparsity mask
    mask = (rng.random(p) > 1.0 - sparsity).astype(float).reshape(-1, 1)

    # beta scaling: sqrt(1/q) * N(0,1) * sqrt(tau/p)
    beta = (np.sqrt(1.0 / sparsity) *
            rng.normal(size=(p, 1)) *
            np.sqrt(tau_n / p) *
            mask)

    # Design X = Sigma1^{1/2} Z Sigma2^{1/2}
    Z = rng.normal(size=(n, p))
    X = sqrtSigma_1 @ Z @ sqrtSigma_2

    # Logistic DGP
    eta = X @ beta
    pi = sigmoid(eta)
    y = rng.binomial(1, pi.reshape(-1)).astype(int)

    # OOS sample
    n_oos = 10_000
    Z_oos = rng.normal(size=(n_oos, p))
    X_oos = Z_oos @ sqrtSigma_2
    pi_oos = sigmoid(X_oos @ beta)
    y_oos = rng.binomial(1, pi_oos.reshape(-1)).astype(int)

    # Fit Logistic Lasso / Ridge
    betahat_lasso, pipe_lasso = fit_logistic_cv(X, y, penalty="l1", seed=seed)
    betahat_ridge, pipe_ridge = fit_logistic_cv(X, y, penalty="l2", seed=seed)
    
    C_opt_lasso =  pipe_lasso.named_steps["model"].C_[0]
    C_opt_ridge =  pipe_ridge.named_steps["model"].C_[0]
    
    print(C_opt_lasso, C_opt_ridge)

    # ---- Delta (same quadratic metric as linear risk, your thesis uses it as proxy) ----
    s2_beta = sqrtSigma_2 @ beta
    s2_diff_l = sqrtSigma_2 @ (betahat_lasso - beta)
    s2_diff_r = sqrtSigma_2 @ (betahat_ridge - beta)

    DeltaLasso = (p / n) * (tau_n ** -2) * (float(np.sum(s2_diff_l ** 2)) - float(np.sum(s2_beta ** 2)))
    DeltaRidge = (p / n) * (tau_n ** -2) * (float(np.sum(s2_diff_r ** 2)) - float(np.sum(s2_beta ** 2)))

    # ---- OOS log-loss + misclassification error ----
    proba_l = pipe_lasso.predict_proba(X_oos)[:, 1]
    proba_r = pipe_ridge.predict_proba(X_oos)[:, 1]

    # avoid log(0)
    eps = 1e-12
    proba_l = np.clip(proba_l, eps, 1 - eps)
    proba_r = np.clip(proba_r, eps, 1 - eps)

    logloss_l = 8 * (p / n) * (tau_n ** -2) * (log_loss(y_oos, proba_l) - np.log(2.0))
    logloss_r = 8 * (p / n) * (tau_n ** -2) * (log_loss(y_oos, proba_r) - np.log(2.0))

    yhat_l = (proba_l >= 0.5).astype(int)
    yhat_r = (proba_r >= 0.5).astype(int)

    mis_l = float(np.mean(yhat_l != y_oos)) 
    mis_r = float(np.mean(yhat_r != y_oos)) 

    print(R2, sparsity, logloss_l, logloss_r, mis_l, mis_r, DeltaLasso, DeltaRidge)

    return np.array([R2, sparsity, logloss_l, logloss_r, mis_l, mis_r, DeltaLasso, DeltaRidge], dtype=float)


def main_logistic(R2, sparsity, n, p, REPS=100, out_path="weaksimu_revised/exam_main_logistic.txt"):
    # Fix covariances with seed 1000 (match the R style)
    rng0 = default_rng(1000)
    rho_1 = 0.5
    sqrtSigma_1, sqrtSigma_2 = build_covariances(n, p, rho_1, rng0)

    #out_file = Path(out_path)
    #out_file.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for ii in range(1, REPS + 1):
        row = one_rep(ii, R2, sparsity, n, p, sqrtSigma_1, sqrtSigma_2)
        rows.append(row)

        # progress print (as you asked)
        tau_n = R2 / (1.0 - R2)
        print(f"(tau={tau_n:.6g}, q={sparsity}) finished {ii}/{REPS}")

        # append write each rep (like your R code)
        #with out_file.open("a", encoding="utf-8") as f:
        #    f.write(" ".join(f"{x:.10g}" for x in row) + "\n")

    return np.vstack(rows)


if __name__ == "__main__":
    # Same grid as your R code: R2 in {0.05, 0.5}, sparsity in {0.05, 0.2, 0.8}
    for R2 in [0.05, 0.5]:
        for q in [0.2, 0.8]:
            main_logistic(R2=R2, sparsity=q, n=500, p=300, REPS=300,
                          out_path="weaksimu_revised/exam_main_logistic.txt")
