import os
import numpy as np
from scipy.linalg import toeplitz, cholesky
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# =========================
# 1) Utilities
# =========================
def sigmoid(z: np.ndarray) -> np.ndarray:
    # numerically stable sigmoid
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def rand_orthogonal(p: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(p, p))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def setup_design_matrices(n: int, p: int, rho: float = 0.5, seed: int = 1000):
    """
    Matches your linear-code design for X:
      Sigma_1(i,j) = rho^{|i-j|} (rho=1/2)
      Sigma_2 = U diag(eigs) U', eigs ~ U(0.5,1.5), U random orthogonal
    Fixed once per experiment (like set.seed(1000) outside foreach).
    """
    rng = np.random.default_rng(seed)

    # Sigma_1: Toeplitz AR(1): rho^{|i-j|}
    c = rho ** np.arange(n)
    Sigma1 = toeplitz(c)
    sqrtSigma1 = cholesky(Sigma1, lower=True)

    # Sigma_2: U diag(eigs) U'
    U = rand_orthogonal(p, rng)
    eigs = 0.5 + rng.random(p)
    sqrtSigma2 = U @ np.diag(np.sqrt(eigs)) @ U.T

    return sqrtSigma1, sqrtSigma2


def generate_beta_from_tau(p: int, tau: float, q: float, rng: np.random.Generator) -> np.ndarray:
    """
    Spike-and-slab with tau controlling Var(eta) scale:
      beta = sqrt(tau/p) * b0
      b0_j = 0 w.p. 1-q;  N(0, q^{-1}) w.p. q

    This implies beta energy scales like tau (up to Sigma_2 effects),
    so eta = X beta stays small when tau is small -> weak signal.
    """
    mask = (rng.random(p) < q).astype(float)         # Bernoulli(q)
    slab = np.sqrt(1.0 / q) * rng.normal(size=p)     # N(0, q^{-1})
    beta = (slab * mask) * np.sqrt(tau / p)
    return beta.reshape(-1, 1)


def coef_back_to_original_scale(pipe: Pipeline) -> np.ndarray:
    """
    Pipeline: StandardScaler -> LogisticRegressionCV
    Return coefficients on original X scale.
    """
    scaler = pipe.named_steps["scaler"]
    model = pipe.named_steps["model"]
    coef_scaled = model.coef_.reshape(-1, 1)
    return coef_scaled / scaler.scale_.reshape(-1, 1)


# =========================
# 2) One replication (logistic)
# =========================
def simulate_one_rep_logistic(
    n: int,
    p: int,
    tau: float,
    q: float,
    sqrtSigma1: np.ndarray,
    sqrtSigma2: np.ndarray,
    rep_seed: int,
    n_oos: int = 10000,
):
    rng = np.random.default_rng(rep_seed)

    beta = generate_beta_from_tau(p, tau, q, rng)

    # Train X: X = Sigma1^{1/2} Z Sigma2^{1/2}
    Z = rng.normal(size=(n, p))
    X = (sqrtSigma1 @ Z) @ sqrtSigma2

    # Train y ~ Bernoulli(sigmoid(X beta))
    eta = (X @ beta).ravel()
    p_true = sigmoid(eta)
    y = rng.binomial(1, p_true).astype(float)

    # Logistic L1/L2 with 10-fold CV, intercept=FALSE, standardize=TRUE (via scaler)
    # In sklearn, C is inverse regularization strength; use a grid similar spirit to glmnet path.
    Cs = np.logspace(-6, 6, 60)

    logit_l1 = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", LogisticRegressionCV(
                Cs=Cs,
                cv=10,
                penalty="l1",
                solver="saga",
                scoring="neg_log_loss",
                fit_intercept=False,
                max_iter=6000,
                n_jobs=-1,  # keep 1 here; parallelize outside if needed
                refit=True,
            )),
        ]
    )

    logit_l2 = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", LogisticRegressionCV(
                Cs=Cs,
                cv=10,
                penalty="l2",
                solver="saga",
                scoring="neg_log_loss",
                fit_intercept=False,
                max_iter=6000,
                n_jobs=-1,
                refit=True,
            )),
        ]
    )

    logit_l1.fit(X, y)
    logit_l2.fit(X, y)

    bhat_l1 = coef_back_to_original_scale(logit_l1)
    bhat_l2 = coef_back_to_original_scale(logit_l2)

    # Delta analog (keep same form as linear paper, using tau as signal scale):
    # Delta = (p/n) * tau^{-2} * ( ||Sigma2^{1/2}(bhat-beta)||^2 - ||Sigma2^{1/2}beta||^2 )
    def Delta(bhat: np.ndarray) -> float:
        term1 = np.sum((sqrtSigma2 @ (bhat - beta)) ** 2)
        term0 = np.sum((sqrtSigma2 @ beta) ** 2)
        return (p / n) * (tau ** -2) * (term1 - term0)

    DeltaL1 = float(Delta(bhat_l1))
    DeltaL2 = float(Delta(bhat_l2))

    # OOS: X_oos iid rows with same Sigma2, y_oos Bernoulli(sigmoid(X_oos beta))
    Z_oos = rng.normal(size=(n_oos, p))
    X_oos = Z_oos @ sqrtSigma2
    eta_oos = (X_oos @ beta).ravel()
    p_oos_true = sigmoid(eta_oos)
    y_oos = rng.binomial(1, p_oos_true).astype(float)

    # Predict probabilities
    eta_hat_l1 = (X_oos @ bhat_l1).ravel()
    eta_hat_l2 = (X_oos @ bhat_l2).ravel()
    p_hat_l1 = sigmoid(eta_hat_l1)
    p_hat_l2 = sigmoid(eta_hat_l2)

    # Brier-based "R2" (bounded, works for binary):
    # R2_brier = 1 - MSE(y, p_hat) / MSE(y, 0.5)
    # Baseline 0.5 matches weak-signal null in logistic around eta≈0.
    brier_null = np.mean((y_oos - 0.5) ** 2)
    brier_l1 = np.mean((y_oos - p_hat_l1) ** 2)
    brier_l2 = np.mean((y_oos - p_hat_l2) ** 2)
    R2oos_l1 = 1.0 - brier_l1 / brier_null
    R2oos_l2 = 1.0 - brier_l2 / brier_null

    return R2oos_l1, R2oos_l2, DeltaL1, DeltaL2


# =========================
# 3) Main: run six (tau,q) groups and append to exam_main_logistic.txt
# =========================
def run_and_save_logistic(
    out_path: str = "exam_main_logistic.txt",
    REPS: int = 1000,
    n: int = 500,
    p: int = 300,
    seed_design: int = 1000,
):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Fixed X design matrices
    sqrtSigma1, sqrtSigma2 = setup_design_matrices(n, p, rho=0.5, seed=seed_design)

    # Choose tau to control weak/strong signal (analog to R2=5%,50% in linear)
    tau_weak = 0.05 / (1.0 - 0.05)    # ≈ 0.05263
    tau_strong = 0.50 / (1.0 - 0.50)  # = 1.0

    settings = [
        #(tau_weak, 0.05),
        (tau_weak, 0.2),
        (tau_weak, 0.8),
        #(tau_strong, 0.05),
        (tau_strong, 0.2),
        #(tau_strong, 0.8),
    ]

    # Write format (each line 6 numbers):
    # tau q R2oos_L1 R2oos_L2 DeltaL1 DeltaL2
    # (L1 = logistic lasso; L2 = logistic ridge)
    with open(out_path, "a", encoding="utf-8") as f:
        for (tau, q) in settings:
            for ii in range(1, REPS + 1):
                rep_seed = seed_design + ii  # matches your R: set.seed(1000+ii)

                R2oos_l1, R2oos_l2, dL1, dL2 = simulate_one_rep_logistic(
                    n=n,
                    p=p,
                    tau=tau,
                    q=q,
                    sqrtSigma1=sqrtSigma1,
                    sqrtSigma2=sqrtSigma2,
                    rep_seed=rep_seed,
                    n_oos=10000,
                )

                f.write(f"{tau} {q} {R2oos_l1} {R2oos_l2} {dL1} {dL2}\n")

                print(f"(tau={tau:.6g}, q={q}) finished {ii}/{REPS}")

    print(f"Done. Results appended to: {out_path}")


if __name__ == "__main__":
    run_and_save_logistic(
        out_path="exam_main_logistic.txt",
        REPS=1000,
        n=500,
        p=300,
        seed_design=1000,
    )
