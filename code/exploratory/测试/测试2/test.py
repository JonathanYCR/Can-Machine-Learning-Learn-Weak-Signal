import os
import numpy as np
from scipy.linalg import toeplitz, cholesky
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
import warnings

warnings.filterwarnings("ignore")

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# -----------------------
# 1) Utilities
# -----------------------
def rand_orthogonal(p: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(p, p))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q

def setup_design_matrices(n: int, p: int, rho: float = 0.5, seed: int = 1000):
    rng = np.random.default_rng(seed)

    c = rho ** np.arange(n)
    Sigma1 = toeplitz(c)
    sqrtSigma1 = cholesky(Sigma1, lower=True)

    U = rand_orthogonal(p, rng)
    eigs = 0.5 + rng.random(p)
    sqrtSigma2 = U @ np.diag(np.sqrt(eigs)) @ U.T

    return sqrtSigma1, sqrtSigma2

def generate_beta(p: int, tau: float, q: float, rng: np.random.Generator) -> np.ndarray:
    mask = (rng.random(p) < q).astype(float)
    slab = np.sqrt(1.0 / q) * rng.normal(size=p)
    beta = (slab * mask) * np.sqrt(tau / p)
    return beta.reshape(-1, 1)

def coef_back_to_original_scale(pipe: Pipeline) -> np.ndarray:
    scaler = pipe.named_steps["scaler"]
    model = pipe.named_steps["model"]
    coef_scaled = model.coef_.reshape(-1, 1)
    return coef_scaled / scaler.scale_.reshape(-1, 1)

def mcfadden_pseudo_r2(y_true, p_hat, p_null=0.5):
    eps = 1e-15
    p_hat = np.clip(p_hat, eps, 1 - eps)
    ll_model = -log_loss(y_true, p_hat, labels=[0, 1], normalize=True)
    ll_null = -log_loss(y_true, np.full_like(p_hat, p_null), labels=[0, 1], normalize=True)
    return 1.0 - (ll_model / ll_null)

# -----------------------
# 2) One replication
# -----------------------
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
    beta = generate_beta(p, tau, q, rng)

    # Train X
    Z = rng.normal(size=(n, p))
    X = (sqrtSigma1 @ Z) @ sqrtSigma2

    # Logistic y
    eta = (X @ beta).ravel()
    p_true = sigmoid(eta)
    y = rng.binomial(1, p_true, size=n)

    # ===== Key fix for "Delta piles at 0" =====
    # Make sure CV sees VERY weak regularization options too (large C).
    Cs = np.logspace(-3, 8, 12)

    common_kwargs = dict(
        Cs=Cs,
        cv=10,
        solver="saga",
        fit_intercept=False,
        scoring="neg_log_loss",
        max_iter=4000,
        tol=1e-4,
        n_jobs=-1,      # (2) your requirement
        refit=True
    )

    lasso = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", LogisticRegressionCV(penalty="l1", **common_kwargs)),
        ]
    )
    ridge = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", LogisticRegressionCV(penalty="l2", **common_kwargs)),
        ]
    )

    lasso.fit(X, y)
    ridge.fit(X, y)

    bhat_lasso = coef_back_to_original_scale(lasso)
    bhat_ridge = coef_back_to_original_scale(ridge)

    # Delta (keep SAME eq(8)-style definition)
    def Delta(bhat: np.ndarray) -> float:
        term1 = np.sum((sqrtSigma2 @ (bhat - beta)) ** 2)
        term0 = np.sum((sqrtSigma2 @ beta) ** 2)
        return (p / n) * (tau ** -2) * (term1 - term0)

    DeltaLasso = float(Delta(bhat_lasso))
    DeltaRidge = float(Delta(bhat_ridge))

    # OOS pseudo-R2 (McFadden)
    Z_oos = rng.normal(size=(n_oos, p))
    X_oos = Z_oos @ sqrtSigma2
    eta_oos = (X_oos @ beta).ravel()
    p_oos = sigmoid(eta_oos)
    y_oos = rng.binomial(1, p_oos, size=n_oos)

    p_hat_lasso = lasso.predict_proba(X_oos)[:, 1]
    p_hat_ridge = ridge.predict_proba(X_oos)[:, 1]
    R2oos_lasso = float(mcfadden_pseudo_r2(y_oos, p_hat_lasso, p_null=0.5))
    R2oos_ridge = float(mcfadden_pseudo_r2(y_oos, p_hat_ridge, p_null=0.5))

    return R2oos_lasso, R2oos_ridge, DeltaLasso, DeltaRidge

# -----------------------
# 3) Main: run all settings + append
# -----------------------
def run_and_save_logistic(
    out_path: str = "exam_main_logistic.txt",
    REPS: int = 100,     # (1) your requirement
    n: int = 500,
    p: int = 300,
    seed_design: int = 1000,
):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    sqrtSigma1, sqrtSigma2 = setup_design_matrices(n, p, rho=0.5, seed=seed_design)

    # Keep six groups (tau,q). You requested qs=[0.2,0.8].
    qs = [0.2, 0.8]       # (3) your requirement
    taus = [0.05, 0.2, 1.0]  # 3 taus x 2 qs = 6 groups

    settings = [(tau, q) for tau in taus for q in qs]

    with open(out_path, "a", encoding="utf-8") as f:
        for (tau, q) in settings:
            for ii in range(1, REPS + 1):
                rep_seed = seed_design + ii
                R2oos_l, R2oos_r, dL, dR = simulate_one_rep_logistic(
                    n=n, p=p, tau=tau, q=q,
                    sqrtSigma1=sqrtSigma1, sqrtSigma2=sqrtSigma2,
                    rep_seed=rep_seed, n_oos=10000
                )
                # tau, q, R2oos_lasso, R2oos_ridge, DeltaLasso, DeltaRidge
                f.write(f"{tau} {q} {R2oos_l} {R2oos_r} {dL} {dR}\n")

                # (4) your requirement: print every ii
                print(f"(tau={tau}, q={q}) finished {ii}/{REPS}")

    print(f"Done. Results appended to: {out_path}")

if __name__ == "__main__":
    run_and_save_logistic(out_path="exam_main_logistic.txt", REPS=100, n=500, p=300, seed_design=1000)
