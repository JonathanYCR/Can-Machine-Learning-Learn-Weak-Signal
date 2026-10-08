# -*- coding: utf-8 -*-
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


def fit_logistic_lasso_fixedC(X, y, C_lambda: float):
    """Directly control C_lambda (sklearn's C). Smaller C -> stronger L1 penalty."""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            penalty="l1",
            C=C_lambda,
            fit_intercept=False,
            solver="saga",
            max_iter=30000,
            tol=1e-4,
            n_jobs=-1
        ))
    ])
    model.fit(X, y.ravel())
    return model


def logloss_from_proba(p1, y):
    eps = 1e-12
    p1 = np.clip(p1, eps, 1 - eps)
    y = y.ravel()
    return -np.mean(y * np.log(p1) + (1 - y) * np.log(1 - p1))


def plot_and_save_boxplot(data, labels, title, xlabel, ylabel, filename):
    plt.figure()
    plt.boxplot(data, labels=[str(x) for x in labels], showfliers=False)
    plt.axhline(0.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved: {filename}")


def print_quartiles(name, data, labels):
    print(f"\n{name} quartiles (Q1, Median, Q3):")
    for j, lab in enumerate(labels):
        q1 = np.percentile(data[:, j], 25)
        q2 = np.percentile(data[:, j], 50)
        q3 = np.percentile(data[:, j], 75)
        print(f"{lab:<10}  Q1={q1:.6f}  Median={q2:.6f}  Q3={q3:.6f}")


def main_logistic_lasso_experiment3_fixedC(
    tau_n: float,
    sparsity: float,
    n: int,
    p: int,
    C_lambdas=(0.1,0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01),  # 你可以改成 (100,50,25) 更像 Figure A2
    REPS: int = 200,
    n_oos: int = 10000,
    rho_1: float = 0.5,
    sigmabeta: float = 1.0,
    seed_global: int = 1000,
    out_prefix: str = "exp3_lasso_fixedC",
):
    sqrtSigma_1, sqrtSigma_2 = build_matrices(n, p, rho_1=rho_1, seed=seed_global)
    C_lambdas = np.array(list(C_lambdas), dtype=float)

    # scaled = (p/(n*tau^2))*(logloss_oos - log2)   <-- 按你的原式，不用 max 截断
    scaled_lasso = np.zeros((REPS, len(C_lambdas)))

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

        for j, Cc in enumerate(C_lambdas):
            model_l = fit_logistic_lasso_fixedC(X, y, C_lambda=Cc)
            p1_oos_l = model_l.predict_proba(X_oos)[:, 1]
            ll_oos_l = logloss_from_proba(p1_oos_l, y_oos)

            scaled_lasso[ii - 1, j] = (p / (n * (tau_n ** 2))) * (ll_oos_l - np.log(2))
            print(Cc, scaled_lasso[ii - 1, j])

        print(f"(tau={tau_n}, q={sparsity}) finished {ii}/{REPS}")

    base = f"{out_prefix}_tau{tau_n}_q{sparsity}_n{n}_p{p}_REPS{REPS}"

    plot_and_save_boxplot(
        scaled_lasso, C_lambdas,
        title=f"Logistic Lasso Scaled Excess (fixed $C_\\lambda$) \n  (tau={tau_n}, q={sparsity}, n={n}, p={p})",
        xlabel=r"$C_\lambda$ (smaller = stronger L1 penalty)",
        ylabel=r"Scaled excess  $\tilde{\Delta} = \frac{p}{n\tau^2}\,(\mathrm{logloss}-\log 2)$",
        filename=f"{base}_LASSO_BOX.png"
    )

    print_quartiles("LASSO (fixed C_lambda)", scaled_lasso, [f"C={c:g}" for c in C_lambdas])
    return C_lambdas, scaled_lasso


if __name__ == "__main__":
    main_logistic_lasso_experiment3_fixedC(tau_n=0.05, sparsity=0.2, n=500, p=300, REPS=200)
