import numpy as np
from numpy.random import default_rng
from scipy.linalg import toeplitz, cholesky
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


def sigmoid(z):
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def rand_orthogonal(p, rng):
    A = rng.normal(size=(p, p))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def setup(n, p, rho_1=0.5, seed=1000):
    rng = default_rng(seed)
    Sigma_1 = toeplitz(rho_1 ** np.arange(n))
    sqrtSigma_1 = cholesky(Sigma_1, lower=True)

    U2 = rand_orthogonal(p, rng)
    eigen_Sigma_2 = 0.5 + rng.random(p)
    sqrtSigma_2 = U2 @ np.diag(np.sqrt(eigen_Sigma_2)) @ U2.T
    return sqrtSigma_1, sqrtSigma_2


def gen_beta(p, tau, q, rng, sigmabeta=1.0):
    mask = (rng.random(p) > 1.0 - q).astype(float).reshape(-1, 1)
    beta = (np.sqrt(1.0 / q) * sigmabeta *
            rng.normal(size=(p, 1)) * np.sqrt(tau / p) * mask)
    return beta


def fit_logistic_ridge_fixed(X, y, C, seed):
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False, with_std=True)),
        ("lr", LogisticRegression(
            penalty="l2", solver="lbfgs",
            C=float(C),
            fit_intercept=False,
            max_iter=8000,
            n_jobs=-1,
            random_state=seed
        ))
    ])
    pipe.fit(X, y)
    return pipe


def simulate_A1_logistic(R2, q, n, p, lam_grid, REPS=1000,
                         n_oos=50000, K=20, print_every=50):
    """
    A.1 style:
      - fixed tunings lambda_n = tau^{-1} * lambda
      - Monte Carlo reps
      - evaluate tildeDelta using out-of-sample logloss vs zero (log2)
      - variance-reduced hard risk: average over K Bernoulli draws at same X_oos
    """
    tau = R2 / (1 - R2)
    sqrtSigma_1, sqrtSigma_2 = setup(n, p, seed=1000)

    out = {lam: [] for lam in lam_grid}

    for ii in range(1, REPS + 1):
        seed = 1000 + ii
        rng = default_rng(seed)

        beta = gen_beta(p, tau=tau, q=q, rng=rng)

        # train
        Z = rng.normal(size=(n, p))
        X = sqrtSigma_1 @ Z @ sqrtSigma_2
        pi = sigmoid(X @ beta).reshape(-1)
        y = rng.binomial(1, pi).astype(int)

        # oos features (fixed for K draws)
        Z_oos = rng.normal(size=(n_oos, p))
        X_oos = Z_oos @ sqrtSigma_2
        pi_oos = sigmoid(X_oos @ beta).reshape(-1)

        for lam in lam_grid:
            lambda_n = (tau ** -1) * lam
            C = n / (2.0 * p * lambda_n)   # mapping

            pipe = fit_logistic_ridge_fixed(X, y, C=C, seed=seed)
            proba = pipe.predict_proba(X_oos)[:, 1]
            proba = np.clip(proba, 1e-15, 1 - 1e-15)

            # variance-reduced hard log-loss
            ll = 0.0
            for _ in range(K):
                y_oos = rng.binomial(1, pi_oos).astype(int)
                ll += log_loss(y_oos, proba)
            ll /= K

            # thesis tildeDelta
            tildeDelta = (p / n) * (tau ** -2) * (ll - np.log(2.0))
            print(tildeDelta)
            out[lam].append(tildeDelta)

        means = {lam: float(np.mean(out[lam])) for lam in lam_grid}
        print(f"(n={n}, tau={tau:.6g}) finished {ii}/{REPS} mean_tildeDelta={means}")

    return tau, out


def boxplot_by_lambda(lam_grid, out, title, save_path):
    data = [out[lam] for lam in lam_grid]
    plt.figure(figsize=(9, 4))
    plt.boxplot(data, labels=[str(lam) for lam in lam_grid], showfliers=False)
    plt.axvline(lam_grid.index(0.25) + 1, linestyle="--") if 0.25 in lam_grid else None
    plt.xlabel("lambda (constant)")
    plt.ylabel("tildeDelta")
    plt.title(title)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print("saved:", save_path)


def mean_curve(lam_grid, out, title, save_path):
    means = np.array([np.mean(out[lam]) for lam in lam_grid])
    ses = np.array([np.std(out[lam], ddof=1) / np.sqrt(len(out[lam])) for lam in lam_grid])

    plt.figure(figsize=(7, 4))
    plt.errorbar(lam_grid, means, yerr=ses, fmt="o-")
    plt.axvline(0.25, linestyle="--")
    plt.xlabel("lambda (constant)")
    plt.ylabel("mean tildeDelta")
    plt.title(title)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print("saved:", save_path)


if __name__ == "__main__":
    q = 0.2
    REPS = 1000

    # 关键：用较大的 n 让渐近更明显（像 A.1 两行对比）
    settings = [
        ("n=5000",  0.005, 5000, 300),
        ("n=25000", 0.005, 25000, 1500),
    ]

    # 你要验证 0.25 最优：网格必须在 0.25 附近加密
    lam_grid = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]

    for tag, R2, n, p in settings:
        tau, out = simulate_A1_logistic(R2=R2, q=q, n=n, p=p,
                                        lam_grid=lam_grid, REPS=REPS,
                                        n_oos=50000, K=20)
        boxplot_by_lambda(lam_grid, out,
                          title=f"{tag}, tau={tau:.4g}, q={q}",
                          save_path=f"box_{tag.replace('=','')}.png")
        mean_curve(lam_grid, out,
                   title=f"{tag}, tau={tau:.4g}, q={q}",
                   save_path=f"mean_{tag.replace('=','')}.png")
