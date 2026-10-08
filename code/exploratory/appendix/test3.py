import numpy as np
from numpy.random import default_rng
from scipy.linalg import toeplitz, cholesky
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

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

    eigen_Sigma_ep = 0.5 + rng.random(n)
    sqrtSigma_ep = np.diag(np.sqrt(eigen_Sigma_ep))

    U2 = rand_orthogonal(p, rng)
    eigen_Sigma_2 = 0.5 + rng.random(p)
    sqrtSigma_2 = U2 @ np.diag(np.sqrt(eigen_Sigma_2)) @ U2.T
    return sqrtSigma_1, sqrtSigma_ep, sqrtSigma_2

def gen_beta(p, tau, q, rng, sigmabeta=1.0):
    mask = (rng.random(p) > 1.0 - q).astype(float).reshape(-1, 1)
    beta = (np.sqrt(1.0 / q) * sigmabeta *
            rng.normal(size=(p, 1)) * np.sqrt(tau / p) * mask)
    return beta

def Delta(beta_hat, beta, sqrtSigma_2, p, n, tau):
    s2_beta = sqrtSigma_2 @ beta
    s2_diff = sqrtSigma_2 @ (beta_hat - beta)
    return (p / n) * (tau ** -2) * (float(np.sum(s2_diff ** 2)) - float(np.sum(s2_beta ** 2)))

def alpha_star(lam_const, theta2=13/12, theta1=1.0, sig_eps2=1.0, sig_beta2=1.0, sig_x2=1.0):
    # alpha*(λ) = 2 θ2 σ_x^4 ( (σ_eps^2 θ1)/(2 λ^2) - (σ_beta^2)/λ )
    return 2 * theta2 * (sig_x2**4) * ((sig_eps2 * theta1) / (2 * lam_const**2) - (sig_beta2) / lam_const)

def simulate_Delta_ridge(R2, q, n, p, lam_grid=(0.5, 1.0, 2.0), REPS=1000):
    tau = R2 / (1 - R2)
    sqrtSigma_1, sqrtSigma_ep, sqrtSigma_2 = setup(n, p, seed=1000)

    out = {lam: [] for lam in lam_grid}

    for ii in range(1, REPS + 1):
        rng = default_rng(1000 + ii)

        beta = gen_beta(p, tau=tau, q=q, rng=rng)

        Z = rng.normal(size=(n, p))
        X = sqrtSigma_1 @ Z @ sqrtSigma_2

        eps = sqrtSigma_ep @ rng.normal(size=(n, 1))
        y = X @ beta + eps
        y = y.ravel()

        for lam_const in lam_grid:
            lambda_n = (tau ** -1) * lam_const
            # ✅ 正确映射：alpha = p * lambda_n
            alpha = p * lambda_n

            # 用回归模型（迭代 solver），不是自己写闭式
            model = Ridge(alpha=alpha, fit_intercept=False, solver="sag", max_iter=5000, random_state=1000+ii)
            model.fit(X, y)
            beta_hat = model.coef_.reshape(-1, 1)

            d = Delta(beta_hat, beta, sqrtSigma_2, p, n, tau)
            print(d)
            out[lam_const].append(d)

        print(f"(n={n}, tau={tau:.6g}) finished {ii}/{REPS}")

    return tau, out

def plot_A1_like(R2=0.05, q=0.2, REPS=1000, lam_grid=(0.5,1.0,2.0), save_path="fig_A1_python.png"):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)

    # top row: n=500, p=300
    tau1, out1 = simulate_Delta_ridge(R2, q, n=500, p=300, lam_grid=lam_grid, REPS=REPS)
    # bottom row: n=2500, p=1500
    tau2, out2 = simulate_Delta_ridge(R2, q, n=2500, p=1500, lam_grid=lam_grid, REPS=REPS)

    for j, lam in enumerate(lam_grid):
        ax = axes[0, j]
        ax.hist(out1[lam], bins=40, density=True)
        ax.axvline(alpha_star(lam), linestyle="--")
        ax.set_title(f"λ = {lam}, n = 500")
        ax.set_xlim(-3, 2)

    for j, lam in enumerate(lam_grid):
        ax = axes[1, j]
        ax.hist(out2[lam], bins=40, density=True)
        ax.axvline(alpha_star(lam), linestyle="--")
        ax.set_title(f"λ = {lam}, n = 2500")
        ax.set_xlim(-3, 2)

    fig.suptitle("Figure A1 (Python): Ridge with Fixed Tuning Parameters", fontsize=12)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print("saved:", save_path)

if __name__ == "__main__":
    plot_A1_like(R2=0.05, q=0.2, REPS=100, lam_grid=(0.5,1.0,2.0), save_path="fig_A1_python.png")
