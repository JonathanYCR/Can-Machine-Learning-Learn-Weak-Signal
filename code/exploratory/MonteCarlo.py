import os
import numpy as np
from numpy.linalg import qr
from scipy.linalg import sqrtm

# glmnet_python
# pip install glmnet_py glmnet
from glmnet import cvglmnet, cvglmnetPredict, cvglmnetCoef

def random_orthogonal(p: int, rng: np.random.RandomState) -> np.ndarray:
    A = rng.randn(p, p)
    Q, R = qr(A)
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q * s
    return Q

def make_toeplitz_ar1(n: int, rho: float) -> np.ndarray:
    idx = np.arange(n)
    return rho ** np.abs(idx[:, None] - idx[None, :])

def simulate_one(ii, R2, sparsity, n, p, tau_n, sqrtSigma_1, sqrtSigma_2, sqrtSigma_ep,
                 sigmaep=1.0, sigmabeta=1.0, n_oos=10000, nfolds=10):
    # 尽量贴近 R：每次 set.seed(1000+ii)
    rng = np.random.RandomState(1000 + ii)

    zero_loc = (rng.rand(p) > (1.0 - sparsity)).astype(float)

    beta = (np.sqrt(1.0 / sparsity) * sigmabeta * rng.randn(p) / np.sqrt(p * (1.0 / tau_n))) * zero_loc
    beta = beta.reshape(-1, 1)

    ep = sqrtSigma_ep @ (rng.randn(n, 1) * sigmaep)
    X = (sqrtSigma_1 @ rng.randn(n, p)) @ sqrtSigma_2
    y = (X @ beta + ep).reshape(-1)

    X_oos = (rng.randn(n_oos, p)) @ sqrtSigma_2
    y_oos = (X_oos @ beta + rng.randn(n_oos, 1) * sigmaep).reshape(-1)

    # ----- 关键：用 cvglmnet，且对齐 glmnet 的默认行为 -----
    # glmnet_python 的参数名和 R 不完全一样，但核心要点：
    # 1) intr=False 对齐 intercept=FALSE
    # 2) standardize=True 对齐 glmnet 默认 standardize=TRUE
    # 3) nfolds=10 对齐 cv.glmnet 默认
    # 4) 用 lambda_min
    fit_lasso = cvglmnet(
        x=X, y=y,
        family='gaussian',
        alpha=1.0,
        nfolds=nfolds,
        intr=False,
        standardize=True
    )
    fit_ridge = cvglmnet(
        x=X, y=y,
        family='gaussian',
        alpha=0.0,
        nfolds=nfolds,
        intr=False,
        standardize=True
    )

    # 系数（去掉截距项；intr=False 时通常没有截距，但稳妥起见按“第一项是截距”处理）
    bh_lasso = cvglmnetCoef(fit_lasso, s='lambda_min')
    bh_ridge = cvglmnetCoef(fit_ridge, s='lambda_min')

    # bh_* 可能是 (p+1,1) 的稀疏/数组：第一行是 intercept（即使 intr=False 也可能占位）
    bh_lasso = np.asarray(bh_lasso).reshape(-1, 1)
    bh_ridge = np.asarray(bh_ridge).reshape(-1, 1)
    if bh_lasso.shape[0] == p + 1:
        bh_lasso = bh_lasso[1:, :]
    if bh_ridge.shape[0] == p + 1:
        bh_ridge = bh_ridge[1:, :]

    # Delta 同作者
    b = sqrtSigma_2 @ beta
    a_l = sqrtSigma_2 @ (bh_lasso - beta)
    a_r = sqrtSigma_2 @ (bh_ridge - beta)

    DeltaLasso = (p / n) * (tau_n ** (-2)) * (float(np.sum(a_l**2)) - float(np.sum(b**2)))
    DeltaRidge = (p / n) * (tau_n ** (-2)) * (float(np.sum(a_r**2)) - float(np.sum(b**2)))

    # OOS R^2 同作者：1 - MSE / mean(y_oos^2)
    yhat_l = cvglmnetPredict(fit_lasso, X_oos, s='lambda_min', ptype='response').reshape(-1)
    yhat_r = cvglmnetPredict(fit_ridge, X_oos, s='lambda_min', ptype='response').reshape(-1)

    mse_l = np.mean((y_oos - yhat_l) ** 2)
    mse_r = np.mean((y_oos - yhat_r) ** 2)
    denom = np.mean(y_oos ** 2)

    R2oos_lasso = 1.0 - mse_l / denom
    R2oos_ridge = 1.0 - mse_r / denom

    return np.array([R2, sparsity, R2oos_lasso, R2oos_ridge, DeltaLasso, DeltaRidge], float)

def main(R2, sparsity, n, p, n_iter=1000, n_oos=10000, rho_1=0.5,
         out_path="weaksimu_revised/exam_main.txt"):
    tau_n = R2 / (1.0 - R2)

    # 对齐作者：set.seed(1000) 后生成 Sigma 结构
    rng0 = np.random.RandomState(1000)

    Sigma_1 = make_toeplitz_ar1(n, rho_1)
    sqrtSigma_1 = np.real_if_close(sqrtm(Sigma_1))

    eigen_Sigma_ep = 0.5 + rng0.rand(n)
    sqrtSigma_ep = np.diag(np.sqrt(eigen_Sigma_ep))

    U2 = random_orthogonal(p, rng0)
    eigen_Sigma_2 = 0.5 + rng0.rand(p)
    sqrtSigma_2 = U2 @ np.diag(np.sqrt(eigen_Sigma_2)) @ U2.T

    res = []
    for ii in range(1, n_iter + 1):
        res.append(simulate_one(ii, R2, sparsity, n, p, tau_n, sqrtSigma_1, sqrtSigma_2, sqrtSigma_ep,
                                n_oos=n_oos, nfolds=10))
    res = np.vstack(res)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for row in res:
            f.write(" ".join(f"{x:.10g}" for x in row) + "\n")

    return res

if __name__ == "__main__":
    main(R2=0.05, sparsity=0.05, n=500, p=300)
    main(R2=0.05, sparsity=0.2,  n=500, p=300)
    main(R2=0.05, sparsity=0.8,  n=500, p=300)
    main(R2=0.5,  sparsity=0.05, n=500, p=300)
    main(R2=0.5,  sparsity=0.2,  n=500, p=300)
    main(R2=0.5,  sparsity=0.8,  n=500, p=300)
