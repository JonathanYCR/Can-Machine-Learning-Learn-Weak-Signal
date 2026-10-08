import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

# ==============================
# 固定参数
# ==============================
MAT_PATH = r"C:\Users\jonat\Desktop\论文\weaksignal\weaksignal\Macro1_construction\FredMD/FredMDlargeHor1.mat"
OUTPUT_PATH = "macro1_realrecession_logistic_ridge_step12.txt"

INITIAL_TRAIN = 179
TEST_WINDOW = 12
STEP = 12                     # ⭐ 每次扩展 12 期
RANDOM_SEED = 42
CV_FOLDS = 10

CS_GRID = np.logspace(-3, 1, 30)

np.random.seed(RANDOM_SEED)

# ==============================
# 读取数据
# ==============================
data = loadmat(MAT_PATH)
X = data["X"]
X = np.delete(X, 5, axis=1)
y_raw = data["Y"].ravel()

# recession 转 0/1
y = (y_raw > 0).astype(int)

n = len(y)
p = X.shape[1]

print(f"Sample size: {n}")
print(f"Dimension p: {p}")
print(f"Recession rate: {y.mean():.4f}")

# ==============================
# Expanding window (step = 12)
# ==============================
rows = []

start_points = range(INITIAL_TRAIN, n - TEST_WINDOW + 1, STEP)

for train_end in start_points:

    test_end = train_end + TEST_WINDOW

    X_train = X[:train_end, :]
    y_train = y[:train_end]

    X_test = X[train_end:test_end, :]
    y_test = y[train_end:test_end]

    # 如果训练集只有一个类别，跳过
    if len(np.unique(y_train)) < 2:
        continue

    # ======================
    # Baseline: null model
    # ======================
    pi0 = float(y_train.mean())
    eps = 1e-6
    pi0 = min(max(pi0, eps), 1 - eps)

    baseline_prob = np.full_like(y_test, pi0, dtype=float)
    baseline_ll = log_loss(y_test, baseline_prob, labels=[0, 1])

    # ======================
    # Logistic Ridge + CV
    # ======================
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegressionCV(
            Cs=CS_GRID,
            cv=cv,
            penalty="l2",
            solver="lbfgs",
            scoring="neg_log_loss",
            max_iter=5000,
            n_jobs=-1,
            refit=True
        ))
    ])

    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1]
    model_ll = log_loss(y_test, prob, labels=[0, 1])

    best_C = float(model.named_steps["lr"].C_[0])

    # 相对 null 的 improvement
    excess_vs_null = model_ll - baseline_ll

    rows.append([
        train_end,
        model_ll,
        baseline_ll,
        excess_vs_null,
        pi0,
        best_C
    ])

    print(f"Window ending at {train_end}: "
          f"excess_vs_null={excess_vs_null:+.6f}, "
          f"pi0={pi0:.3f}, best_C={best_C:.4f}")

rows = np.array(rows, dtype=float)

# ==============================
# 保存结果
# ==============================
np.savetxt(
    OUTPUT_PATH,
    rows,
    fmt="%.8f",
    header="train_end model_logloss baseline_logloss excess_vs_null pi0 best_C"
)

print("\nFinished.")
print("Saved to:", OUTPUT_PATH)
print("Average excess_vs_null:", rows[:, 3].mean())
print("Median  excess_vs_null:", np.median(rows[:, 3]))
print("Share (excess<0):", np.mean(rows[:, 3] < 0))