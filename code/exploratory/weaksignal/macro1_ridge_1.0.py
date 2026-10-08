import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold

# ==============================
# 固定参数
# ==============================
MAT_PATH = r"C:\Users\jonat\Desktop\论文\weaksignal\weaksignal\Macro1_construction\FredMD/FredMDlargeHor1.mat"
OUTPUT_PATH = "macro1_realrecession_logistic_ridge_step12.txt"

INITIAL_TRAIN = 179
TEST_WINDOW = 12
STEP = 12                     # 每次扩展 12 期
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
        print(f"Skip window ending at {train_end} (only one class in training).")
        continue

    # ======================
    # Baseline: null model (intercept-only)
    # ======================
    pi0 = float(y_train.mean())
    eps = 1e-6
    pi0 = min(max(pi0, eps), 1 - eps)

    baseline_prob = np.full_like(y_test, pi0, dtype=float)
    baseline_ll = log_loss(y_test, baseline_prob, labels=[0, 1])

    # baseline 的分类预测：常数类别
    baseline_pred = np.full_like(y_test, 1 if pi0 >= 0.5 else 0, dtype=int)
    baseline_acc = accuracy_score(y_test, baseline_pred)

    # ======================
    # Logistic Ridge + CV (no shuffle)
    # ======================
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=False
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

    # 模型分类预测（0.5 阈值）
    model_pred = (prob >= 0.5).astype(int)
    model_acc = accuracy_score(y_test, model_pred)

    best_C = float(np.ravel(model.named_steps["lr"].C_)[0])

    # 相对 null 的 improvement（logloss：越小越好）
    excess_vs_null = model_ll - baseline_ll

    # accuracy 相对 baseline 的提升（accuracy：越大越好）
    acc_improve = model_acc - baseline_acc

    rows.append([
        train_end,
        model_ll,
        baseline_ll,
        excess_vs_null,
        model_acc,
        baseline_acc,
        acc_improve,
        pi0,
        best_C
    ])

    print(
        f"Window ending at {train_end}: "
        f"excess_vs_null={excess_vs_null:+.6f}, "
        f"acc_improve={acc_improve:+.4f}, "
        f"acc_model={model_acc:.4f}, acc_base={baseline_acc:.4f}, "
        f"pi0={pi0:.3f}, best_C={best_C:.4f}"
    )

# 转数组 + 安全检查
if len(rows) == 0:
    raise RuntimeError("No valid windows (training data has only one class in all windows).")

rows = np.array(rows, dtype=float)

# ==============================
# 保存结果
# ==============================
np.savetxt(
    OUTPUT_PATH,
    rows,
    fmt="%.8f",
    header="train_end model_logloss baseline_logloss excess_vs_null "
           "model_acc baseline_acc acc_improve pi0 best_C"
)

print("\nFinished.")
print("Saved to:", OUTPUT_PATH)

print("\n=== Summary (across windows) ===")
print("Average excess_vs_null:", rows[:, 3].mean())
print("Median  excess_vs_null:", np.median(rows[:, 3]))
print("Share (excess<0):", np.mean(rows[:, 3] < 0))

print("Average acc_improve:", rows[:, 6].mean())
print("Median  acc_improve:", np.median(rows[:, 6]))
print("Share (acc_improve>0):", np.mean(rows[:, 6] > 0))
print("Share (acc_improve<0):", np.mean(rows[:, 6] < 0))