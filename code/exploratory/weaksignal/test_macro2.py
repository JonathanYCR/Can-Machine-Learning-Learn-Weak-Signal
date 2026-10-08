import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

# ==============================
# 1) 固定参数：不需要输入
# ==============================
MAT_PATH = r"C:\Users\jonat\Desktop\论文\weaksignal\weaksignal\ecmadata\data\Macro2\GrowthData.mat"
OUTPUT_PATH = r"macro2_logistic_ridge_empirical.txt"

AN = 1                 # 你要的 an 参数：在这里手动设置，不需要输入
INITIAL_TRAIN = 179
TEST_WINDOW = 12

RANDOM_SEED = 42
CV_FOLDS = 10
CS_GRID = np.logspace(-2, 1, 25)   # C 越小正则越强

np.random.seed(RANDOM_SEED)

# ==============================
# 2) 读取数据（保持原作者结构：data最后一列是y）
# ==============================
Macro2 = loadmat(MAT_PATH)
data = Macro2["data"]

# 原代码：加一列1当作截距，再拆分；这里保持一致
data = np.column_stack((np.ones(data.shape[0]), data))
p = data.shape[1] - 1
n = data.shape[0]

y_raw = data[:, p]
X = data[:, :p]
X = X[:, 1:]   # drop intercept column (跟原作者一致)

print(f"[info] n={n}, p={X.shape[1]}")

# ==============================
# 3) y 二分类：>=0 -> 1, <0 -> 0
# ==============================
y = (y_raw >= 0).astype(int)
print(f"[info] overall y mean={y.mean():.4f}")

# ==============================
# 4) 按“之前那套”训练测试划分：不读文件
# ==============================
train_end = INITIAL_TRAIN + (AN - 1) * TEST_WINDOW
test_end = train_end + TEST_WINDOW

if test_end > n:
    raise ValueError(f"Not enough observations for AN={AN}: need test_end={test_end}, but n={n}")

train_index = np.arange(0, train_end)
test_index = np.arange(train_end, test_end)

X_train, y_train = X[train_index], y[train_index]
X_test, y_test = X[test_index], y[test_index]

# 训练集如果只有一个类别，logistic 无法训练
if len(np.unique(y_train)) < 2:
    raise ValueError(f"Training set has only one class for AN={AN}. "
                     f"Try a larger AN or check y construction.")

# ==============================
# 5) Baseline：intercept-only, π0 = mean(y_train)
# ==============================
pi0 = float(y_train.mean())
eps = 1e-6
pi0 = min(max(pi0, eps), 1 - eps)

baseline_prob = np.full(shape=len(y_test), fill_value=pi0, dtype=float)
baseline_ll = log_loss(y_test, baseline_prob, labels=[0, 1])

# ==============================
# 6) Logistic Ridge + CV (neg_log_loss)
# ==============================
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

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

excess_vs_null = model_ll - baseline_ll  # <0 表示优于 intercept-only

print(f"[AN={AN}] model_ll={model_ll:.6f}, baseline_ll={baseline_ll:.6f}, "
      f"excess_vs_null={excess_vs_null:+.6f}, pi0={pi0:.3f}, best_C={best_C:.3g}")

# ==============================
# 7) 保存结果
# ==============================
with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
    f.write(
        f"{AN} {train_end} "
        f"{model_ll:.8f} {baseline_ll:.8f} {excess_vs_null:.8f} "
        f"{pi0:.8f} {best_C:.8g}\n"
    )

print(f"[saved] {OUTPUT_PATH}")