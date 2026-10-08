import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

path = r"C:\Users\jonat\Desktop\论文\bank-additional\bank-additional-full.csv"
df = pd.read_csv(path, sep=";")

print(df.shape)
#print(df.head())
#print(df["y"].value_counts())

df["y_bin"] = (df["y"] == "yes").astype(int)
df["y_bin"].mean()


# 0) 构造二元标签
df = df.copy()
df["y_bin"] = (df["y"] == "yes").astype(int)

# 1) 准备 X / y
X = df.drop(columns=["y", "y_bin"])
y = df["y_bin"].values

# 2) 识别数值/类别列
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("Numeric cols:", numeric_features)
print("Categorical cols:", categorical_features)
print("Positive rate (mean y):", y.mean())

# 3) Train/Test split（为了先做弱信号对比：简单随机切分）
#    你后面写论文建议做：time split 或 nested CV
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

# 4) 预处理：数值标准化 + 类别 OneHot
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# 5) Baseline：永远预测训练集正类比例
p_base = np.mean(y_train)
pred_base = np.full_like(y_test, fill_value=p_base, dtype=float)

baseline_metrics = {
    "Model": "Baseline (freq)",
    "AUC": roc_auc_score(y_test, pred_base),
    "LogLoss": log_loss(y_test, pred_base),
    "Brier": brier_score_loss(y_test, pred_base),
}

# 6) Ridge Logistic (L2)
ridge = Pipeline(
    steps=[
        ("prep", preprocess),
        ("clf", LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=5000
        ))
    ]
)
ridge.fit(X_train, y_train)
pred_ridge = ridge.predict_proba(X_test)[:, 1]

ridge_metrics = {
    "Model": "Ridge Logistic (L2)",
    "AUC": roc_auc_score(y_test, pred_ridge),
    "LogLoss": log_loss(y_test, pred_ridge),
    "Brier": brier_score_loss(y_test, pred_ridge),
}

# 7) Lasso Logistic (L1)
# liblinear 支持 L1/L2；elastic-net 需要 saga。:contentReference[oaicite:2]{index=2}
lasso = Pipeline(
    steps=[
        ("prep", preprocess),
        ("clf", LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=5000
        ))
    ]
)
lasso.fit(X_train, y_train)
pred_lasso = lasso.predict_proba(X_test)[:, 1]

lasso_metrics = {
    "Model": "Lasso Logistic (L1)",
    "AUC": roc_auc_score(y_test, pred_lasso),
    "LogLoss": log_loss(y_test, pred_lasso),
    "Brier": brier_score_loss(y_test, pred_lasso),
}

# 8) 输出结果
results = pd.DataFrame([baseline_metrics, ridge_metrics, lasso_metrics])
print(results)

# 取出 lasso 的最终模型
lasso_clf = lasso.named_steps["clf"]

# 注意：这是 OneHot 后的 expanded feature 空间系数
coef = lasso_clf.coef_.ravel()

nonzero = np.sum(np.abs(coef) > 1e-12)
total = coef.shape[0]

print(f"Lasso non-zero coefficients: {nonzero} / {total}")
