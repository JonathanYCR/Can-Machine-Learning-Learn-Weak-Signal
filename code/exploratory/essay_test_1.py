import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

def data_design():
    # 1) 列名（Adult 数据集固定 15 列：14 个特征 + 1 个label）
    cols = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income"
    ]
    
    # 2) 读训练集 adult.data
    train_path = r"C:\Users\jonat\Desktop\论文\adult\adult.data"
    df_train = pd.read_csv(
        train_path,
        header=None,
        names=cols,
        sep=r",\s*",          # 逗号+可选空格
        engine="python"
    )
    
    # 3) 读测试集 adult.test（第一行是注释，需要跳过）
    test_path = r"C:\Users\jonat\Desktop\论文\adult\adult.test"
    df_test = pd.read_csv(
        test_path,
        header=None,
        names=cols,
        sep=r",\s*",
        engine="python",
        skiprows=1            # 跳过第一行 "|1x3 Cross validator" 之类的说明
    )
    
    print(df_train.shape, df_test.shape)
    
    # 去掉测试集标签末尾的点号
    df_test["income"] = df_test["income"].str.replace(".", "", regex=False)
    
    # 统一一下空格（保险起见）
    df_train["income"] = df_train["income"].str.strip()
    df_test["income"] = df_test["income"].str.strip()
    
    # 转成二元标签 y：>50K 为 1
    df_train["y"] = (df_train["income"] == ">50K").astype(int)
    df_test["y"] = (df_test["income"] == ">50K").astype(int)
    
    df_train["y"].value_counts(), df_test["y"].value_counts()
    
    # 把 ? 变成 NaN
    df_train = df_train.replace("?", np.nan)
    df_test = df_test.replace("?", np.nan)
    
    # 直接删掉缺失行（简单可复现）
    df_train = df_train.dropna().copy()
    df_test = df_test.dropna().copy()
    
    print(df_train.shape, df_test.shape)
    return df_train, df_test

def machine_learning(df_train, df_test):
    # ------------------------
    # 1) 准备 X, y
    # ------------------------
    target_col = "y"
    drop_cols = ["income", "y"]   # income 是原始标签文本，y 是二元标签
    X_train = df_train.drop(columns=drop_cols)
    y_train = df_train[target_col].values
    
    X_test = df_test.drop(columns=drop_cols)
    y_test = df_test[target_col].values
    
    # 自动识别数值/类别列
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
    
    print("Numeric cols:", numeric_features)
    print("Categorical cols:", categorical_features)
    
    # ------------------------
    # 2) 预处理：数值标准化 + 类别 OneHot
    # ------------------------
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    
    # ------------------------
    # 3) Baseline：只预测训练集正类概率（frequency baseline）
    # ------------------------
    p_base = np.mean(y_train)  # 训练集里 y=1 的比例
    pred_base = np.full_like(y_test, fill_value=p_base, dtype=float)
    
    baseline_metrics = {
        "Model": "Baseline (freq)",
        "AUC": roc_auc_score(y_test, pred_base),               # 这里 AUC 会接近 0.5
        "LogLoss": log_loss(y_test, pred_base),
        "Brier": brier_score_loss(y_test, pred_base),
    }
    
    # ------------------------
    # 4) Ridge Logistic (L2)
    # ------------------------
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
    
    # ------------------------
    # 5) Lasso Logistic (L1)
    # 注意：L1 需要 solver='liblinear' 或 'saga'
    # ------------------------
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
    
    # ------------------------
    # 6) 汇总输出
    # ------------------------
    results = pd.DataFrame([baseline_metrics, ridge_metrics, lasso_metrics])
    print(results)


def main():
    df_train, df_test = data_design()
    machine_learning(df_train, df_test)
    
if __name__ == '__main__':
    main()
