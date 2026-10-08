import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# 文件路径
# ==============================
file_path = "macro1_realrecession_logistic_ridge_step12.txt"

# ==============================
# 指定列名（必须和保存时一致）
# ==============================
colnames = [
    "train_end",
    "model_logloss",
    "baseline_logloss",
    "excess_vs_null",
    "pi0",
    "best_C"
]

# 读取数据
df = pd.read_csv(
    file_path,
    sep=r"\s+",
    comment="#",   # 忽略 np.savetxt 自动加的 header 行
    header=None,
    names=colnames
)

# ==============================
# 打印统计信息
# ==============================
means = df.mean(numeric_only=True)

print("Column Means:\n", means)
print("\nAverage excess_vs_null:", means["excess_vs_null"])
print("Median  excess_vs_null:", df["excess_vs_null"].median())
print("Share (excess < 0):", np.mean(df["excess_vs_null"] < 0))

# ==============================
# 画图1：和你原程序一致
# log10(pi0) & log10(best_C)
# ==============================
plt.figure(figsize=(12, 6))

# 左图：log10(pi0)
plt.subplot(1, 2, 1)
plt.hist(np.log10(df["pi0"]), bins=20,
         color='blue', edgecolor='black')
plt.title('Histogram of log10(pi0)')
plt.xlabel('log10(pi0)')
plt.ylabel('Frequency')

# 右图：log10(best_C)
plt.subplot(1, 2, 2)
plt.hist(np.log10(df["best_C"]), bins=20,
         color='green', edgecolor='black')
plt.title('Histogram of log10(best_C)')
plt.xlabel('log10(best_C)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# ==============================
# 画图2（推荐）：Excess Log-Loss 分布
# ==============================
plt.figure(figsize=(6, 5))
plt.hist(df["excess_vs_null"], bins=20,
         edgecolor='black')

plt.axvline(0, linestyle='--')
plt.title("Histogram of Excess Log-Loss (Model - Null)")
plt.xlabel("excess_vs_null")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# ==============================
# 时间路径图：Excess Log-Loss
# ==============================
plt.figure(figsize=(10, 5))

plt.plot(df["train_end"], df["excess_vs_null"], marker='o')

plt.axhline(0, linestyle='--')

plt.title("Time Path of Excess Log-Loss (Ridge vs Null)")
plt.xlabel("Training Sample End (Time Index)")
plt.ylabel("Excess Log-Loss")

plt.tight_layout()
plt.show()