import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ======================
# 1) 读取数据
# ======================
# 你的输出文件路径（按需改）
DATA_PATH = "weaksimu_revised\exam_main.txt"

# 你希望画在 Figure 5 里的三个 setup（按论文：R2=5%, q=0.2 / 0.8；R2=50%, q=0.2）
# 如果你的文件里没有这些组合，对应子图就会显示 "No data"
SETUPS = [
    (0.05, 0.2),
    (0.05, 0.8),
    (0.5, 0.2),
]

# 输出目录
OUT_DIR = Path("weaksimu_revised\outputs")
OUT_DIR.mkdir(exist_ok=True)

# 读入：文件无表头、空格分隔
df = pd.read_csv(
    DATA_PATH,
    sep=r"\s+",
    header=None,
    names=["R2", "q", "R2oos_lasso", "R2oos_ridge", "DeltaLasso", "DeltaRidge"],
)

# 处理浮点比较：用四舍五入避免 0.0500000001 这种问题
df["R2"] = df["R2"].round(6)
df["q"] = df["q"].round(6)

print("Loaded rows:", len(df))
print("Unique (R2,q) groups:", df[["R2", "q"]].drop_duplicates().sort_values(["R2", "q"]).to_string(index=False))


# ======================
# 2) 画 Figure 5 风格的图
# ======================
def plot_hist(ax, values, title):
    """
    画直方图 + 0 参考线（蓝色实线）+ 中位数线（红色虚线）
    """
    values = np.asarray(values)

    if values.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10)
        ax.set_xlim(-2, 2)
        ax.set_ylim(0, 2.0)
        ax.set_title(title, fontsize=10)
        return

    # 论文图看起来是 density（y 轴到 2.0），这里用 density=True 模拟
    ax.hist(values, bins=50, density=True)

    # 蓝色实线：0 参考线
    ax.axvline(0.0, linestyle="-", linewidth=1.5)

    # 红色虚线：中位数（Q2）
    q2 = float(np.quantile(values, 0.5))
    ax.axvline(q2, linestyle="--", linewidth=1.5)

    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 2.0)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="both", labelsize=8)


fig, axes = plt.subplots(2, 3, figsize=(10, 5.8))

for col, (r2, q) in enumerate(SETUPS):
    sub = df[(df["R2"] == round(r2, 6)) & (df["q"] == round(q, 6))]

    # 标题格式按论文：R^2=5%, q=0.2
    r2_pct = int(round(r2 * 100))
    title = rf"$R^2={r2_pct}\%,\, q={q}$"

    # 上排：Lasso 的 Delta
    plot_hist(axes[0, col], sub["DeltaLasso"].values, title)

    # 下排：Ridge 的 Delta
    plot_hist(axes[1, col], sub["DeltaRidge"].values, title)

# 总标题（可选）
fig.suptitle("Simulation Results for Ridge and Lasso (Linear DGPs)", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])

fig_path = OUT_DIR / "figure5_like.png"
plt.savefig(fig_path, dpi=200)
print(f"Saved figure to: {fig_path}")


# ======================
# 3) 做 Table 1 风格的汇总表
# ======================
def summarize_group(values, zero_tol=0.0):
    """
    返回 Q1/Q2/Q3 以及 #Zero（Delta == 0 的次数）
    zero_tol=0.0 表示严格等于 0；
    如果你担心浮点误差，可以用 zero_tol=1e-12。
    """
    values = np.asarray(values)
    if values.size == 0:
        return (np.nan, np.nan, np.nan, 0)

    q1 = float(np.quantile(values, 0.25))
    q2 = float(np.quantile(values, 0.50))
    q3 = float(np.quantile(values, 0.75))
    if zero_tol == 0.0:
        n_zero = int(np.sum(values == 0.0))
    else:
        n_zero = int(np.sum(np.abs(values) <= zero_tol))
    return (q1, q2, q3, n_zero)

rows = []
for (r2, q), sub in df.groupby(["R2", "q"], sort=True):
    l_q1, l_q2, l_q3, l_zero = summarize_group(sub["DeltaLasso"].values, zero_tol=0.0)
    r_q1, r_q2, r_q3, r_zero = summarize_group(sub["DeltaRidge"].values, zero_tol=0.0)

    rows.append({
        "q": q,
        "R2(%)": int(round(r2 * 100)),
        "Lasso_Q1": l_q1,
        "Lasso_Q2": l_q2,
        "Lasso_Q3": l_q3,
        "Lasso_#Zero": l_zero,
        "Ridge_Q1": r_q1,
        "Ridge_Q2": r_q2,
        "Ridge_Q3": r_q3,
        "Ridge_#Zero": r_zero,
        "N": len(sub),
    })

table = pd.DataFrame(rows).sort_values(["q", "R2(%)"]).reset_index(drop=True)

# 论文表格通常保留 3 位小数
pd.set_option("display.float_format", lambda x: f"{x: .3f}")

print("\nTable 1-like summary:")
print(table.to_string(index=False))

table_path = OUT_DIR / "table1_like.csv"
table.to_csv(table_path, index=False)
print(f"\nSaved table to: {table_path}")
