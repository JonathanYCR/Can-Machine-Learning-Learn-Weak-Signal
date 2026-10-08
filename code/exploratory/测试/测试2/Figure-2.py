import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_exam_main_logistic(path: str) -> pd.DataFrame:
    # 每行 6 列：tau q R2oos_lasso R2oos_ridge DeltaLasso DeltaRidge
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["tau", "q", "R2oos_lasso", "R2oos_ridge", "DeltaLasso", "DeltaRidge"],
        engine="python"
    )

    # 强制转成数值，转不了的变 NaN
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 删掉坏行（比如空行）
    df = df.dropna().reset_index(drop=True)

    return df

def table_like(df: pd.DataFrame, setups, zero_tol: float = 1e-12) -> pd.DataFrame:
    rows = []
    for (tau, q) in setups:
        sub = df[(np.isclose(df["tau"], tau)) & (np.isclose(df["q"], q))].copy()
        if sub.empty:
            raise ValueError(f"No rows found for (tau={tau}, q={q}).")

        def summarize(col: str):
            x = sub[col].to_numpy()
            q1, q2, q3 = np.quantile(x, [0.25, 0.5, 0.75])
            num_zero = int(np.sum(np.abs(x) <= zero_tol))
            return float(q1), float(q2), float(q3), num_zero

        L_q1, L_q2, L_q3, L_zero = summarize("DeltaLasso")
        R_q1, R_q2, R_q3, R_zero = summarize("DeltaRidge")

        rows.append(
            {
                "q": q,
                "tau": tau,
                "Lasso Q1": L_q1,
                "Lasso Q2": L_q2,
                "Lasso Q3": L_q3,
                "Lasso #Zero": L_zero,
                "Ridge Q1": R_q1,
                "Ridge Q2": R_q2,
                "Ridge Q3": R_q3,
                "Ridge #Zero": R_zero,
            }
        )
    return pd.DataFrame(rows)

def figure_like(df: pd.DataFrame, setups, bins: int = 20, xlim=(-2, 2), ylim=(0, 2.0), out_png="figure_logistic.png"):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2), constrained_layout=True)

    for j, (tau, q) in enumerate(setups):
        sub = df[(np.isclose(df["tau"], tau)) & (np.isclose(df["q"], q))]

        dr = sub["DeltaRidge"].to_numpy()
        dl = sub["DeltaLasso"].to_numpy()

        # Top: Ridge
        ax = axes[0, j]
        ax.hist(dr, bins=bins, density=True)
        ax.axvline(0.0, linestyle="--")
        ax.axvline(np.median(dr), linestyle="-")
        ax.set_title(rf"$\tau={tau},\, q={q}$")
        ax.set_xlim(xlim); ax.set_ylim(ylim)

        # Bottom: Lasso
        ax = axes[1, j]
        ax.hist(dl, bins=bins, density=True)
        ax.axvline(0.0, linestyle="--")
        ax.axvline(np.median(dl), linestyle="-")
        ax.set_title(rf"$\tau={tau},\, q={q}$")
        ax.set_xlim(xlim); ax.set_ylim(ylim)

    axes[0, 0].set_ylabel("Density (Ridge)")
    axes[1, 0].set_ylabel("Density (Lasso)")
    fig.suptitle("Simulation Results (Logistic DGP): Ridge vs Lasso", fontsize=12)
    fig.savefig(out_png, dpi=200)
    print(f"Saved: {out_png}")

def main(in_path="exam_main_logistic.txt"):
    df = load_exam_main_logistic(in_path)

    # 对齐论文 Figure 5 的“三组展示”：弱信号两组 + 强信号一组
    setups = [
        (0.2, 0.2),
        (0.2, 0.8),
        (1.0, 0.2),
    ]

    figure_like(df, setups, out_png="repro_figure_logistic.png")

    table = table_like(df, setups, zero_tol=1e-12)
    table.to_csv("repro_table_logistic.csv", index=False)
    print("Saved: repro_table_logistic.csv")
    print(table)

if __name__ == "__main__":
    main("exam_main_logistic.txt")
