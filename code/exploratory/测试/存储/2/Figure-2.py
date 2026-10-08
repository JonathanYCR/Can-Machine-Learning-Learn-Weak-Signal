# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 03:25:57 2026

@author: jonat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_exam_main(path: str) -> pd.DataFrame:
    # 文件每行 6 列：R2 q R2oos_lasso R2oos_ridge DeltaLasso DeltaRidge
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["R2", "q", "R2oos_lasso", "R2oos_ridge", "DeltaLasso", "DeltaRidge"],
    )
    #df[["DeltaLasso", "DeltaRidge"]] = df[["DeltaLasso", "DeltaRidge"]] * 4
    return df

def table1_like(df: pd.DataFrame, setups, zero_tol: float = 1e-12) -> pd.DataFrame:
    """
    Table 1: for each (R2,q), compute Q1/Q2/Q3 and #Zero for Lasso/Ridge.
    In paper, #Zero corresponds to Delta==0 (i.e., predicts as zero baseline).
    We use |Delta|<=tol to be robust to floating error.
    """
    rows = []
    for (R2, q) in setups:
        sub = df[(np.isclose(df["R2"], R2)) & (np.isclose(df["q"], q))].copy()
        if sub.empty:
            raise ValueError(f"No rows found for (R2={R2}, q={q}). Check file or filters.")

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
                "R2(%)": int(round(R2 * 100)),
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

    out = pd.DataFrame(rows)
    return out

def figure5_like(df: pd.DataFrame, setups, bins: int = 20, xlim=(-1, 1), ylim=(0, 5.0), out_png="figure5.png"):
    """
    Figure 5: 2x3 histograms
      top: Delta(Ridge)
      bottom: Delta(Lasso)
    with vertical line at 0 (baseline). Another line can be median if desired.
    """
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2), constrained_layout=True)

    for j, (R2, q) in enumerate(setups):
        sub = df[(np.isclose(df["R2"], R2)) & (np.isclose(df["q"], q))]

        dr = sub["DeltaRidge"].to_numpy()
        dl = sub["DeltaLasso"].to_numpy()

        # Top row: Ridge
        ax = axes[0, j]
        ax.hist(dr, bins=bins, density=True)
        ax.axvline(0.0, linestyle="--")         # baseline
        ax.axvline(np.median(dr), linestyle="-")# solid line (median)
        ax.set_title(rf"$R^2={int(round(R2*100))}\%,\, q={q}$")
        ax.set_xlim(xlim); ax.set_ylim(ylim)

        # Bottom row: Lasso
        ax = axes[1, j]
        ax.hist(dl, bins=bins, density=True)
        ax.axvline(0.0, linestyle="--")
        ax.axvline(np.median(dl), linestyle="-")
        ax.set_title(rf"$R^2={int(round(R2*100))}\%,\, q={q}$")
        ax.set_xlim(xlim); ax.set_ylim(ylim)

    axes[0, 0].set_ylabel("Density (Ridge)")
    axes[1, 0].set_ylabel("Density (Lasso)")
    fig.suptitle("Figure 5: Simulation Results for Ridge and Lasso in Linear DGPs", fontsize=12)

    fig.savefig(out_png, dpi=200)
    print(f"Saved: {out_png}")

def main(in_path="exam_main.txt"):
    df = load_exam_main(in_path)
    tau_weak = 0.05/(1-0.05) 
    tau_strong = 0.5/(1-0.5) 

    # Paper Figure 5 / Table 1 uses three setups:
    setups = [
        (tau_weak, 0.2),
        (tau_weak, 0.8),
        (tau_strong, 0.2),
    ]

    # Figure 5
    figure5_like(df, setups, out_png="repro_figure5-2.png")

    # Table 1
    table = table1_like(df, setups, zero_tol=1e-6)
    table.to_csv("repro_table1-2.csv", index=False)
    print("Saved: repro_table1-2.csv")
    print(table)

if __name__ == "__main__":
    main("exam_main_logistic.txt")
