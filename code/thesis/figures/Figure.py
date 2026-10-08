import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------
# Columns in your txt file
# -----------------------
COLS = [
    "R2", "q",
    "logloss_lasso", "logloss_ridge",   # already (logloss - log2)
    "mis_lasso", "mis_ridge",
    "DeltaLasso", "DeltaRidge"
]

# Paper-like cap, but panels will auto-zoom symmetrically around 0
XLIM_CAP = (-2, 2)

def load_results(path="weaksimu_revised/exam_main_logistic.txt"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
    for c in COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["R2", "q"]).copy()
    return df

def available_setups(df):
    g = df.groupby(["R2", "q"], dropna=True)
    setups = []
    for (R2, q), subdf in g:
        ok = subdf["logloss_ridge"].notna().any() or subdf["logloss_lasso"].notna().any()
        if ok:
            setups.append((float(R2), float(q)))
    setups.sort(key=lambda t: (t[0], t[1]))
    return setups

def sub(df, R2, q):
    return df[np.isclose(df["R2"], R2) & np.isclose(df["q"], q)].copy()

def auto_xlim_symmetric(vals, cap=XLIM_CAP, qlo=0.01, qhi=0.99, pad_ratio=0.15, min_halfwidth=0.15):
    """
    Symmetric x-limits around 0: [-a, a]
    a is derived from bulk quantiles + padding, then capped to [-2,2].
    """
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return cap

    lo, hi = np.quantile(vals, [qlo, qhi])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        a = min_halfwidth
    else:
        width = hi - lo
        pad = max(min_halfwidth, pad_ratio * width)
        a = max(abs(lo), abs(hi)) + pad

    a = max(a, min_halfwidth)
    a = min(a, cap[1])  # cap = (-2,2)
    return (-a, a)

def hist_panel(ax, vals, title, bins=60, zero_eps=1e-6, cap=XLIM_CAP):
    """
    - Histogram uses density=True for comparability across panels.
    - Near-zeros (|x|<=zero_eps) are drawn as an explicit spike at 0 so Lasso mass at 0 is visible.
    - x-axis is symmetric around 0 and auto-zooms to show trends (esp. R2=50% cases).
    - Only annotates P(<0), no #Zero printing.
    """
    ax.grid(False)

    if vals is None or len(vals) == 0:
        ax.set_xlim(*cap)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        return

    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        ax.set_xlim(*cap)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        return

    # Split near-zeros to force a visible spike at 0 (robust to bin edges)
    mask_zero = np.abs(vals) <= zero_eps
    zeros = vals[mask_zero]
    nonzero = vals[~mask_zero]

    # Symmetric adaptive x-limits around 0 (use nonzero if available, else all)
    base_for_xlim = nonzero if nonzero.size > 0 else vals
    x0, x1 = auto_xlim_symmetric(base_for_xlim, cap=cap)
    ax.set_xlim(x0, x1)

    # Histogram for nonzero part
    heights_max = 0.0
    binwidth = (x1 - x0) / bins if bins > 0 else 1.0

    if nonzero.size > 0:
        counts, edges, patches = ax.hist(nonzero, bins=bins, range=(x0, x1), density=True)
        if len(counts):
            heights_max = max(heights_max, float(np.max(counts)))

    # Explicit spike at 0 for zeros (density-consistent height)
    if zeros.size > 0:
        n = vals.size
        spike_height = (zeros.size / n) / max(binwidth, 1e-12)
        ax.bar(0.0, spike_height, width=binwidth, align="center", alpha=0.9)
        heights_max = max(heights_max, float(spike_height))

    # Vertical lines: mean (solid) and 0 (dashed)
    #ax.axvline(float(np.mean(vals)), linestyle="--", linewidth=1.2)
    ax.axvline(0.0, linestyle="--", linewidth=1.2, color="red")

    # Adaptive y-limit so spikes aren't clipped
    if heights_max <= 0:
        ax.set_ylim(0, 1)
    else:
        ax.set_ylim(0, 1.10 * heights_max)

    ax.set_title(title, fontsize=10)

    # Only annotate P(<0)
    #pneg = float(np.mean(vals < 0.0))
    #ax.text(0.02, 0.92, f"P(<0)={pneg:.3f}", transform=ax.transAxes, fontsize=9, va="top")

def make_figure(df, out_png="weaksimu_revised/figs/Figure_logloss.png", prefer_setups=None,
                bins=60, zero_eps=1e-6):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if prefer_setups is None:
        setups = available_setups(df)
    else:
        av = set(available_setups(df))
        setups = [s for s in prefer_setups if (float(s[0]), float(s[1])) in av]

    if len(setups) == 0:
        raise RuntimeError("No usable (R2,q) setups with logloss data found in the txt file.")

    ncols = len(setups)
    fig, axes = plt.subplots(2, ncols, figsize=(10.5, 5.8), constrained_layout=True)
    if ncols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for j, (R2, q) in enumerate(setups):
        d = sub(df, R2, q)
        ridge = d["logloss_ridge"].dropna().values
        lasso = d["logloss_lasso"].dropna().values

        title = f"$tau={int(round(R2*100))}\\%$, $q={q}$"

        # Top row: Ridge
        hist_panel(axes[0, j], ridge, title, bins=bins, zero_eps=zero_eps)
        # Bottom row: Lasso
        hist_panel(axes[1, j], lasso, title, bins=bins, zero_eps=zero_eps)

        # y-axis tick labels only on the leftmost column (paper-like)
        if j != 0:
            axes[0, j].set_yticklabels([])
            axes[1, j].set_yticklabels([])

    fig.suptitle("Simulation Results for Ridge and Lasso in Logistic DGPs", fontsize=12)

    #note = ("Note: The histograms depict the log-loss improvement relative to the zero predictor "
    #        "(i.e., logloss − log 2) across Monte Carlo samples. The dashed vertical line marks 0. "
    #        f"Near-zeros are treated as |x|≤{zero_eps:g} and shown as an explicit spike at 0.")
    #fig.text(0.02, -0.02, note, fontsize=9)

    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {out_png.resolve()}")

def make_table(df, out_csv="weaksimu_revised/figs/Table_logloss.csv",
               out_tex="weaksimu_revised/figs/Table_logloss.tex",
               prefer_setups=None,
               zero_eps=1e-6):
    """
    Table style:
    q, R2(%),
    Lasso(Q1,Q2,Q3,#Zero),
    Ridge(Q1,Q2,Q3,#Zero)
    (You can ignore #Zero later if you don't want it in the paper.)
    """
    if prefer_setups is None:
        setups = available_setups(df)
    else:
        av = set(available_setups(df))
        setups = [s for s in prefer_setups if (float(s[0]), float(s[1])) in av]

    rows = []

    for (R2, q) in setups:
        d = sub(df, R2, q)

        l = d["logloss_lasso"].dropna().values
        r = d["logloss_ridge"].dropna().values
        if len(l) == 0 and len(r) == 0:
            continue

        def qstats(x):
            if len(x) == 0:
                return (np.nan, np.nan, np.nan, np.nan)
            q1, q2, q3 = np.quantile(x, [0.25, 0.50, 0.75])
            zero_count = int(np.sum(np.abs(x) <= zero_eps))
            print(x.shape)
            return (q1, q2, q3, zero_count)

        q1_l, q2_l, q3_l, zero_l = qstats(l)
        q1_r, q2_r, q3_r, zero_r = qstats(r)

        rows.append({
            "q": q,
            "tau(%)": int(round(R2 * 100)),

            "Lasso Q1": q1_l,
            "Lasso Q2": q2_l,
            "Lasso Q3": q3_l,
            "Lasso #Zero": zero_l,

            "Ridge Q1": q1_r,
            "Ridge Q2": q2_r,
            "Ridge Q3": q3_r,
            "Ridge #Zero": zero_r,
        })

    tab = pd.DataFrame(rows).sort_values(["tau(%)", "q"], ascending=[True, True])

    out_csv = Path(out_csv)
    out_tex = Path(out_tex)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    tab.to_csv(out_csv, index=False)

    tex = tab.to_latex(
        index=False,
        float_format=lambda x: f"{x:.3f}",
        caption="Summary Statistics for Ridge and Lasso in Logistic DGPs (metric: logloss$-\\log 2$)",
        label="tab:logloss_summary"
    )
    out_tex.write_text(tex, encoding="utf-8")

    print("\nTable preview:\n", tab)
    print(f"\nSaved CSV  -> {out_csv.resolve()}")
    print(f"Saved TEX  -> {out_tex.resolve()}")

if __name__ == "__main__":
    df = load_results("weaksimu_revised/exam_main_logistic.txt")

    # Keep these three setups; missing ones will be skipped
    prefer = [(0.05, 0.2), (0.05, 0.8), (0.5, 0.2)]

    make_figure(
        df,
        out_png="weaksimu_revised/figs/Figure_logloss.png",
        prefer_setups=prefer,
        bins=60,
        zero_eps=1e-6
    )

    make_table(
        df,
        out_csv="weaksimu_revised/figs/Table_logloss.csv",
        out_tex="weaksimu_revised/figs/Table_logloss.tex",
        prefer_setups=prefer,
        zero_eps=1e-6
    )
