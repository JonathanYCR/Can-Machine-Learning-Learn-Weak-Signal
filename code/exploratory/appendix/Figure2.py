# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 06:36:14 2026

@author: jonat
"""

# -*- coding: utf-8 -*-
# Read exam_logistic_fixed.txt and plot Appendix A.1 style Figure A1 (Ridge fixed lambda)

import numpy as np
import matplotlib.pyplot as plt


def alpha_star(lam: float) -> float:
    """
    Appendix A.1 simplified setting (Sigma2=I, theta1=theta2=sigma_x=sigma_beta=sigma_eps=1):
    alpha*(lam) = 1/lam^2 - 2/lam
    """
    return 1.0/(lam**2) - 2.0/lam


def load_data(path: str) -> np.ndarray:
    """
    Expect each row:
      R2 sparsity n p lambda Delta
    """
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 6:
        raise ValueError("Data file must have at least 6 columns: R2 sparsity n p lambda Delta")
    return data


def plot_figure_A1_from_file(
    path="exam_logistic_fixed.txt",
    # filters: set these to match the exact A.1 scenario you want to display
    sparsity_target=0.2,
    n_list=(500, 2500),
    R2_by_n={500: 0.05, 2500: 0.025},
    lambda_list=(0.5, 1.0, 2.0),
    out_png="Figure_A1_from_exam_logistic_fixed.png",
):
    data = load_data(path)

    # columns
    R2_col = data[:, 0]
    q_col  = data[:, 1]
    n_col  = data[:, 2].astype(int)
    lam_col = data[:, 4]
    delta_col = data[:, 5]

    fig, axes = plt.subplots(2, 3, figsize=(10, 5), constrained_layout=True)

    for r, n in enumerate(n_list):
        R2_target = R2_by_n.get(n, None)

        for c, lam in enumerate(lambda_list):
            ax = axes[r, c]

            mask = (n_col == n) & (np.isclose(lam_col, lam))

            # optional: filter sparsity q
            if sparsity_target is not None:
                mask &= np.isclose(q_col, sparsity_target)

            # optional: filter R2 (if present for that n)
            if R2_target is not None:
                mask &= np.isclose(R2_col, R2_target)

            vals = delta_col[mask]

            if vals.size == 0:
                ax.text(0.5, 0.5, "No data\n(check filters)",
                        ha="center", va="center", transform=ax.transAxes)
            else:
                ax.hist(vals, bins=60, density=True)
                ax.axvline(alpha_star(lam), linestyle="--", color="red")  # red dashed line like paper

            ax.set_title(f"λ = {lam}, n = {n}")
            ax.set_xlim(-2, 2)
            ax.set_ylim(0, 2.0)

    fig.suptitle("Figure A1: Simulation Results for Ridge with Fixed Tuning Parameters", y=1.02)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved figure to: {out_png}")


if __name__ == "__main__":
    plot_figure_A1_from_file()
