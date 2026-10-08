# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 00:49:12 2026

@author: jonat
"""

import os
import re
import pandas as pd

# ==================================================
# 1️⃣  填你自己的文件路径
# ==================================================
FILES = {
    "Finance1": {
        "Ridge":      r"finance1_logistic_ridge_step1.txt",
        "Lasso":      r"finance1_logistic_lasso_step1.txt",
        "Ridgeless":  r"finance1_logistic_ridgeless_step1.txt",
    },
    "Macro1": {
        "Ridge":      r"macro1_logistic_ridge_step1.txt",
        "Lasso":      r"macro1_logistic_lasso_step1.txt",
        "Ridgeless":  r"macro1_logistic_ridgeless_step1.txt",
    }
}

# 输出位置
OUT_DIR = "./"
OUT_TEX = os.path.join(OUT_DIR, "Logistic_Comparison_Table.tex")
OUT_CSV = os.path.join(OUT_DIR, "Logistic_Comparison_Table.csv")


# ==================================================
# 2️⃣  解析函数
# ==================================================
FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def extract_first_float(line):
    m = FLOAT_RE.search(line)
    return float(m.group()) if m else None

def parse_metrics(path):
    mcfadden = None
    bss = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            low = line.lower()

            # McFadden R²（你存的是百分比）
            if mcfadden is None and (
                "mcfadden" in low or "logloss_improve" in low
            ):
                mcfadden = extract_first_float(line)

            # Brier Skill Score（百分比）
            if bss is None and (
                "brier" in low and "improve" in low
                or "brier skill" in low
            ):
                bss = extract_first_float(line)

    if mcfadden is None or bss is None:
        raise ValueError(f"Cannot parse file: {path}")

    return mcfadden, bss


# ==================================================
# 3️⃣  构建表格
# ==================================================
columns = pd.MultiIndex.from_tuples([
    ("Ridge", "McFadden R$^2$ (%)"),
    ("Ridge", "Brier Skill Score (%)"),
    ("Lasso", "McFadden R$^2$ (%)"),
    ("Lasso", "Brier Skill Score (%)"),
    ("Ridgeless", "McFadden R$^2$ (%)"),
    ("Ridgeless", "Brier Skill Score (%)"),
])

rows = []
index = []

for dataset, models in FILES.items():
    row = []
    for model in ["Ridge", "Lasso", "Ridgeless"]:
        mcf, bss = parse_metrics(models[model])
        row.extend([mcf, bss])
    rows.append(row)
    index.append(dataset)

df = pd.DataFrame(rows, index=index, columns=columns)

# 保留三位小数
df_formatted = df.round(3)

print("\n=== Final Table ===")
print(df_formatted)

# 保存 CSV
df.to_csv(OUT_CSV)

# 保存 LaTeX（带分组表头）
latex = df.to_latex(
    float_format="%.3f",
    multicolumn=True,
    multicolumn_format="c",
    escape=False
)

with open(OUT_TEX, "w", encoding="utf-8") as f:
    f.write(latex)

print("\nSaved to:")
print(OUT_CSV)
print(OUT_TEX)