# Weak-Signal Learning in Binary Logistic Models

二元 Logistic 模型下的弱信号学习：Monte Carlo 模拟、Ridge / Lasso / Ridgeless 比较，以及 Finance1、Macro1 实证分析。

## 文件结构

```text
.
├── README.md
├── code/
│   ├── thesis/
│   │   ├── simulation/       # Logistic CV 和固定惩罚参数模拟
│   │   ├── empirical/        # Finance1 / Macro1 的六个模型及汇总表程序
│   │   └── figures/          # 直方图、统计表和图片拼接程序
│   ├── exploratory/          # 原有探索代码，按原相对路径保存
│   └── reference/
│       ├── monte_carlo/      # 参考项目的 Python / R 模拟代码
│       └── empirical/        # NN、RF、XGBoost 和 MATLAB 数据构造代码
├── data/
│   ├── thesis/               # 两个论文实证数据文件
│   ├── benchmarks/           # Adult、Bank、USREC / FRED 数据
│   ├── reference/            # 参考项目 MAT、CSV、XLSX 等输入
│   └── local/finance2/       # 大型 Finance2 输入，仅本地保存
└── results/
    ├── master/
    │   ├── figures/          # 三月论文的四张最终图
    │   └── tables/           # 两张表的原有 CSV / LaTeX 文件
    └── supporting/
        ├── panels/          # 最终拼图的六个组成图，也包含一月稿图 1、2
        ├── simulation/      # 图 1 / 表 1 使用的逐次模拟结果
        └── empirical/       # 表 2 使用的六份模型统计结果
```

共保留 188 个代码文件（80 Python、106 MATLAB、2 R）、741 个数据文件、21 个结果及支撑文件。`.txt` 中保留的是数值结果或结构化输入数据；`.tex` 中保留的是结果表格，不是论文正文。

## 论文图表与文件对应

| 论文位置 | 保留文件（相对于 `results/`） |
| --- | --- |
| 三月稿 Figure 1，PDF 第 23 页 | `master/figures/Figure_logloss.png` |
| 三月稿 Figure 2，PDF 第 24 页 | `master/figures/merged_side_by_side_1.png` |
| 三月稿 Figure 3，PDF 第 25 页 | `master/figures/merged_side_by_side_2.png` |
| 三月稿 Figure 4，PDF 第 26 页 | `master/figures/merged_side_by_side_3.png` |
| 三月稿 Table 1 | `master/tables/Table_logloss.csv`、同名 `.tex`；存在下述计数差异 |
| 三月稿 Table 2 | `master/tables/Logistic_Comparison_Table.csv`、同名 `.tex` |
| 一月稿图 1，PDF 第 16 页 | `supporting/panels/exp3_tau0.05_q0.2_n500_p300_REPS200_RIDGE_1.png` |
| 一月稿图 2，PDF 第 17 页 | `supporting/panels/exp3_tau0.05_q0.2_n500_p300_REPS200_RIDGE_2.png` |

以上六张论文插图均与 PDF 内图片逐像素匹配。最终组合图的六个组成图保留在 `supporting/panels/`，避免重复保存同一文件。

一月稿 PDF 第 18 页的 Lasso 图没有找到逐像素一致的独立文件。按照整理要求，未从 PDF 新增提取图片；保留的 Lasso 组成图属于三月稿，不能视为该旧图的精确副本。

## 程序入口

| 程序 | 用途 |
| --- | --- |
| `code/thesis/simulation/test.py` | CV 选择惩罚参数的 Logistic 模拟，函数 `main_logistic` 返回逐次结果 |
| `code/thesis/simulation/Fixed_lambda.py` | 给定 lambda 网格的 Ridge / Lasso 模拟，函数 `main_logistic_experiment3` |
| `code/thesis/simulation/Fixed_lambda_lasso.py` | 给定 C 网格的 Lasso 模拟，函数 `main_logistic_lasso_experiment3_fixedC` |
| `code/thesis/figures/Figure.py` | 读取已保存模拟结果，绘制图 1 并生成表 1 |
| `code/thesis/empirical/Finance1_*.py` | 股票市场方向预测，分别运行 ridge、lasso、ridgeless |
| `code/thesis/empirical/Macro1_*.py` | 工业生产增长方向预测，分别运行 ridge、lasso、ridgeless |
| `code/thesis/empirical/table.py` | 读取六份模型结果，生成表 2 |
| `code/thesis/figures/combine*.py` | 原有图片拼接程序，含旧绝对路径 |

论文模拟主程序原位于 `appendix/保存/test.py`；实证程序和汇总程序原位于 `appendix/保存/`。探索代码不是自动化测试套件，名称含 `test` 的脚本可能执行耗时模拟。

## 环境准备

主要依赖为 NumPy、SciPy、pandas、scikit-learn、Matplotlib、Pillow、joblib、Jinja2。原项目没有锁定依赖版本。以下给出适用于旧 API 的兼容环境安装方案；该完整环境未在此次整理中重新安装验证。

在仓库根目录打开 PowerShell，使用 Python 3.11 或 3.12 建立环境。例如已安装 Python 3.12 时：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install "numpy>=1.26,<2" "scipy>=1.11,<1.15" "pandas>=2.1,<3" "scikit-learn>=1.3,<1.6" "matplotlib>=3.8,<3.10" "Pillow>=10,<12" joblib jinja2
```

代码使用 `boxplot(labels=...)`、`LogisticRegression(penalty=...)` 等旧接口，因此不建议直接用所有依赖的最新版本。Jinja2 用于 pandas 导出 LaTeX 表格。

下面的 PowerShell 代码块均从仓库根目录执行。新计算结果放入 `generated/`，论文归档结果位于 `results/`。加载原程序后，只在内存中设置输入、输出路径，不编辑源文件。

## 用已有结果重绘图 1 和生成表 1

无需重新进行 Monte Carlo 模拟：

```powershell
@'
from pathlib import Path
import os, runpy
os.environ["MPLBACKEND"] = "Agg"
root = Path.cwd()
out = root / "generated" / "simulation"
out.mkdir(parents=True, exist_ok=True)
m = runpy.run_path(str(root / "code/thesis/figures/Figure.py"))
df = m["load_results"](root / "results/supporting/simulation/exam_main_logistic.txt")
setups = [(0.05, 0.2), (0.05, 0.8), (0.5, 0.2)]
m["make_figure"](df, out_png=out / "Figure_logloss.png", prefer_setups=setups)
m["make_table"](df, out_csv=out / "Table_logloss.csv",
                out_tex=out / "Table_logloss.tex", prefer_setups=setups)
'@ | python -B -
```

## 用已有实证结果生成表 2

将六份归档结果复制到新输出目录，再运行原汇总程序：

```powershell
@'
from pathlib import Path
import os, runpy, shutil
root = Path.cwd()
out = root / "generated" / "empirical_table"
out.mkdir(parents=True, exist_ok=True)
for p in (root / "results/supporting/empirical").glob("*.txt"):
    shutil.copy2(p, out / p.name)
os.chdir(out)
runpy.run_path(str(root / "code/thesis/empirical/table.py"), run_name="__main__")
'@ | python -B -
```

## 重新运行论文实证模型

以下示例运行 Finance1 Ridge。将 `dataset` 改为 `Macro1`，或将 `model` 改为 `lasso` / `ridgeless`，可运行其他模型。交叉验证和滚动预测可能较耗时。

```powershell
@'
from pathlib import Path
import importlib.util
root = Path.cwd()
dataset, model = "Finance1", "ridge"
source = root / "code/thesis/empirical" / f"{dataset}_{model}.py"
spec = importlib.util.spec_from_file_location("experiment", source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
data_file = {"Finance1": "Goyal_monthly.mat", "Macro1": "FredMDlargeHor1.mat"}
m.MAT_PATH = str(root / "data/thesis" / data_file[dataset])
out = root / "generated" / "empirical_models"
out.mkdir(parents=True, exist_ok=True)
m.OUT_PATH = str(out / Path(m.OUT_PATH).name)
m.main()
'@ | python -B -
```

输入数据的实际维度：`Goyal_monthly.mat` 的 X 为 888 × 14；`FredMDlargeHor1.mat` 的 X 为 743 × 120。Macro1 程序删除其中一列后使用 119 个预测变量。

## 重新运行 Monte Carlo 模拟

CV 模拟程序的原有文件写入语句已被注释，直接执行脚本不会保存其返回数组。下面显式保存函数返回值，并按现有归档每组 100 次的设置运行：

```powershell
@'
from pathlib import Path
import runpy, numpy as np
root = Path.cwd()
out = root / "generated" / "cv_simulation"
out.mkdir(parents=True, exist_ok=True)
m = runpy.run_path(str(root / "code/thesis/simulation/test.py"))
rows = []
for r2 in [0.05, 0.5]:
    for q in [0.2, 0.8]:
        rows.append(m["main_logistic"](R2=r2, sparsity=q, n=500, p=300, REPS=100))
np.savetxt(out / "exam_main_logistic.txt", np.vstack(rows), fmt="%.10g")
'@ | python -B -
```

固定参数模拟示例运行原脚本默认网格。大样本设置和 200 次重复计算量较大；输出文件名及图片样式不保证与历史归档完全相同。

```powershell
@'
from pathlib import Path
import os, runpy
os.environ["MPLBACKEND"] = "Agg"
root = Path.cwd()
out = root / "generated" / "fixed_tuning"
out.mkdir(parents=True, exist_ok=True)
os.chdir(out)
ridge = runpy.run_path(str(root / "code/thesis/simulation/Fixed_lambda.py"))
lasso = runpy.run_path(str(root / "code/thesis/simulation/Fixed_lambda_lasso.py"))
ridge["main_logistic_experiment3"](
    tau_n=0.025, sparsity=0.2, n=2500, p=1500, REPS=200)
lasso["main_logistic_lasso_experiment3_fixedC"](
    tau_n=0.05, sparsity=0.2, n=500, p=300, REPS=200)
'@ | python -B -
```

Ridge 函数可通过 `lambdas` 指定惩罚网格，Lasso 函数可通过 `C_lambdas` 指定网格。历史图中的小 lambda、大 lambda 和两种样本量来自不同设置；原脚本只保存了当前默认配置。

## 原有差异与复现边界

- 表 1：归档模拟文件共有 400 行，四组各 100 次；CSV 的 Lasso `#Zero` 为 72、78、27，而三月论文为 144、156、54。分位数与论文三位小数一致。此次没有补算、翻倍或改写结果。
- 表 2：归档六份统计结果与论文数值对应。当前 `Macro1_lasso.py` 输出文件名带 `_new`，其先前运行结果与论文采用的旧结果不同；重新运行当前脚本不能保证恢复旧结果。
- Macro1 Ridge 和 Ridgeless 当前代码的 `STEP=12`，即使输出文件名包含 `step1`；Macro1 Lasso 的 `STEP=1`。README 中运行示例保留各程序原参数。
- Finance1 输入实际含 14 列 X，论文文字写了 16 个预测变量；数据保持原样。
- 固定参数图保存了最终 PNG，但没有完整对应的逐次模拟数组和全部历史配置。最终图可以直接使用，重新计算不能保证逐像素一致。
- 原代码中的旧绝对路径、参考项目的缺失运行目录和历史 API 均保留。上述论文入口示例处理了主要数据路径；探索和参考代码需要按各自读写路径单独准备运行环境。

## 探索与参考代码

`code/exploratory/` 保存 Adult、Bank、线性模型、不同 Logistic 设定及绘图尝试；未纳入论文的旧输出已删除。仅绘图的探索脚本需要先重新生成其输入。

`code/reference/empirical/` 中 NN、RF、XGBoost 程序通常要求一个位置整数参数。其含义由各脚本决定，不能统一视为重复次数。部分脚本读写 `weak/`、`Macro1data/`、`Micro1data/`、`Micro2data/`、`Macro2data/` 或模型输出目录；个别脚本还写死了其他作者的工作目录。因此这些代码作为参考源码保存，没有声明可以在当前根目录直接运行。

参考输入集中在 `data/reference/`，保留原嵌套路径；大型 Finance2 输入位于 `data/local/finance2/`，使用时对应脚本原来的 `data1957-2022/`。NN / XGBoost 代码还需要匹配历史版本的 TensorFlow、Keras、XGBoost；部分探索代码需要 Python `glmnet`。MATLAB `.m` 程序需要 MATLAB 及相应工具箱；R 程序的依赖以其 `library(...)` 声明为准。这些扩展环境不包含在上面的论文 Python 环境中。

数据来源沿用原项目：Finance1 为 Welch–Goyal 预测变量数据，Macro1 为 FRED-MD；其他输入包括 Adult、Bank Marketing、USREC、增长数据及原参考项目的微观数据。本次整理没有新增数据授权或许可证。

## 上传 GitHub

`data/local/finance2/` 约 6.46 GB，其中 `data.csv` 约 5.69 GB，仅在本地保留，不加入普通 Git 提交。论文主实验只需要 `data/thesis/`；参考项目与大型数据可分别管理。

在确认仓库名称和远程地址后，可以先准备论文主项目的本地提交：

```powershell
git init
git add README.md code results data/thesis
git status --short
git commit -m "Organize thesis code and results"
```

如需一并提供较小的探索和参考输入，再单独执行：

```powershell
git add data/benchmarks data/reference
git status --short
git commit -m "Add benchmark and reference inputs"
```

采用明确列出的路径暂存文件，不使用 `git add .`，以免把大型本地数据、`.venv/` 或 `generated/` 一并提交。本次整理未初始化 Git、创建远程仓库或上传任何文件。

## 整理时的验证

移动前后对 950 个保留文件核对 SHA-256，内容一致。80 个 Python 文件和 README 中五段 Python 示例通过语法解析；论文六个实证模块可导入，两份 MAT 输入可读取；图 1 可由保存的模拟记录在内存中重绘。表 1 的全部统计量与保留的模拟输入一致，表 2 的 12 个数值与原汇总程序从六份统计结果计算的数值一致。未执行完整 Monte Carlo、全部滚动交叉验证、MATLAB / R 或参考机器学习模型训练，也未将新计算结果覆盖到论文归档。
