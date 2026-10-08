# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 01:27:46 2026

@author: jonat
"""

from PIL import Image
import os

filename = r"C:\Users\jonat\Desktop\论文\appendix\保存"

# === 1. 读取两张图片 ===
img_left = Image.open(os.path.join(filename, "exp3_lasso_fixedC_tau0.05_q0.2_n500_p300_REPS200_LASSO_BOX.png"))  # 左上
img_right = Image.open(os.path.join(filename, "exp3_lasso_fixedC_tau0.025_q0.2_n2500_p1500_REPS200_LASSO_BOX.png"))  # 右上

# === 2. 获取尺寸 ===
w1, h1 = img_left.size
w2, h2 = img_right.size

# 新图高度取两张图中较大的高度
new_height = max(h1, h2)
new_width = w1 + w2

# === 3. 创建白色背景画布 ===
new_img = Image.new("RGB", (new_width, new_height), (255, 255, 255))

# === 4. 粘贴图片（顶部对齐）===
new_img.paste(img_left, (0, 0))
new_img.paste(img_right, (w1, 0))

# === 5. 保存 ===
new_img.save("merged_side_by_side.png", dpi=(300, 300))

print("合并完成，已保存为 merged_side_by_side.png")