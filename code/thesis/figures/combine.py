# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 01:05:57 2026

@author: jonat
"""

from PIL import Image
import os

filename = r"C:\Users\jonat\Desktop\论文\appendix\保存"

# === 1. 读取四张图片 ===
img1 = Image.open(os.path.join(filename, "exp3_tau0.05_q0.2_n500_p300_REPS200_RIDGE_1.png"))  # 左上
img2 = Image.open(os.path.join(filename, "exp3_tau0.05_q0.2_n500_p300_REPS200_RIDGE_2.png"))  # 右上
img3 = Image.open(os.path.join(filename, "exp3_tau0.025_q0.2_n2500_p1500_REPS200_RIDGE_1.png"))  # 左下
img4 = Image.open(os.path.join(filename, "exp3_tau0.025_q0.2_n2500_p1500_REPS200_RIDGE_2.png"))  # 右下

# === 2. 获取每张图的尺寸 ===
w1, h1 = img1.size
w2, h2 = img2.size
w3, h3 = img3.size
w4, h4 = img4.size

# 假设同一行图片高度一致，同一列宽度一致（通常成立）
top_height = max(h1, h2)
bottom_height = max(h3, h4)
left_width = max(w1, w3)
right_width = max(w2, w4)

# === 3. 创建新画布（比例保持不变）===
total_width = left_width + right_width
total_height = top_height + bottom_height

new_img = Image.new("RGB", (total_width, total_height), (255, 255, 255))

# === 4. 按四个角落粘贴 ===
new_img.paste(img1, (0, 0))                          # 左上
new_img.paste(img2, (left_width, 0))                 # 右上
new_img.paste(img3, (0, top_height))                 # 左下
new_img.paste(img4, (left_width, top_height))        # 右下

# === 5. 保存合并后的图片 ===
new_img.save("merged_figure.png", dpi=(300, 300))

print("合并完成，已保存为 merged_figure.png")