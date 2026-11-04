import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cbam import load_cbam_model
from grade_severity import segment_lesion

MODEL_PATH = r"D:\python\TomatoDiseaseProject\runs\detect\runs\detect\cbam_s\weights\best.pt"
IMG_DIR = r"D:\python\TomatoDiseaseProject\datasets\tomato\images\val"
LBL_DIR = r"D:\python\TomatoDiseaseProject\datasets\tomato\labels\val"

model = load_cbam_model(MODEL_PATH)

# 只统计A组（病斑型）病害
A_GROUP = ["bacterialspot", "earlyblight", "lateblight",
           "septorialeafspot", "targetspot"]

ratios = []
for fname in os.listdir(IMG_DIR)[:500]:   # 先跑500张，够用
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    img = cv2.imread(os.path.join(IMG_DIR, fname))
    if img is None:
        continue
    res = model.predict(img, conf=0.25, verbose=False)[0]
    if len(res.boxes) == 0:
        continue
    best = int(res.boxes.conf.argmax())
    cls_name = model.names[int(res.boxes.cls[best])]
    if not any(k in cls_name.lower() for k in A_GROUP):
        continue

    x1, y1, x2, y2 = map(int, res.boxes.xyxy[best])
    h, w = img.shape[:2]
    crop = img[max(y1-10,0):min(y2+10,h), max(x1-10,0):min(x2+10,w)]
    if crop.size == 0:
        continue
    _, _, ratio = segment_lesion(crop, cls_name)
    ratios.append(ratio)

print(f"有效样本：{len(ratios)} 张")
print(f"ratio 分布：min={min(ratios):.4f}, max={max(ratios):.4f}, "
      f"mean={np.mean(ratios):.4f}, median={np.median(ratios):.4f}")

# 画直方图
plt.figure(figsize=(10, 5))
plt.hist(ratios, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Symptom Coverage Ratio')
plt.ylabel('Count')
plt.title('Ratio Distribution on Validation Set (A-group diseases)')
plt.axvline(x=0.05, color='g', linestyle='--', label='T1=0.05')
plt.axvline(x=0.25, color='r', linestyle='--', label='T2=0.25')
plt.legend()
plt.tight_layout()
plt.savefig(r"D:\python\TomatoDiseaseProject\outputs\ratio_distribution.png", dpi=150)
plt.show()

# 输出分位数
for q in [10, 25, 50, 75, 90]:
    print(f"  {q}% 分位数：{np.percentile(ratios, q):.4f}")