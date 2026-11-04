import os
from collections import Counter

LABEL_DIR = r"D:\python\TomatoDiseaseProject\datasets\tomato\labels\train"

CLASS_NAMES = [
    "Tomato__BacterialSpot",
    "Tomato__EarlyBlight",
    "Tomato__Healthy",
    "Tomato__LateBlight",
    "Tomato__LeafMold",
    "Tomato__MosaicVirus",
    "Tomato__SeptoriaLeafSpot",
    "Tomato__SpiderMites",
    "Tomato__TargetSpot",
    "Tomato__YellowLeafCurlVirus",
]

counter = Counter()
for fname in os.listdir(LABEL_DIR):
    if not fname.endswith(".txt"):
        continue
    with open(os.path.join(LABEL_DIR, fname), "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                # 使用 int(float(...)) 兼容 "5" 和 "5.0"
                counter[int(float(parts[0]))] += 1

print("当前训练集各类别标注数量：")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {i}  {name}: {counter[i]}")
print(f"\n总标注数：{sum(counter.values())}")