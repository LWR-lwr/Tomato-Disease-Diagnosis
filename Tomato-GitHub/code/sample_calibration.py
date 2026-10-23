import os
import random
import shutil

IMG_DIR = r"D:\python\TomatoDiseaseProject\datasets\tomato\images\val"
LBL_DIR = r"D:\python\TomatoDiseaseProject\datasets\tomato\labels\val"
OUT_DIR = r"D:\python\TomatoDiseaseProject\calibration_samples"

os.makedirs(OUT_DIR, exist_ok=True)

# 读 data.yaml 的类别
names = ['Tomato__BacterialSpot', 'Tomato__EarlyBlight', 'Tomato__Healthy',
         'Tomato__LateBlight', 'Tomato__LeafMold', 'Tomato__MosaicVirus',
         'Tomato__SeptoriaLeafSpot', 'Tomato__SpiderMites',
         'Tomato__TargetSpot', 'Tomato__YellowLeafCurlVirus']

# 按类别分组
class_files = {i: [] for i in range(10)}
for lbl_file in os.listdir(LBL_DIR):
    if not lbl_file.endswith('.txt'):
        continue
    with open(os.path.join(LBL_DIR, lbl_file)) as f:
        for line in f:
            cls_id = int(line.split()[0])
            class_files[cls_id].append(lbl_file.replace('.txt', '.jpg'))
            break

# 每类抽 6 张
random.seed(42)
selected = []
for cls_id in range(10):
    files = class_files[cls_id]
    picks = random.sample(files, min(6, len(files)))
    selected.extend(picks)
    print(f"{names[cls_id]}: 抽 {len(picks)} 张")

# 复制到标定集
for f in selected:
    src = os.path.join(IMG_DIR, f)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(OUT_DIR, f))

print(f"\n共 {len(selected)} 张，保存在 {OUT_DIR}")