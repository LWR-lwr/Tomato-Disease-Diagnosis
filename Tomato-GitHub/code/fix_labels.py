import os

LABEL_DIR = r"D:\python\TomatoDiseaseProject\datasets\tomato\labels\train"

fixed = 0
for fname in os.listdir(LABEL_DIR):
    if not fname.startswith("aug_") or not fname.endswith(".txt"):
        continue
    path = os.path.join(LABEL_DIR, fname)
    with open(path, "r") as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        cls_id = int(float(parts[0]))   # 将 5.0 转为 5
        new_lines.append(f"{cls_id} " + " ".join(parts[1:]) + "\n")
    with open(path, "w") as f:
        f.writelines(new_lines)
    fixed += 1

print(f"已修复 {fixed} 个增强标签文件。")