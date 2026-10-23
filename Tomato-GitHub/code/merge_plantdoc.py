import os
import shutil

# ================= 配置区（只需改这里） =================
# PlantDoc 数据集根目录，下面应该有 images/train 和 labels/train
PLANTDOC_ROOT = r"D:\python\TomatoDiseaseProject\datasets\plantdoc"

# 你的番茄数据集根目录，下面有 images/train 和 labels/train
TARGET_ROOT = r"D:\python\TomatoDiseaseProject\datasets\tomato"

# 合并到训练集
SPLIT = "train"

# PlantDoc 原始类别 ID -> 你的类别 ID
CLASS_MAP = {
    7: 3,    # Tomato leaf late blight -> Tomato__LateBlight
    8: 4,    # Tomato mold leaf -> Tomato__LeafMold
    11: 9,   # Tomato leaf yellow virus -> Tomato__YellowLeafCurlVirus
    13: 5,   # Tomato leaf mosaic virus -> Tomato__MosaicVirus
    15: 0,   # Tomato leaf bacterial spot -> Tomato__BacterialSpot
    19: 1,   # Tomato Early blight leaf -> Tomato__EarlyBlight
    21: 6,   # Tomato Septoria leaf spot -> Tomato__SeptoriaLeafSpot
    28: 7,   # Tomato two spotted spider mites leaf -> Tomato__SpiderMites
}
# ==========================================================

src_img_dir = os.path.join(PLANTDOC_ROOT, "images", SPLIT)
src_lbl_dir = os.path.join(PLANTDOC_ROOT, "labels", SPLIT)

dst_img_dir = os.path.join(TARGET_ROOT, "images", SPLIT)
dst_lbl_dir = os.path.join(TARGET_ROOT, "labels", SPLIT)

# 检查源目录是否存在
if not os.path.exists(src_img_dir):
    print(f"错误：找不到 PlantDoc 图片目录 {src_img_dir}")
    print("请检查 PLANTDOC_ROOT 路径是否正确，以及下面是否有 images/train 文件夹。")
    exit(1)

if not os.path.exists(src_lbl_dir):
    print(f"错误：找不到 PlantDoc 标签目录 {src_lbl_dir}")
    exit(1)

os.makedirs(dst_img_dir, exist_ok=True)
os.makedirs(dst_lbl_dir, exist_ok=True)

count = 0
for img_name in os.listdir(src_img_dir):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join(src_lbl_dir, label_name)
    if not os.path.exists(label_path):
        continue

    new_lines = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            if cls_id in CLASS_MAP:
                new_id = CLASS_MAP[cls_id]
                new_lines.append(f"{new_id} " + " ".join(parts[1:]) + "\n")

    if new_lines:  # 只有包含番茄类别的图片才复制
        # 加前缀避免与原有图片重名
        new_img_name = "plantdoc_" + img_name
        new_lbl_name = "plantdoc_" + label_name

        shutil.copy(os.path.join(src_img_dir, img_name),
                    os.path.join(dst_img_dir, new_img_name))
        with open(os.path.join(dst_lbl_dir, new_lbl_name), "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        count += 1

print(f"完成！共合并 {count} 张 PlantDoc 番茄图片到训练集。")
print(f"图片保存到：{dst_img_dir}")
print(f"标签保存到：{dst_lbl_dir}")