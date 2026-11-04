import os
import cv2
import albumentations as A

# ================= 配置区 =================
TARGET_ROOT = r"D:\python\TomatoDiseaseProject\datasets\tomato"
SPLIT = "train"
MOSAIC_CLASS_ID = 5          # Tomato__MosaicVirus
AUG_TIMES = 1                # 每张图生成 1 个增强版本
# ==========================================

IMG_DIR = os.path.join(TARGET_ROOT, "images", SPLIT)
LBL_DIR = os.path.join(TARGET_ROOT, "labels", SPLIT)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.3),
    A.MotionBlur(p=0.2),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

count = 0
for fname in os.listdir(LBL_DIR):
    if not fname.endswith(".txt") or fname.startswith("aug_"):
        continue

    label_path = os.path.join(LBL_DIR, fname)
    with open(label_path, "r") as f:
        lines = [l.strip().split() for l in f if l.strip()]

    has_mosaic = any(int(float(p[0])) == MOSAIC_CLASS_ID for p in lines)
    if not has_mosaic:
        continue

    # 找对应的图片文件（可能 .jpg 或 .png）
    base = os.path.splitext(fname)[0]
    img_path = None
    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = os.path.join(IMG_DIR, base + ext)
        if os.path.exists(candidate):
            img_path = candidate
            break
    if img_path is None:
        continue

    image = cv2.imread(img_path)
    bboxes = [[float(x) for x in p[1:5]] for p in lines]
    class_labels = [int(float(p[0])) for p in lines]   # 确保是整数

    for i in range(AUG_TIMES):
        try:
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        except Exception:
            continue

        new_img_name = f"aug_{i}_{os.path.basename(img_path)}"
        new_lbl_name = f"aug_{i}_{fname}"

        cv2.imwrite(os.path.join(IMG_DIR, new_img_name), augmented["image"])
        with open(os.path.join(LBL_DIR, new_lbl_name), "w") as f:
            for bbox, cls in zip(augmented["bboxes"], augmented["class_labels"]):
                # 关键修复：写入前将类别ID转为整数
                f.write(f"{int(cls)} " + " ".join([f"{v:.6f}" for v in bbox]) + "\n")
        count += 1

print(f"完成！共生成 {count} 张 MosaicVirus 增强图片。")