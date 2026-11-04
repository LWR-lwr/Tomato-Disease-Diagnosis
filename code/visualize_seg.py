import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cbam import load_cbam_model
from grade_severity import segment_lesion, T1, T2

MODEL_PATH = r"D:\python\TomatoDiseaseProject\runs\detect\runs\detect\cbam_s\weights\best.pt"
IMG_DIR = r"D:\python\TomatoDiseaseProject\datasets\tomato\images\val"
OUT_DIR = r"D:\python\TomatoDiseaseProject\outputs\seg_check_v25"

os.makedirs(OUT_DIR, exist_ok=True)
model = load_cbam_model(MODEL_PATH)

img_files = sorted([f for f in os.listdir(IMG_DIR)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:30]

for idx, img_name in enumerate(img_files):
    img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    results = model.predict(img, conf=0.25, verbose=False)[0]
    if len(results.boxes) == 0:
        continue

    best = int(results.boxes.conf.argmax())
    cls_name = model.names[int(results.boxes.cls[best])]
    conf_val = float(results.boxes.conf[best])

    x1, y1, x2, y2 = map(int, results.boxes.xyxy[best])
    h, w = img.shape[:2]
    crop = img[max(y1 - 10, 0):min(y2 + 10, h),
               max(x1 - 10, 0):min(x2 + 10, w)]
    if crop.size == 0:
        continue

    if "healthy" in cls_name.lower():
        ratio = 0.0
        sev = "Healthy"
        lesion_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    else:
        lesion_mask, _, ratio = segment_lesion(crop, cls_name)
        if ratio <= T1:
            sev = "Mild"
        elif ratio <= T2:
            sev = "Moderate"
        else:
            sev = "Severe"

    panel_h, panel_w = 320, 320
    def resize_keep(im):
        return cv2.resize(im, (panel_w, panel_h))

    p1 = resize_keep(crop)
    green_vis = cv2.bitwise_and(crop, crop, mask=cv2.bitwise_not(lesion_mask))
    p2 = resize_keep(green_vis)
    overlay = crop.copy()
    overlay[lesion_mask > 0] = (0, 0, 255)
    p3 = resize_keep(cv2.addWeighted(crop, 0.6, overlay, 0.4, 0))

    p4 = np.ones((panel_h, panel_w, 3), dtype=np.uint8) * 240
    for i, line in enumerate([f"Class: {cls_name}", f"Conf: {conf_val:.3f}",
                              f"Ratio: {ratio:.4f}", f"Severity: {sev}"]):
        cv2.putText(p4, line, (10, 60 + i * 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    top = np.hstack([p1, p2])
    bottom = np.hstack([p3, p4])
    quad = np.vstack([top, bottom])
    cv2.putText(quad, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(quad, "Healthy(Green)", (panel_w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(quad, "Lesion(Red)", (10, panel_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    out_path = os.path.join(OUT_DIR, f"{idx:03d}_{cls_name}.jpg")
    cv2.imwrite(out_path, quad)
    print(f"[{idx+1}/{len(img_files)}] {img_name} -> {cls_name}  ratio={ratio:.4f}  {sev}")

print(f"\nDone. Saved to: {OUT_DIR}")