import cv2
import numpy as np
from scipy import ndimage

# ===== 分级阈值（已校准） =====
T1 = 0.10
T2 = 0.30

# ===== B组：黄化型病害（颜色分割不可靠，输出保守值） =====
CHLOROSIS_DISEASES = ["mosaic", "yellowleafcurl"]
CHLOROSIS_RATIO = 0.15
CHLOROSIS_SEV = "Moderate"


def segment_lesion(crop_bgr, cls_name=""):
    """
    V2.6: HSV + 排除红/白背景 + S下限提到70
    返回: lesion, leaf, ratio
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # 健康组织：S下限70
    healthy = cv2.inRange(hsv, (35, 70, 40), (85, 255, 255))

    # 纯红背景/红反光
    red = cv2.inRange(hsv, (0, 40, 40), (10, 255, 255)) | \
          cv2.inRange(hsv, (170, 40, 40), (180, 255, 255))

    # 白色背景/高光
    white = (S < 40) & (V > 200)

    # 叶片前景
    leaf = ((S > 30) & (V > 40) & (V < 250)
            & (red == 0) & (~white)).astype(np.uint8)

    # 最大连通域 + 闭运算
    labeled, num = ndimage.label(leaf)
    if num > 0:
        sizes = ndimage.sum(leaf, labeled, range(1, num + 1))
        leaf = (labeled == (sizes.argmax() + 1)).astype(np.uint8)
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    # 病征 = 叶片内非健康区
    lesion = ((leaf > 0) & (healthy == 0)).astype(np.uint8) * 255
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    lesion = cv2.bitwise_and(lesion, leaf * 255)

    leaf_area = int((leaf > 0).sum())
    lesion_area = int((lesion > 0).sum())
    ratio = lesion_area / leaf_area if leaf_area > 0 else 0.0

    return lesion, leaf * 255, ratio


def grade_severity(model, img_bgr, conf=0.25):
    results = model.predict(img_bgr, conf=conf, verbose=False)[0]

    if len(results.boxes) == 0:
        return {"status": "no_detection"}

    best = int(results.boxes.conf.argmax())
    cls_name = model.names[int(results.boxes.cls[best])]
    conf_val = float(results.boxes.conf[best])

    # Healthy 直接返回
    if "healthy" in cls_name.lower():
        return {
            "class": cls_name,
            "conf": conf_val,
            "ratio": 0.0,
            "severity": "Healthy",
            "note": ""
        }

    # B组：黄化型病害，输出保守值
    if any(k in cls_name.lower() for k in CHLOROSIS_DISEASES):
        return {
            "class": cls_name,
            "conf": conf_val,
            "ratio": CHLOROSIS_RATIO,
            "severity": CHLOROSIS_SEV,
            "note": "该类病害颜色特征不显著，分级结果为参考值"
        }

    # A组：病斑型病害，走颜色分割
    x1, y1, x2, y2 = map(int, results.boxes.xyxy[best])
    h, w = img_bgr.shape[:2]
    crop = img_bgr[max(y1 - 10, 0):min(y2 + 10, h),
                   max(x1 - 10, 0):min(x2 + 10, w)]

    if crop.size == 0:
        return {"class": cls_name, "conf": conf_val,
                "ratio": 0.0, "severity": "Unknown", "note": ""}

    _, _, ratio = segment_lesion(crop, cls_name)

    if ratio <= T1:
        sev = "Mild"
    elif ratio <= T2:
        sev = "Moderate"
    else:
        sev = "Severe"

    return {
        "class": cls_name,
        "conf": conf_val,
        "ratio": round(ratio, 4),
        "severity": sev,
        "note": ""
    }