import ultralytics.nn.tasks as tasks
from modules import CBAM, EMA, ECA, SimAM, CA

tasks.CBAM = CBAM
tasks.EMA = EMA
tasks.ECA = ECA
tasks.SimAM = SimAM
tasks.CA = CA

from ultralytics import YOLO

yaml_list = [
    'yolov8s_eca.yaml',
    'yolov8s_simam.yaml',
    'yolov8s_ca.yaml',
    'yolov8n_ema.yaml',
    'yolov8s_seg_cbam.yaml',
    'yolov8s_seg_ema.yaml',
]

for y in yaml_list:
    try:
        m = YOLO(y)
        print(f'[OK] {y} 构建成功')
    except Exception as e:
        print(f'[FAIL] {y} 构建失败: {e}')