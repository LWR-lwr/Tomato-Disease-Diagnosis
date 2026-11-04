import ultralytics.nn.tasks as tasks
from cbam import CBAM
from ultralytics import YOLO

tasks.CBAM = CBAM

if __name__ == '__main__':
    model = YOLO('code/yolov8s-CBAM.yaml')
    model.train(
        data='datasets/tomato/data.yaml',
        epochs=100,
        imgsz=640,
        batch=8,  # 如果显存宽裕，可以试 batch=12 或 16
        workers=2,
        amp=True,
        project='runs/detect',
        name='cbam_s'
    )