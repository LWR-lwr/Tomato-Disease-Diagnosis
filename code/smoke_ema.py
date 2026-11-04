import ultralytics.nn.tasks as tasks
from ema import EMA
from ultralytics import YOLO

tasks.EMA = EMA

if __name__ == '__main__':
    model = YOLO('code/yolov8s-EMA.yaml')
    model.train(
        data='datasets/tomato/data.yaml',
        epochs=1,
        imgsz=640,
        batch=4,
        workers=2,
        fraction=0.02,
        project='runs/detect',
        name='ema_s_smoke'
    )