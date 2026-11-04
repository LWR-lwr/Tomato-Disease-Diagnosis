import ultralytics.nn.tasks as tasks
from ema import EMA
from ultralytics import YOLO

tasks.EMA = EMA

if __name__ == '__main__':
    model = YOLO(r'D:\python\TomatoDiseaseProject\code\yolov8s_ema.yaml')
    model.train(
        data='datasets/tomato/data.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        workers=2,
        amp=True,
        project='runs/detect',
        name='ema_s'
    )