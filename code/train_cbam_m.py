import ultralytics.nn.tasks as tasks
from cbam import CBAM
from ultralytics import YOLO

tasks.CBAM = CBAM

if __name__ == '__main__':
    model = YOLO(r'D:\python\TomatoDiseaseProject\code\yolov8m-CBAM.yaml')
    model.train(
        data='datasets/tomato/data.yaml',
        epochs=1,
        imgsz=640,
        batch=4,
        workers=2,
        amp=True,
        project='runs/detect',
        name='cbam_m_smoke',
        fraction=0.02
    )