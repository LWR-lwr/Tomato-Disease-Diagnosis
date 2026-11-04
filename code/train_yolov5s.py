from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov5su.pt')  # Ultralytics 版 YOLOv5s
    model.train(
        data='datasets/tomato/data.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        workers=2,
        amp=True,
        seed=0,
        project='runs/detect',
        name='yolov5s',
    )