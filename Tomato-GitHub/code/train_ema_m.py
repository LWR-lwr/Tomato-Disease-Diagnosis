import ultralytics.nn.tasks as tasks
from ema import EMA
from ultralytics import YOLO

tasks.EMA = EMA

if __name__ == '__main__':
    model = YOLO(r'D:\python\TomatoDiseaseProject\runs\detect\runs\detect\ema_m\weights\last.pt')
    model.train(resume=True)