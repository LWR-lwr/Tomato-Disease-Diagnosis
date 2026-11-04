import argparse
import os
import sys
import importlib

# 确保可以导入 modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 动态注册模块到 ultralytics
import ultralytics.nn.tasks as tasks
from modules import CBAM, EMA, ECA, SimAM, CA

tasks.CBAM = CBAM
tasks.EMA = EMA
tasks.ECA = ECA
tasks.SimAM = SimAM
tasks.CA = CA

from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['detect', 'segment'])
    parser.add_argument('--model', type=str, required=True, help='模型配置 yaml 或 .pt 权重')
    parser.add_argument('--data', type=str, required=True, help='数据集 data.yaml 路径')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--project', type=str, default='runs')
    parser.add_argument('--name', type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        task=args.task,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        amp=True,
        project=args.project,
        name=args.name,
    )

if __name__ == '__main__':
    main()