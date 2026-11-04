from ultralytics import YOLO

if __name__ == '__main__':
    # 加载预训练权重
    model = YOLO('yolov8s-seg.pt')

    # 开始训练
    model.train(
        # ===== 基础配置 =====
        data='datasets/tomato_seg/data.yaml',
        epochs=200,              # 从100增加到200
        imgsz=640,
        batch=4,                 # 3050 8GB 推荐 batch=4
        workers=2,
        amp=True,
        patience=30,             # 30轮无提升自动早停

        # ===== 数据增强 =====
        hsv_h=0.015,             # 色调增强
        hsv_s=0.7,               # 饱和度增强
        hsv_v=0.4,               # 亮度增强
        degrees=15.0,            # 随机旋转 ±15°
        translate=0.1,           # 平移
        scale=0.5,               # 缩放
        shear=2.0,               # 剪切
        perspective=0.0,         # 透视（默认0）
        flipud=0.3,              # 上下翻转概率
        fliplr=0.5,              # 左右翻转概率
        mosaic=1.0,              # Mosaic增强
        mixup=0.1,               # MixUp增强
        copy_paste=0.3,          # 分割专用Copy-Paste增强，缓解类别不平衡

        # ===== 损失权重 =====
        box=7.5,                 # 边界框损失权重
        cls=0.5,                 # 分类损失权重
        dfl=1.5,                 # DFL损失权重

        # ===== 学习率 =====
        lr0=0.01,                # 初始学习率
        lrf=0.01,                # 最终学习率
        cos_lr=True,             # 余弦退火
        warmup_epochs=5,         # 预热轮次

        # ===== 多尺度训练 =====
        multi_scale=True,

        # ===== 保存路径 =====
        project='runs/segment',
        name='seg_optimized',
    )