# 基于改进 YOLOv8 的番茄叶片健康状态智能诊断系统

基于改进 YOLOv8 的番茄叶片健康状态智能诊断系统。系统采用 Flask 搭建全栈 Web 应用，支持番茄叶片图像上传、病害检测与类别识别、严重程度分级评估，并自动生成 PDF 检测报告。本项目为本科毕业设计作品，完整覆盖从模型改进、训练实验到 Web 工程落地的全流程。

## 功能特性

- **改进 YOLOv8 检测模型**：在 YOLOv8n 基线中引入 CBAM 注意力机制（`modules/cbam.py`），同时提供 CA / ECA / SimAM / EMA 等可选注意力模块；通过动态注册方式接入 Ultralytics，不改动框架源码
- **番茄病害检测**：基于 9 类番茄病害数据集训练，支持叶片病斑识别与置信度输出
- **严重程度分级**：基于病斑检测框面积占比自动分级（轻度 ≤ 10% / 中度 10%–30% / 重度 > 30%）
- **PDF 检测报告**：检测完成后自动生成含结果图与诊断信息的 PDF 报告（`services/report.py`）
- **历史记录管理**：检测记录持久化存储（SQLite），支持历史查询与回溯（`services/database.py`）
- **防治建议**：根据病害类别输出对应防治建议（`services/treatment.py`）

## 技术栈

| 层        | 技术                                   |
| --------- | -------------------------------------- |
| 后端      | Flask                                  |
| 模型      | Ultralytics YOLOv8（PyTorch）          |
| 图像处理  | OpenCV / NumPy / SciPy                 |
| 报告生成  | ReportLab                              |
| 数据存储  | SQLite                                 |
| 配置      | PyYAML                                 |
| 可视化    | Matplotlib                             |

## 项目结构

```
Tomato-GitHub/
├── code/
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── ca.py
│   │   ├── cbam.py
│   │   ├── eca.py
│   │   ├── ema.py
│   │   └── simam.py
│   ├── services/
│   │   ├── database.py
│   │   ├── report.py
│   │   └── treatment.py
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   ├── templates/
│   │   ├── about.html
│   │   ├── base.html
│   │   ├── batch.html
│   │   ├── dashboard.html
│   │   ├── history.html
│   │   ├── index.html
│   │   └── result.html
│   ├── app.py
│   ├── augment_mosaic.py
│   ├── cbam.py
│   ├── count_classes.py
│   ├── ema.py
│   ├── fix_labels.py
│   ├── grade_severity.py
│   ├── merge_plantdoc.py
│   ├── ratio_distribution.py
│   ├── run_queue.py
│   ├── sample_calibration.py
│   ├── smoke_ema.py
│   ├── summarize_results.py
│   ├── test_yaml.py
│   ├── train_cbam.py
│   ├── train_cbam_m.py
│   ├── train_cbam_s.py
│   ├── train_ema.py
│   ├── train_ema_m.py
│   ├── train_ema_s.py
│   ├── train_seg_optimized.py
│   ├── train_task.py
│   ├── train_yolov5s.py
│   ├── visualize_seg.py
│   ├── yolov8m-CBAM.yaml
│   ├── yolov8m_ema.yaml
│   ├── yolov8n-CBAM.yaml
│   ├── yolov8n_ema.yaml
│   ├── yolov8s-CBAM.yaml
│   ├── yolov8s-EMA.yaml
│   ├── yolov8s_ca.yaml
│   ├── yolov8s_cbam_ema.yaml
│   ├── yolov8s_eca.yaml
│   ├── yolov8s_ema.yaml
│   ├── yolov8s_seg_cbam.yaml
│   ├── yolov8s_seg_ema.yaml
│   └── yolov8s_simam.yaml
├── images/
├── .gitignore
└── requirements.txt
```

## 快速开始

### 1. 环境要求

- Python 3.8+（建议 3.10 / 3.11 以获得完整 PyTorch 支持）
- 训练与推理建议使用 NVIDIA GPU（CUDA 环境）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 准备模型权重

本仓库不包含训练权重（`*.pt` 已被 .gitignore 排除）。请将训练好的权重文件（如 `yolov8s-CBAM.pt`）放入 `code/` 目录后使用。

### 4. 启动系统

```bash
cd code
python app.py
```

浏览器访问 <http://127.0.0.1:5000> 即可使用。

## 界面预览

![系统主界面](images/ui.png)

> 占位说明：请将系统运行界面截图保存为 `images/ui.png` 后，此处将自动显示。

## 说明

- 分级阈值（轻度 ≤ 10% / 中度 10%–30% / 重度 > 30%）为本系统的工程化分级规则，不代表农业病害领域的统一标准
- `datasets/`、`runs/`、`*.pt`、`*.db` 等训练与运行产物均被 `.gitignore` 排除，未包含在本仓库中

## 声明

本项目为个人毕业设计作品，代码仅供学习与交流使用。
