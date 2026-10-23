# Tomato Leaf Intelligent Diagnosis System
基于改进 YOLOv8 与 Flask 的番茄叶片健康状态智能诊断与严重程度分级系统。

## ✨ 核心功能
- **目标检测**：基于 YOLOv8，支持 10 类番茄叶片病害与健康状态的识别。
- **量化分级**：提出 A/B 分组策略，针对病斑型采用 HSV 病征覆盖率提取，科学标定阈值输出轻/中/重等级。
- **智能诊断系统**：Flask 全栈开发，集成批量处理、SQLite 历史记录、ECharts 数据看板及 PDF 诊断报告导出。

## 🔬 算法实验亮点 (Ablation Study)
- 系统开展了 n/s/m 三种模型规模与 5 种主流注意力机制（CBAM, EMA 等）的对比实验。
- 揭示了注意力机制的**规模依赖性**：轻量级模型（n）上 EMA 提升显著（mAP50-95 +2.78%），但在大模型上因特征饱和增益有限。
- 引入 YOLOv5s 与 YOLOv8-seg 进行跨代及分割技术的对比验证。

## 🛠️ 技术栈
Python, PyTorch, Ultralytics (YOLOv8), OpenCV, Flask, SQLite, ECharts.

## 📸 系统界面
![系统界面](images/ui.png)
![检测结果](images/result.png)
