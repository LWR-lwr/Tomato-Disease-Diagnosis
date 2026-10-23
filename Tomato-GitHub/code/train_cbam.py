# -*- coding: utf-8 -*-
"""YOLOv8n-CBAM 训练脚本（唯一改进点：CBAM 注意力机制）。

本脚本特点：
    1. 通过 cbam.py 的“动态注册”机制把 CBAM 接入 ultralytics，
       不修改 ultralytics 源码（不碰 site-packages 里的任何文件）；
    2. 训练超参数与基线模型 baseline 完全一致，保证改进前后的对比公平；
    3. 训练结果固定保存到 runs/detect/cbam（绝对路径推导，换电脑/换盘符都不失效）；
    4. 训练结束后自动读取结果文件，打印与基线的指标对比表（写论文直接用）。

用法（两种都行）：
    conda activate yolo_env
    python code/train_cbam.py                      # 正式训练：100 epochs
    python code/train_cbam.py --smoke              # 快速自检：1 epoch + 2% 数据，先确认环境没问题
    python code/train_cbam.py --epochs 150 --batch 8 --device 0
    python D:/python/TomatoDiseaseProject/code/train_cbam.py    # 从任意目录调用也可以
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 一、路径配置：全部由脚本自身位置推导，避免“相对路径 + 当前工作目录”踩坑
#     （整个项目拷到移动硬盘、换电脑后路径依旧有效）
# ---------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent  # TomatoDiseaseProject/code
PROJECT_ROOT = CODE_DIR.parent  # TomatoDiseaseProject

if str(CODE_DIR) not in sys.path:  # 保证无论从哪个目录运行都能 import cbam
    sys.path.insert(0, str(CODE_DIR))

from cbam import register_cbam  # noqa: E402  （导入 cbam 时已自动完成注册，这里再显式调用一次）

DATA_YAML = PROJECT_ROOT / "datasets" / "tomato" / "data.yaml"  # 数据集配置
MODEL_YAML = CODE_DIR / "yolov8n-CBAM.yaml"  # 改进后的模型结构
PRETRAINED_WEIGHTS = PROJECT_ROOT / "yolov8n.pt"  # COCO 预训练权重（与基线初始化方式一致）
RUN_ROOT = PROJECT_ROOT / "runs" / "detect"  # 结果保存根目录 -> runs/detect
RUN_NAME = "cbam"  # 结果子目录名      -> runs/detect/cbam

# 基线结果所在位置（用户目录下有两种可能层级，按顺序查找）
BASELINE_CSV_CANDIDATES = (
    RUN_ROOT / "baseline" / "results.csv",
    RUN_ROOT / "runs" / "detect" / "baseline" / "results.csv",
)

# ---------------------------------------------------------------------------
# 二、训练超参数：与 baseline 完全一致，保证“唯一变量 = CBAM”
# ---------------------------------------------------------------------------
EPOCHS = 100
BATCH = 8
IMGSZ = 640
WORKERS = 2
SEED = 0
DETERMINISTIC = True
PATIENCE = 100
OPTIMIZER = "auto"
LR0 = 0.01
LRF = 0.01
COS_LR = False
CLOSE_MOSAIC = 10
AMP = True
CACHE = False

METRIC_COLUMNS = (
    ("metrics/precision(B)", "Precision"),
    ("metrics/recall(B)", "Recall"),
    ("metrics/mAP50(B)", "mAP50"),
    ("metrics/mAP50-95(B)", "mAP50-95"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数（不传参数时即为“正式训练”的默认配置）。"""
    parser = argparse.ArgumentParser(description="YOLOv8n-CBAM 训练脚本")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"训练轮数，默认 {EPOCHS}")
    parser.add_argument("--batch", type=int, default=BATCH, help=f"批大小，默认 {BATCH}")
    parser.add_argument("--imgsz", type=int, default=IMGSZ, help=f"输入尺寸，默认 {IMGSZ}")
    parser.add_argument("--workers", type=int, default=WORKERS, help=f"数据加载线程，默认 {WORKERS}")
    parser.add_argument("--device", default=None, help="训练设备，如 0 / 0,1 / cpu，默认自动选择 GPU")
    parser.add_argument("--name", default=RUN_NAME, help=f"结果子目录名，默认 {RUN_NAME}（保存到 runs/detect/<name>）")
    parser.add_argument("--resume", action="store_true", help="从上次中断的权重继续训练（需配合 --name 指定原目录）")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="快速自检模式：1 epoch + 2%% 训练数据，用于确认环境与代码正常（结果保存在 runs/detect/cbam_smoke）",
    )
    return parser.parse_args(argv)


def check_files() -> None:
    """训练前检查依赖文件是否齐全，缺文件时给出明确提示。"""
    missing = [str(p) for p in (DATA_YAML, MODEL_YAML) if not p.exists()]
    if missing:
        raise FileNotFoundError("缺少必要文件：\n  " + "\n  ".join(missing))
    if not PRETRAINED_WEIGHTS.exists():
        print(f"[警告] 未找到预训练权重 {PRETRAINED_WEIGHTS}，本次将从零开始训练（收敛会更慢）。")


def read_metrics(csv_path: Path) -> tuple[dict[str, float], dict[str, float]]:
    """读取 ultralytics 的 results.csv，返回（最后一轮指标, 最优一轮指标）。

    最优一轮按 mAP50-95 最大来挑选（该指标是目标检测最严格的综合评价指标）。
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到结果文件：{csv_path}")

    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"结果文件为空：{csv_path}")

    def _row_to_metrics(row: dict[str, str]) -> dict[str, float]:
        return {name: float(row[col]) for col, name in METRIC_COLUMNS if row.get(col)}

    last = _row_to_metrics(rows[-1])
    best_row = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
    best = _row_to_metrics(best_row)
    best["epoch"] = float(best_row["epoch"])
    last["epoch"] = float(rows[-1]["epoch"])
    return last, best


def find_baseline_csv() -> Path | None:
    """查找基线模型的结果文件（找不到则返回 None）。"""
    for path in BASELINE_CSV_CANDIDATES:
        if path.exists():
            return path
    return None


def print_comparison(cbam_csv: Path) -> None:
    """打印 YOLOv8n-CBAM 与基线 YOLOv8n 的指标对比表。"""
    cbam_last, cbam_best = read_metrics(cbam_csv)
    baseline_csv = find_baseline_csv()

    print("\n" + "=" * 78)
    print("验证集结果对比（最后一轮）")
    print("=" * 78)

    if baseline_csv is None:
        print(f"未找到基线结果文件，请手动对比。CBAM 结果文件：{cbam_csv}")
        for k, v in cbam_last.items():
            print(f"  {k:<12}{v:.4f}")
        return

    base_last, base_best = read_metrics(baseline_csv)
    print(f"基线结果：{baseline_csv}")
    print(f"CBAM 结果：{cbam_csv}\n")
    header = f"{'指标':<12}{'基线 YOLOv8n':>16}{'YOLOv8n-CBAM':>16}{'变化':>14}"
    print(header)
    print("-" * len(header))
    for name in ("Precision", "Recall", "mAP50", "mAP50-95"):
        b, c = base_last[name], cbam_last[name]
        print(f"{name:<12}{b:>16.4f}{c:>16.4f}{c - b:>+14.4f}")
    print("-" * len(header))
    print(f"（基线第 {base_last['epoch']:.0f} 轮 / CBAM 第 {cbam_last['epoch']:.0f} 轮）")

    print("\n各模型“最优轮次”参考：")
    print(f"  基线  最优 mAP50={base_best['mAP50']:.4f}  mAP50-95={base_best['mAP50-95']:.4f}（第 {base_best['epoch']:.0f} 轮）")
    print(f"  CBAM  最优 mAP50={cbam_best['mAP50']:.4f}  mAP50-95={cbam_best['mAP50-95']:.4f}（第 {cbam_best['epoch']:.0f} 轮）")
    print("\n提示：论文中建议写一组对比（如 mAP50 / mAP50-95 / 参数量 / GFLOPs），")
    print("      best.pt 对应最优轮次权重；曲线图见 save_dir 下的 results.png、BoxPR_curve.png 等。")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # ---------- 1. 环境检查 ----------
    check_files()
    register_cbam()  # 动态注册 CBAM（幂等，可重复调用）

    import ultralytics
    from ultralytics import YOLO

    print(f"\nultralytics 版本：{ultralytics.__version__}")
    print(f"数据集配置：{DATA_YAML}")
    print(f"模型结构：  {MODEL_YAML}")

    # ---------- 2. 构建模型 ----------
    model = YOLO(str(MODEL_YAML))  # 解析自定义 YAML（依赖上一步的动态注册）
    model.info()  # 打印网络层数、参数量、GFLOPs（写论文的模型复杂度对比可直接用）

    # 预训练权重：通过 pretrained=<权重路径> 传给 ultralytics（与基线模型初始化方式一致）
    # 训练开始时日志会打印 "Transferred xxx/358 items from pretrained weights"：
    # 类别数不同（COCO 80 类 -> 番茄病害 10 类），检测头权重不迁移，主干与颈部权重正常迁移，
    # 这是正常的微调行为，和基线模型的初始化方式完全一致。
    pretrained = str(PRETRAINED_WEIGHTS) if PRETRAINED_WEIGHTS.exists() else False

    # ---------- 3. 组装训练参数 ----------
    train_kwargs = {
        "data": str(DATA_YAML),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "device": args.device,
        "project": str(RUN_ROOT),  # 结果保存到 <项目根>/runs/detect
        "name": args.name,  # 默认 cbam -> runs/detect/cbam
        "exist_ok": False,  # 已有同名目录时自动改名为 cbam2、cbam3……避免覆盖历史结果
        "seed": SEED,
        "deterministic": DETERMINISTIC,
        "patience": PATIENCE,
        "optimizer": OPTIMIZER,
        "lr0": LR0,
        "lrf": LRF,
        "cos_lr": COS_LR,
        "close_mosaic": CLOSE_MOSAIC,
        "amp": AMP,
        "cache": CACHE,
        "pretrained": pretrained,  # 用 yolov8n.pt 的权重初始化（与基线一致）；缺权重时为 False（从零训练）
        "val": True,  # 每轮在验证集上评估
        "plots": True,  # 生成 PR 曲线、混淆矩阵等图
        "resume": args.resume,
    }
    if args.smoke:
        train_kwargs.update(epochs=1, fraction=0.02, name=f"{args.name}_smoke", plots=False, patience=1)
        print("\n[自检模式] 仅用 2% 训练数据跑 1 轮，用于验证环境与代码，正式训练请去掉 --smoke。\n")

    # ---------- 4. 开始训练 ----------
    print("=" * 78)
    print("开始训练 YOLOv8n-CBAM")
    print("=" * 78)
    model.train(**train_kwargs)

    save_dir = Path(model.trainer.save_dir)
    best_weight = save_dir / "weights" / "best.pt"
    print("\n训练结束。")
    print(f"  结果目录：{save_dir}")
    print(f"  最优权重：{best_weight}")

    # ---------- 5. 与基线对比 ----------
    results_csv = save_dir / "results.csv"
    if results_csv.exists():
        try:
            print_comparison(results_csv)
        except Exception as e:  # 对比只是附加功能，失败不影响训练结果
            print(f"[提示] 自动对比失败（可手动对比 results.csv）：{e}")


if __name__ == "__main__":
    main()
