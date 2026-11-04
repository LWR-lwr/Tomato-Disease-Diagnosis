import subprocess
import os
import sys
from datetime import datetime

PROJECT_DIR = r"D:\python\TomatoDiseaseProject"
CODE_DIR = os.path.join(PROJECT_DIR, "code")
PYTHON = sys.executable
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ========== 任务队列 ==========
TASKS = [
    # ---------- 第一梯队：必须补 ----------
    ("detect_baseline_s", [
        "--task", "detect",
        "--model", "yolov8s.pt",
        "--data", "datasets/tomato/data.yaml",
        "--batch", "8",
        "--name", "baseline_s",
        "--project", "runs/detect",
    ]),
    ("detect_baseline_m", [
        "--task", "detect",
        "--model", "yolov8m.pt",
        "--data", "datasets/tomato/data.yaml",
        "--batch", "4",
        "--name", "baseline_m",
        "--project", "runs/detect",
    ]),
    ("detect_cbam_ema_s", [
        "--task", "detect",
        "--model", "code/yolov8s_cbam_ema.yaml",
        "--data", "datasets/tomato/data.yaml",
        "--batch", "8",
        "--name", "cbam_ema_s",
        "--project", "runs/detect",
    ]),
    ("seg_baseline_s", [
        "--task", "segment",
        "--model", "yolov8s-seg.pt",
        "--data", "datasets/tomato_seg/data.yaml",
        "--batch", "4",
        "--name", "seg_baseline_s",
        "--project", "runs/segment",
    ]),
    ("seg_cbam_s", [
        "--task", "segment",
        "--model", "code/yolov8s_seg_cbam.yaml",
        "--data", "datasets/tomato_seg/data.yaml",
        "--batch", "4",
        "--name", "seg_cbam_s",
        "--project", "runs/segment",
    ]),
    ("seg_ema_s", [
        "--task", "segment",
        "--model", "code/yolov8s_seg_ema.yaml",
        "--data", "datasets/tomato_seg/data.yaml",
        "--batch", "4",
        "--name", "seg_ema_s",
        "--project", "runs/segment",
    ]),

    # ---------- 第二梯队：注意力扩展 ----------
    ("detect_eca_s", [
        "--task", "detect",
        "--model", "code/yolov8s_eca.yaml",
        "--data", "datasets/tomato/data.yaml",
        "--batch", "8",
        "--name", "eca_s",
        "--project", "runs/detect",
    ]),
    ("detect_simam_s", [
        "--task", "detect",
        "--model", "code/yolov8s_simam.yaml",
        "--data", "datasets/tomato/data.yaml",
        "--batch", "8",
        "--name", "simam_s",
        "--project", "runs/detect",
    ]),
    ("detect_ca_s", [
        "--task", "detect",
        "--model", "code/yolov8s_ca.yaml",
        "--data", "datasets/tomato/data.yaml",
        "--batch", "8",
        "--name", "ca_s",
        "--project", "runs/detect",
    ]),

    # ---------- 第三梯队：可选 ----------
    ("detect_ema_n", [
        "--task", "detect",
        "--model", "code/yolov8n_ema.yaml",
        "--data", "datasets/tomato/data.yaml",
        "--batch", "16",
        "--name", "ema_n",
        "--project", "runs/detect",
    ]),
    ("seg_baseline_n", [
        "--task", "segment",
        "--model", "yolov8n-seg.pt",
        "--data", "datasets/tomato_seg/data.yaml",
        "--batch", "8",
        "--name", "seg_baseline_n",
        "--project", "runs/segment",
    ]),
]


def run_task(task_name, args):
    log_file = os.path.join(LOG_DIR, f"{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    cmd = [PYTHON, os.path.join(CODE_DIR, "train_task.py")] + args
    print(f"\n{'='*60}")
    print(f"开始：{task_name}")
    print(f"命令：{' '.join(cmd)}")
    print(f"日志：{log_file}")
    print(f"{'='*60}")
    with open(log_file, "w", encoding="utf-8") as f:
        result = subprocess.run(cmd, cwd=PROJECT_DIR, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode == 0:
        print(f"完成：{task_name}")
        return True
    else:
        print(f"失败：{task_name}，请查看日志 {log_file}")
        return False


def main():
    print(f"共 {len(TASKS)} 个任务待执行")
    for i, (name, args) in enumerate(TASKS, 1):
        print(f"\n[{i}/{len(TASKS)}] {name}")
        ok = run_task(name, args)
        if not ok:
            print("任务失败，停止后续。")
            sys.exit(1)
    print("\n所有任务完成！")


if __name__ == "__main__":
    main()