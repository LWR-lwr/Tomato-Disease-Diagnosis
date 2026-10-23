import os
import csv

DETECT_DIR = r"D:\python\TomatoDiseaseProject\runs\detect\runs\detect"
SEG_DIR = r"D:\python\TomatoDiseaseProject\runs\segment\runs\segment"


def summarize(folder, task_type):
    print(f"\n{'='*80}")
    print(f"{task_type} 任务汇总")
    print(f"{'='*80}")
    print(f"{'任务名':<20} {'Precision':<12} {'Recall':<12} {'mAP50':<12} {'mAP50-95':<12}")
    print("-" * 80)

    if not os.path.exists(folder):
        print(f"目录不存在：{folder}")
        return

    for task_name in sorted(os.listdir(folder)):
        task_path = os.path.join(folder, task_name)
        if not os.path.isdir(task_path):
            continue
        csv_path = os.path.join(task_path, "results.csv")
        if not os.path.exists(csv_path):
            continue

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            if len(reader) < 2:
                continue
            header = reader[0]
            last_row = reader[-1]

        # 找到指标列
        try:
            p_idx = header.index("metrics/precision(B)")
            r_idx = header.index("metrics/recall(B)")
            m50_idx = header.index("metrics/mAP50(B)")
            m5095_idx = header.index("metrics/mAP50-95(B)")
        except ValueError:
            print(f"{task_name:<20} (指标列名不匹配)")
            continue

        p = float(last_row[p_idx])
        r = float(last_row[r_idx])
        m50 = float(last_row[m50_idx])
        m5095 = float(last_row[m5095_idx])

        print(f"{task_name:<20} {p:<12.4f} {r:<12.4f} {m50:<12.4f} {m5095:<12.4f}")


if __name__ == "__main__":
    summarize(DETECT_DIR, "检测")
    summarize(SEG_DIR, "分割")