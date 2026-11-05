# Tomato-Disease-Diagnosis

**An improved YOLOv8 framework for tomato leaf disease detection and severity grading**

A single pipeline that goes from a raw leaf photograph to a disease label, a severity grade, a treatment recommendation, and a PDF report — spanning model comparison, a hand-designed grading algorithm, and a deployed Flask application.

---

## Research Highlights

- **Attention helps small models but saturates on mid-sized ones.** Adding an EMA attention block to YOLOv8n raised mAP50-95 from 0.9242 to **0.9520 (+2.78%)**. Applying the same family of modules — CBAM, EMA, ECA, SimAM, and CA — to YOLOv8s produced *no* gain: all five landed between 0.9637 and 0.9648, below the 0.9673 baseline. This negative result is reported as-is rather than tuned away.
- **Coarse annotations break box-area grading, so the grading problem was reformulated.** Because bounding boxes in the dataset are drawn loosely around whole leaves, lesion area cannot be read off box ratios. We split diseases into two groups: lesion-type diseases are graded by HSV colour segmentation inside the detected leaf, while chlorosis-type diseases — whose colour signature is not discriminative — return a conservative reference value instead of a spurious measurement.
- **A complete, runnable system rather than a model checkpoint.** Detection, grading, history browsing, an ECharts dashboard, and PDF report export are all wired together and were validated end-to-end on held-out images.

---

## System Overview

<p align="center">
  <img src="images/ui.png" width="720" alt="Upload interface">
</p>

<p align="center"><em>Figure 1. Upload interface of the diagnostic system.</em></p>

### Detection and grading output

A single inference returns three aligned views — the original crop, the detected lesion with its class and confidence, and the segmented lesion mask used to compute the severity ratio.

| Original | Detection | Lesion mask |
| :--: | :--: | :--: |
| <img src="images/detect_original.jpg" width="220"> | <img src="images/detect_anno.jpg" width="220"> | <img src="images/detect_lesion.jpg" width="220"> |

<p align="center"><em>Figure 2. Early blight, confidence 0.94. The mask (right) drives the severity ratio.</em></p>

### Dashboard and history

<p align="center">
  <img src="images/dashboard.png" width="720" alt="ECharts dashboard">
</p>

<p align="center"><em>Figure 3. Detection dashboard: class distribution, severity distribution, and a 7-day trend.</em></p>

<p align="center">
  <img src="images/history.png" width="720" alt="Detection history">
</p>

<p align="center"><em>Figure 4. Persistent detection history backed by SQLite.</em></p>

---

## Experimental Results

All detection experiments were run on a single RTX 3050 (8 GB) with Ultralytics 8.4.147 and PyTorch 2.6.0. Attention modules are registered dynamically into `ultralytics.nn.tasks` — no framework source is modified.

| Backbone | Attention | mAP50-95 | Δ vs. baseline |
| :-- | :-- | --: | :-- |
| YOLOv8n | — | 0.9242 | — |
| YOLOv8n | **EMA** | **0.9520** | **+2.78%** |
| YOLOv8s | — | **0.9673** | — |
| YOLOv8s | CBAM / EMA / ECA / SimAM / CA | 0.9637 – 0.9648 | negative |
| YOLOv5s | — | 0.9640 | reference |

**Interpretation.** The n-scale model has spare capacity, so a well-placed attention block yields a clear gain. At the s-scale the baseline already sits near the ceiling this dataset supports; every variant tested fell slightly below it. The takeaway is not that attention is ineffective, but that its benefit is capacity-dependent — which is why the final deployed model uses the n-scale backbone with EMA.

A segmentation branch (YOLOv8s-seg) was also trained on a merged AgriVision + PlantSeg set. It plateaued at mAP50 ≈ 0.588 because the two sources differ sharply in background, lighting, and annotation standard, and one class is severely under-represented. Rather than continue tuning, the segmentation model was kept as a comparison arm and the HSV approach was adopted for grading.

---

## Method: Severity Grading

Bounding-box area ratios are unreliable under coarse annotation, so grading is handled by an A/B split (`code/grade_severity.py`):

- **Group A — lesion-type diseases.** Within the detected leaf ROI, healthy tissue is extracted in HSV space; everything else is treated as lesion. The severity ratio is `lesion area / leaf area`.
- **Group B — chlorosis-type diseases** (mosaic virus, yellow leaf curl). Colour segmentation is not discriminative for these, so the system returns a conservative reference ratio and labels the output as advisory.
- **Healthy.** Ratio fixed at 0.0; segmentation is skipped entirely.

Grade thresholds: `ratio ≤ 0.10` mild, `0.10 < ratio ≤ 0.30` moderate, `ratio > 0.30` severe.

> These thresholds are an engineering rule set calibrated for this system. They are not an agronomic standard.

---

## Reproducing

### 1. Environment

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended. A CUDA-capable GPU is advised for training, not required for inference.

### 2. Model weights

Weights are excluded from the repository (`*.pt` is git-ignored). Place the trained checkpoint at `code/weights/best.pt`:

```bash
mkdir -p code/weights
# copy best.pt into code/weights/
```

Or point to it explicitly:

```bash
# Windows
set TOMATO_MODEL_PATH=D:\path\to\best.pt
# Linux / macOS
export TOMATO_MODEL_PATH=/path/to/best.pt
```

### 3. Run

```bash
cd code
python app.py
```

Then open <http://127.0.0.1:5000>.

### 4. Retrain

`code/train_task.py` is the general entry point; `code/run_queue.py` replays the full experiment queue serially and writes logs to `logs/`.

```bash
python code/train_task.py --task ema --model yolov8n --data datasets/tomato/data.yaml --name ema_n
```

---

## Repository Structure

```
code/
├── modules/              # Attention modules: CBAM, EMA, ECA, SimAM, CA
├── services/
│   ├── database.py       # SQLite persistence for detection records
│   ├── report.py         # PDF report generation (ReportLab)
│   └── treatment.py      # Treatment recommendation lookup
├── templates/            # Bootstrap 5 front-end pages
├── app.py                # Flask entry point
├── grade_severity.py     # A/B severity grading algorithm
├── train_task.py         # General training entry point
├── run_queue.py          # Serial experiment queue
└── *.yaml                # Backbone + attention configurations
```

---

## Stack

`PyTorch` · `Ultralytics YOLOv8` · `OpenCV` · `NumPy` · `SciPy` · `Flask` · `SQLite` · `ReportLab` · `Bootstrap 5` · `ECharts`

---

## Note

Undergraduate thesis project. Code is shared for learning and reference.

Datasets, training runs, checkpoints, and the SQLite database are git-ignored and not distributed with this repository.
