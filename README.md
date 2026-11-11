# Tomato-Disease-Diagnosis

> 中文文档：[README.zh-CN.md](README.zh-CN.md) · Undergraduate thesis project, 2027.

**An improved YOLOv8 framework for tomato leaf disease detection and severity grading**

A single pipeline that goes from a raw leaf photograph to a disease label, a severity grade, a treatment recommendation, and a PDF report — spanning model comparison, a hand-designed grading algorithm, and a deployed Flask application.

---

## Research Highlights

- **No stable gain from attention under the present training configuration.** Adding an EMA block to YOLOv8n raised mAP50-95 from 0.9242 to **0.9520 (absolute +0.0278)**, while the same family of modules — CBAM, EMA, ECA, SimAM and CA — applied to YOLOv8s all landed between 0.9637 and 0.9648, below the 0.9673 baseline. **Important caveat:** the baselines were initialised from COCO-pretrained weights whereas the modified models were trained from scratch from YAML configs, and batch sizes differ across tasks (8/16 for n, 4/6 for m). The comparison therefore measures initialisation as much as architecture, and the negative result is reported as an observation rather than a conclusion about attention itself.
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
| YOLOv5s | — | 0.9636 | reference |

**Interpretation.** The spread across all s-scale variants is 0.0011, and each configuration was trained once with a single seed, so differences of this size are within run-to-run noise. A defensible statement requires identical initialisation, identical batch size and at least three seeds per configuration; that experiment is listed as future work. For deployment the Web system loads the `cbam_s` checkpoint (mAP50-95 0.9637), which was the first checkpoint to run the full register–load–infer–grade–store pipeline end to end. It is **not** the best-performing model — the YOLOv8s baseline reaches 0.9673 and is also cheaper to run — so this deployment choice is an engineering legacy rather than an accuracy-based decision.

A segmentation branch (YOLOv8s-seg) was also trained on a merged AgriVision + PlantSeg set. Its mAP50 rose only from 0.5723 to 0.6097 over 187 epochs because the two sources differ sharply in background, lighting, and annotation standard, and one class is severely under-represented. Rather than continue tuning, the segmentation model was kept as a comparison arm and the HSV approach was adopted for grading.

---

## Method: Severity Grading

Bounding-box area ratios are unreliable under coarse annotation, so grading is handled by an A/B split (`code/grade_severity.py`):

- **Group A — lesion-type diseases.** Within the detected leaf ROI, healthy tissue is extracted in HSV space; everything else is treated as lesion. The severity ratio is `lesion area / leaf area`.
- **Group B — chlorosis-type diseases** (mosaic virus, yellow leaf curl). Colour segmentation is not discriminative for these, so the system returns a conservative reference ratio and labels the output as advisory.
- **Healthy.** Ratio fixed at 0.0; segmentation is skipped entirely.

Grade thresholds: `ratio ≤ 0.10` mild, `0.10 < ratio ≤ 0.30` moderate, `ratio > 0.30` severe.

> These thresholds are an engineering rule set calibrated for this system. They are not an agronomic standard. They remain **unchanged** in the running system; the recalibrated values in the next section are exploratory only.


---

## Human-Annotated Validation (98 samples)

The thresholds above were calibrated from the coverage-ratio distribution, not from human judgement. To quantify the gap, 100 lesion-type validation images were drawn by stratified sampling (fixed seed `20261110`) and graded independently by a human annotator; 2 ambiguous images were excluded, leaving **98 valid samples** (51 mild / 32 moderate / 15 severe).

| Setting | T1 / T2 | Agreement | Cohen's κ |
| :-- | :-- | --: | --: |
| Running system (unchanged) | 0.10 / 0.30 | **56.12%** (55/98) | 0.3400 |
| Recalibrated on annotations | 0.23 / 0.38 | 73.47% | 0.5551 |
| 5-fold cross-validation (held-out folds) | — | 66.21% | 0.4445 |

Confusion matrix (rows = human, columns = system): 27/20/4 (mild), 6/13/13 (moderate), 0/0/15 (severe). The system **systematically overestimates severity**: 24 of 51 mild samples were graded moderate or severe, while all 15 severe samples were recognised. The main cause is that the HSV "non-green" mask also counts veins, chlorotic leaf margins, specular highlights and residual background as lesion area.

Recalibration is an **exploratory analysis** — 98 samples and no independent validation set. The running system therefore still uses `T1 = 0.10 / T2 = 0.30`; a production threshold change would require a larger annotation set, multiple independent annotators and a held-out evaluation. Full data are in the thesis, Section 4.5.1 (Tables 4-3 and 4-4).

---

## Reproducing

### 1. Environment

```bash
pip install -r requirements.txt
```

Verified on Windows 11 / Python 3.12.14 / PyTorch 2.6.0+cu124 / Ultralytics 8.4.147 with an RTX 3050 (8 GB). A CUDA-capable GPU is advised for training, not required for inference.

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
python code/train_task.py --task detect --model yolov8s_cbam.yaml --data datasets/tomato/data.yaml --name cbam_s --epochs 100
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

Undergraduate thesis project. Code is shared for learning and reference. The final thesis (45 pages, 23 figures, 9 tables, 23 references), the 100-image annotation set and the agreement / recalibration analyses are **not distributed with this repository**; only the verified numbers are summarised above. All results are reproducible from the training logs and the annotation CSV.

Datasets, training runs, checkpoints, and the SQLite database are git-ignored and not distributed with this repository.
