# CastLens — casting defect inspection

**Live demo:** <https://castlens.onrender.com/>  

> The demo runs on Render's free plan, which sleeps after about 15 minutes without traffic. If the page is slow to open, wait up to a minute for it to wake.

```bash
git clone https://github.com/thenun123/Castlens.git
cd Castlens
```

---

## Contents

1. [Overview](#overview)
2. [Repository layout](#repository-layout)
3. [Exploratory data analysis](#1-exploratory-data-analysis)
4. [Training](#2-training)
5. [Results and model selection](#3-results-and-model-selection)
6. [The decision threshold](#4-the-decision-threshold)
7. [The application](#5-the-application)
8. [Setup: run the app locally](#6-setup-run-the-app-locally)
9. [Docker](#7-docker)
10. [Deploying to Render](#8-deploying-to-render)
11. [Tests and CI](#9-tests-and-ci)
12. [Limitations and responsible use](#10-limitations-and-responsible-use)
13. [Roadmap](#11-roadmap)
14. [Data, acknowledgements and license](#12-data-acknowledgements-and-license)

---

## Overview

CastLens is an end-to-end computer-vision project for quality inspection of cast parts: data analysis, model training and comparison,
and a deployable web application.

**The problem.** A casting is photographed top-down. The task is to decide whether the part is defective (`def_front`, the positive class)
or good (`ok_front`). Missing a defect is more costly than flagging a good part for a second look, so the system is tuned for
defect recall, not plain accuracy.

**What was built**

| Stage | What happens | Where |
|---|---|---|
| Data analysis | Counts, balance, image quality, duplicates and leakage, brightness confound | `notebooks/01_eda.ipynb` |
| Training | Three models trained under one identical protocol, evaluated honestly | `notebooks/02_training.ipynb` |
| Export | PyTorch checkpoint converted to ONNX, with a numerical parity check | `scripts/export_onnx.py` |
| Application | FastAPI backend + React/Framer Motion frontend in one Docker image | `backend/`, `frontend/`, `Dockerfile` |

**Result at a glance.** The selected model is **EfficientNet-B0, fully fine-tuned**. On the 651-image test set it reaches
99.85% accuracy, 99.78% defect recall (95% lower bound 98.76%) and 100% specificity, with **1 missed defect and 0 false alarms**,
using a decision threshold of **0.21**. Sections 3 and 4 explain why this model was chosen and how far these numbers can be trusted.

---

## Repository layout

```
Castlens/
├── backend/
│   ├── app/
│   │   ├── main.py              FastAPI app: routes, upload limits, rate limiting, security headers
│   │   └── inference.py         ONNX Runtime classifier, preprocessing, threshold decision
│   ├── tests/                   pytest suite (API contract, preprocessing parity, export parity)
│   ├── requirements.txt         runtime dependencies (no PyTorch)
│   └── requirements-dev.txt     adds PyTorch, onnx, pytest for export and tests
├── frontend/                    Vite + React + TypeScript + Framer Motion
├── model/                       trained model: .pt (source), .onnx (runtime), inference_config.json
├── notebooks/
│   ├── 01_eda.ipynb             exploratory data analysis
│   └── 02_training.ipynb        training, evaluation, robustness, Grad-CAM (recorded run)
├── scripts/
│   └── export_onnx.py           .pt -> .onnx with parity verification
├── docs/
│   └── design-system.md         UI design decisions
├── Dockerfile                   multi-stage build (Node -> Python runtime)
├── render.yaml                  Render Blueprint
├── Makefile                     common commands
└── .github/workflows/ci.yml     tests, frontend build, Docker build
```

---

## 1. Exploratory data analysis

Notebook: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb). The goal was to understand the data before any model was trained, so that
preprocessing, augmentation, loss and evaluation choices are justified by evidence instead of guessed.

### 1.1 Dataset

The data is the Kaggle dataset
[Real-life Industrial Dataset of Casting Product](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product):
top-down photographs of a cast part, organised as

```
casting_data/
├── train/{def_front, ok_front}
└── test/{def_front, ok_front}
```

### 1.2 Size and structure

| Split | `def_front` (defective) | `ok_front` (good) | Total |
|---|---:|---:|---:|
| train | 3,758 | 2,875 | 6,633 |
| test | 453 | 262 | 715 |
| **total** | **4,211** | **3,137** | **7,348** |

Two classes, 7,348 images in total.

### 1.3 Class balance

| Scope | Defective | Good | Ratio (majority : minority) |
|---|---:|---:|---:|
| Overall | 57.31% | 42.69% | 1.34 : 1 |
| Train | 56.66% | 43.34% | 1.31 : 1 |
| Test (as shipped) | 63.36% | 36.64% | 1.73 : 1 |

The imbalance is **mild**: well under 1.5 : 1, so aggressive resampling is not needed. Two points matter for evaluation:

* The *defective* class is the **majority**, so a model that always answers "defect" would score well on accuracy. Accuracy alone is misleading.
* The test split is more skewed than train, so test accuracy cannot be compared naively with train accuracy.

### 1.4 Image properties and integrity

* **Resolution and format:** every sampled image (2,000 of 7,348) is exactly **300 × 300 px, RGB, JPEG**.
* **Corrupt files:** a check over the **full** dataset found **0** unreadable images.
* **Framing:** the parts are centred and consistently framed, so the average image per class is nearly identical in silhouette. The signal that
  separates the classes is therefore **local surface and edge detail**, not overall shape.

### 1.5 Duplicates and train/test leakage

An MD5 hash of every file found **64 duplicate groups (128 files)**, and **all 64 groups span both train and test**. Identical images in both splits
inflate test accuracy, so this is direct **data leakage**. Later in the project it turned out that all 64 duplicated test images were good parts
(`ok_front`), which is why removing them shifts the test set from 63.4% to 69.6% defective.

### 1.6 Brightness: a possible shortcut

| Class | Mean brightness (0–255) | Std |
|---|---:|---:|
| `ok_front` | 150.3 | 6.2 |
| `def_front` | 139.4 | 5.7 |

Good parts are about **7% brighter** on average. This could be a lighting artifact, or partly genuine (rough or damaged surfaces can look darker),
and the EDA cannot tell which. It is a **confound**: a model could learn "darker means defective" instead of looking for defects. This finding drove
three later steps: a brightness-only baseline (section 2.4), strong brightness jitter during training (section 2.5), and robustness tests (section 3.3).

### 1.7 What the EDA changed

| Finding | Decision |
|---|---|
| Mild imbalance (1.31 : 1 in train) | Cross-entropy with near-uniform per-image weights; no oversampling |
| Defective is the majority class, and a miss is costly | Judge models by defect recall, specificity and error counts, not accuracy; tune the threshold for cost |
| 64 exact duplicates across train and test | Remove them from the test set; later also check near-duplicates |
| Test split imbalance differs from train | Report class-aware metrics with confidence intervals |
| Uniform 300 × 300 RGB JPEG | Plain resize to 224 × 224 and ImageNet normalisation |
| Defect signal lives in fine edge and surface detail | Conservative augmentation: no elastic or shear distortion |
| Brightness differs by class | Brightness-only baseline, strong brightness jitter, brightness robustness tests |

---

## 2. Training

Notebook: [`notebooks/02_training.ipynb`](notebooks/02_training.ipynb). It is a single Colab notebook (GPU) that cleans the data, builds a
leak-free split, trains **three models under one identical protocol**, evaluates them, stress-tests them, and saves everything needed for deployment.

### 2.1 Pipeline

```
raw data -> readability check -> exact-duplicate removal -> near-duplicate analysis -> grouped train/val split
        -> brightness-only baseline -> augmentation + loaders -> train 3 models -> evaluate (eval mode, no augmentation)
        -> tune threshold on validation -> robustness + error analysis + Grad-CAM -> save weights and inference config
```

### 2.2 Data hygiene

**Exact duplicates.** The 64 test images that were byte-identical to training files were removed (test: 715 → **651**). The notebook also checks that duplicate copies
agree on their label and would drop any with conflicting labels.

**Near-duplicates (perceptual hash).** Exact hashes cannot catch resized, recompressed, flipped or rotated copies, so a perceptual hash (pHash) was used.
The key lesson: *a threshold only means something relative to a baseline*, because all images are the same part and look alike.

| Measurement (nearest neighbour within Hamming distance 2) | Share |
|---|---:|
| train → other train images (baseline) | 34.8% |
| test → train images (original orientation) | 36.3% |
| test → train including flips and rotations | 70.5% (459 of 651) |

The test share (36.3%) is almost the same as the within-train baseline (34.8%), so close look-alikes are a **property of the dataset**, not a leak specific
to the test set. The 70.5% figure takes the best match over six orientations, so it is not directly comparable to the baseline. Because the dataset has no
part identifiers, look-alikes cannot be separated from truly independent parts.

To stay honest, results are reported on **two test views**:

* **test**: all 651 images that remain after exact-duplicate removal (the headline view);
* **test_strict**: the 192 images with no near-duplicate in train (173 defective, only 19 good). Its specificity is weakly tested because of the 19 good parts.

Twelve flagged pairs had near-identical hashes but different labels. They should be inspected at full resolution as possible label noise.

### 2.3 Splits

Near-copies inside the training pool were grouped (4,918 groups; largest 41 images), and the validation split uses `StratifiedGroupKFold` so a group never
sits on both sides.

| Split | Images | Defective share |
|---|---:|---:|
| Train | 5,304 | 56.5% |
| Validation | 1,329 | 56.7% |
| Test | 651 | 69.6% |

### 2.4 Brightness-only baseline

A logistic regression that sees **only the mean and standard deviation of brightness** gives the bar every model must clear:

| Split | Accuracy | AUC | Defect recall | Specificity |
|---|---:|---:|---:|---:|
| Validation | 0.818 | 0.892 | 0.847 | 0.779 |
| Test | 0.831 | 0.902 | 0.854 | 0.778 |

Brightness is a real but incomplete signal. A model scoring near this level has learned little more than lighting.

### 2.5 Preprocessing and augmentation

| Stage | Operations |
|---|---|
| Training | Resize to 224 × 224 → random rotation ±15° → horizontal flip → vertical flip → colour jitter (brightness ±25%, contrast ±20%) → tensor → ImageNet normalisation |
| Evaluation and serving | Resize to 224 × 224 (bilinear) → tensor → ImageNet normalisation |

* Brightness jitter (±25%) is deliberately **larger than the class gap (about 7%)**, so average brightness is a poor shortcut.
* Flips and ±15° rotations assume the centred part is roughly rotation-symmetric. Set `USE_VFLIP = False` if defects turn out to be orientation-specific.
* No elastic or shear distortion, because the defect signal sits in fine edges.
* An optional switch, `NORMALIZE_BRIGHTNESS`, equalises every image's mean brightness. It stayed off because the robustness tests showed no shortcut.

### 2.6 Handling class imbalance

* **Stratified, grouped split** keeps the 57 / 43 mix in train and validation.
* **Loss weights** start from "balanced" weights and multiply the defect class by 1.3. Plain balanced weighting would make each defect image count *less* than a good image (defects are the majority), which is the opposite of the inspection priority. The result is about **[1.15, 1.15]**, i.e. nearly uniform per image.
* **The cost preference is handled by the decision threshold**, not the loss (section 4).
* **Metrics that do not hide imbalance:** defect recall with a Wilson confidence interval, specificity, precision, F1, ROC-AUC, PR-AUC, error counts and confusion matrices.

### 2.7 Models

| Key | Model | Strategy | Total params | Trainable params |
|---|---|---|---:|---:|
| `custom_cnn` | Small CNN | Trained from scratch | 405,954 | 405,954 |
| `effnet_b0_finetune` | EfficientNet-B0 | ImageNet weights, **all** layers trained | 4,010,110 | 4,010,110 |
| `effnet_b1_frozen` | EfficientNet-B1 | ImageNet weights, backbone **frozen** (transfer learning), only the new head trained | 6,515,746 | 2,562 |

* **Custom CNN:** four blocks of Conv3×3 → BatchNorm → ReLU → MaxPool (32, 64, 128, 256 channels), global average pooling, then Dropout 0.3 → Linear 256→64 → ReLU → Dropout 0.3 → Linear 64→2.
* **EfficientNets:** torchvision models with ImageNet-1k weights and a new `Linear(1280, 2)` head.
* **Frozen B1** runs its backbone in `eval()` mode throughout training, so BatchNorm statistics and stochastic depth do not drift.
* Comparing fine-tuned **B0** with frozen **B1** changes both the architecture and the strategy, so a gap between them cannot be credited to fine-tuning alone.

### 2.8 Training policy (identical for all models)

| Setting | Value |
|---|---|
| Loss | Cross-entropy with class weights ≈ [1.15, 1.15] |
| Optimiser | AdamW, weight decay 1e-4 |
| Learning rate | CNN 3e-4 · B0 fine-tune 1e-4 · B1 head 1e-3 |
| Schedule | ReduceLROnPlateau on validation loss (factor 0.5, patience 2) |
| Gradient clipping | 1.0 |
| Batch size / image size | 32 / 224 × 224 |
| Max epochs / early stopping | 20 / patience 5 |
| Model selection | Lowest **validation loss** (never the test set) |
| Seed | 42 (seeded data loaders, deterministic cuDNN) |

### 2.9 Training runs

| Model | Epochs run | Best epoch | Best val loss | Val accuracy (at 0.5) | Training time |
|---|---:|---:|---:|---:|---:|
| Custom CNN | 16 (early stop) | 11 | 0.0549 | 0.9827 | 8.1 min |
| EfficientNet-B0 fine-tuned | 12 (early stop) | 7 | 0.0089 | 0.9970 | 7.0 min |
| EfficientNet-B1 frozen | 20 (max) | 19 | 0.0515 | 0.9902 | 10.2 min |

Frozen B1 was still improving at epoch 20, so more epochs might help it. The Custom CNN's validation accuracy swung between about 0.55 and 0.98 from epoch to epoch
while training accuracy rose smoothly (section 3.5).

### 2.10 Evaluation protocol

* Every number comes from **eval mode with no augmentation**.
* The decision threshold is chosen on **validation only** and applied unchanged to test.
* Two test views are reported (full and strict), plus 95% Wilson intervals on defect recall.
* Models are stress-tested with a brightness sweep (×0.7 to ×1.3) and a brightness-equalisation test, and inspected with Grad-CAM and an error gallery.
* The notebook's automatic recommendation rule (lowest validation cost, ties go to the smaller model) is only a starting point; the final choice also weighs stability and robustness.

### 2.11 Reproducibility and saved artefacts

Running the notebook on Colab with the data at `/content/drive/MyDrive/casting_data.zip` writes to `MyDrive/casting_v2/baseline/`:

| File | Purpose |
|---|---|
| `<model>.pt` | Best-epoch weights (saved on every improvement, so a disconnect does not lose them) |
| `<model>_history.json` | Per-epoch losses and metrics |
| `inference_config.json` | Class mapping, image size, normalisation, **per-model threshold** |
| `results_all.csv`, `robustness_*.csv`, `brightness_baseline.csv` | Result tables |
| `split_manifest.csv` | Exact train / val / test membership, groups and near-duplicate flags |

Re-running with saved weights present loads them instead of retraining; set `FORCE_RETRAIN = True` to retrain.

---

## 3. Results and model selection

### 3.1 Test results (651 images, thresholds tuned on validation)

| Model | Threshold | Accuracy | Defect recall (95% lower bound) | Specificity | F1 | AUC | Missed defects | False alarms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **EfficientNet-B0 fine-tuned** | 0.21 | **0.9985** | 0.9978 (0.988) | 1.0000 | 0.9989 | 1.0000 | **1** | **0** |
| Custom CNN | 0.10 | 0.9877 | 0.9934 (0.981) | 0.9747 | 0.9912 | 0.9995 | 3 | 5 |
| EfficientNet-B1 frozen | 0.41 | 0.9892 | 0.9956 (0.984) | 0.9747 | 0.9923 | 0.9985 | 2 | 5 |

At the default threshold of 0.5 (shown for transparency):

| Model | Accuracy | Defect recall | Missed defects | False alarms |
|---|---:|---:|---:|---:|
| EfficientNet-B0 fine-tuned | 0.9969 | 0.9956 | 2 | 0 |
| Custom CNN | 0.9539 | 0.9338 | **30** | 0 |
| EfficientNet-B1 frozen | 0.9923 | 0.9934 | 3 | 2 |

**Strict test subset (192 images: 173 defective, 19 good):** every model reaches at least 0.9896 accuracy and 0.9942 recall, with 1 missed defect each.
EfficientNet-B0 fine-tuned and B1 frozen make no false alarms.

**Validation (1,329 images):** B0 fine-tuned 3 missed / 0 false alarms · Custom CNN 1 / 6 · B1 frozen 9 / 5.

### 3.2 Generalisation gap

Train, validation and test accuracy, all measured the same way (eval mode, no augmentation, threshold 0.5):

| Model | Train | Validation | Test |
|---|---:|---:|---:|
| EfficientNet-B0 fine-tuned | 0.9974 | 0.9970 | 0.9969 |
| EfficientNet-B1 frozen | 0.9900 | 0.9902 | 0.9923 |
| Custom CNN | 0.9840 | 0.9827 | 0.9539 |

### 3.3 Robustness

**Brightness sweep** (test accuracy, each model at its tuned threshold):

| Model | ×0.7 | ×0.9 | ×1.0 | ×1.1 | ×1.2 | ×1.3 | Brightness equalised |
|---|---:|---:|---:|---:|---:|---:|---:|
| EfficientNet-B0 fine-tuned | 0.9985 | 0.9985 | 0.9985 | 1.0000 | 1.0000 | 1.0000 | 0.9985 |
| EfficientNet-B1 frozen | 0.9923 | 0.9892 | 0.9892 | 0.9831 | 0.9831 | 0.9770 | 0.9892 |
| Custom CNN | 0.8971 | 0.9816 | 0.9877 | 0.9908 | 0.9401 | 0.7558 | 0.9770 |

Equalising every test image to the same mean brightness does not change B0 or B1 and costs the Custom CNN 1.1 points. There is **no evidence that the models rely on the
average-brightness gap**, and the brightness-only baseline (AUC 0.90) is far below the CNNs. The Custom CNN is, however, fragile to larger lighting shifts.

### 3.4 Where the models look, and the errors

* **Grad-CAM** (eight test images inspected): B0 fine-tuned concentrates on the bore and rim of defective parts and stays almost cold on good parts. B1 frozen puts much of its heat on the background and image corners for both classes, which suggests it uses context rather than the part (its 7×7 map is coarse, so treat this as a warning, not proof).
* **Errors:** 14 of 651 test images are misclassified by at least one model. B0 fine-tuned misses one: `cast_def_0_1591`, labelled defective, which looks clean at thumbnail size and which B1 also misses. It should be checked at full resolution for label noise.

### 3.5 Which model is best, and why

**EfficientNet-B0, fully fine-tuned, is the selected model.**

* **Fewest errors:** 1 error in 651 images, with no false alarms.
* **Stable training:** validation accuracy sits near 0.997 from the first epochs, and the best epoch (7) is clear.
* **No overfitting:** train, validation and test accuracy agree within 0.1 point.
* **Insensitive to the threshold:** 2 errors at 0.5 versus 1 at the tuned 0.21.
* **Robust to lighting:** at least 0.9985 accuracy across the whole brightness sweep, unchanged under brightness equalisation.
* **Looks at the part:** Grad-CAM evidence sits on the bore and rim of defective parts.

**Why not the Custom CNN.** It is the smallest and its validation cost was within a tie of B0's, but:

* its validation accuracy swung between about 0.55 and 0.98 across epochs while training accuracy was smooth (consistent with unstable BatchNorm statistics in eval mode), so the saved model is a lucky epoch;
* its probabilities are poorly calibrated: at 0.5 it misses 30 defects, and it only works at a low threshold of 0.10 fitted on that noisy validation set;
* it drops to 0.94 accuracy at ×1.2 brightness and 0.76 at ×1.3, even though training used ±25% jitter;
* it missed a part with a visibly rough rim.

**Why not frozen B1.** It has the weakest validation results (9 missed, 5 false alarms), the most parameters (6.5M, so it is the heaviest to *run*, even though only 2,562 are trained), and its Grad-CAM heat falls largely on the background. Freezing the backbone reduces *training* cost, not inference cost.

> The notebook's automatic rule (lowest validation cost, ties go to the smaller model) picked the Custom CNN because its validation cost (11) was within the tie tolerance of B0's (15).
> That rule ignores stability and robustness, so the choice was overridden. The saved `inference_config.json` still lists `custom_cnn` as `recommended`; the app ignores that field and loads
> `MODEL_KEY` (default `effnet_b0_finetune`).

### 3.6 Caveats

* **The test set is nearly saturated.** B0's lead over B1 is 6 images out of 651 and the confidence intervals overlap. The choice rests on validation behaviour, robustness and Grad-CAM, not on test accuracy alone. The 95% lower bound on B0's defect recall (0.988) means up to about 1.2% of defects could still be missed.
* **The false-alarm rate on genuinely new parts is the least certain number.** The strict subset has only 19 good parts, and the dataset has no part identifiers.
* **Single split, single seed, one part type and one camera setup.** A gap of a few images between models could change with another seed, and other lighting, cameras or part variants are untested.
* **Possible label noise** in `cast_def_0_1591` and in the 12 near-duplicate pairs with conflicting labels.

---

## 4. The decision threshold

### 4.1 Probability versus decision

The model outputs, for each photo, a probability that the part is defective, `p_defect`, and one that it is good, `p_ok = 1 − p_defect`. Turning that into a verdict needs a **threshold**:

```
verdict = "defect"  if  p_defect >= threshold  else  "ok"
```

The common default is 0.5, which is the same as "pick the higher probability". But the two errors are not equally costly: a missed defect (false negative) reaches a customer, while a false alarm
(false positive) costs one extra look. So the threshold is **set deliberately below 0.5**.

### 4.2 How it was chosen

On the **validation** set, for thresholds from 0.02 to 0.98, the notebook computed

```
cost(t) = COST_FN × (missed defects) + COST_FP × (false alarms)        with COST_FN = 5, COST_FP = 1
```

and picked the threshold with the lowest cost (ties resolve to the middle of the tied range). The test set was never used to choose it.

*Sanity check.* If probabilities were perfectly calibrated, the cost-optimal threshold for a 5 : 1 cost ratio would be `1 / (1 + 5) ≈ 0.17`. The chosen 0.21 is in the same neighbourhood.

### 4.3 Thresholds per model

| Model | Tuned threshold | Test errors at tuned | Test errors at 0.5 |
|---|---:|---:|---:|
| EfficientNet-B0 fine-tuned | **0.21** | 1 (1 missed, 0 false alarms) | 2 (2 missed, 0 false alarms) |
| Custom CNN | 0.10 | 8 (3 missed, 5 false alarms) | 30 (30 missed, 0 false alarms) |
| EfficientNet-B1 frozen | 0.41 | 7 (2 missed, 5 false alarms) | 5 (3 missed, 2 false alarms) |

B0's results barely depend on the threshold, which is a sign of a well-separated model. The Custom CNN swings from 8 to 30 errors, which is a sign of poor calibration. The thresholds were tuned
on validation sets with very few errors (3 for B0), so the exact values are noisy.

### 4.4 How the app uses it

The API returns both numbers, because they can disagree:

| Field | Meaning |
|---|---|
| `highest` | Whichever class has the larger probability (plain argmax, i.e. threshold 0.5) |
| `verdict` | The decision using the tuned threshold (0.21) |
| `borderline` | `true` when `verdict` and `highest` differ |

Example: `p_defect = 0.30` gives `highest = "ok"` (70% vs 30%) but `verdict = "defect"`, because 0.30 ≥ 0.21. The UI shows the verdict, both probability bars with the threshold marked on the defect
bar, a "Higher" tag on the larger probability, and a plain-language note when the case is borderline. This is deliberate: borderline parts are flagged for a second look instead of passing.

### 4.5 Changing the threshold

The app reads the threshold from `model/inference_config.json` at startup (`models.effnet_b0_finetune.threshold`). To change it, edit that value and restart: no retraining and no re-export are needed.
To re-derive it from a different cost ratio, change `COST_FN` and `COST_FP` in the notebook's CONFIG and re-run the threshold section. A threshold is tied to its checkpoint, so replace it together with the weights if you retrain.

---

## 5. The application

```
Browser (React + Framer Motion)
   │  POST /api/predict   (multipart: file)
   ▼
FastAPI ── size / format / rate-limit checks ── PIL decode ── preprocess (resize 224, normalise)
   ▼
ONNX Runtime (EfficientNet-B0) ── softmax ── apply threshold ── JSON response
```

In production the same FastAPI process also serves the built frontend, so the whole application is **one container and one port**.

### 5.1 Features

* Upload by drag-and-drop, file picker or keyboard; the check runs immediately.
* Verdict, **both probabilities** (OK and defect), a "Higher" tag on the larger one, and the threshold marked on the defect bar.
* A borderline explanation when `verdict` and `highest` differ.
* Warnings for small or non-square images.
* Friendly error messages (file too large, not an image, too many requests, server unreachable).
* A scan-line animation while the part is checked and spring-animated probability bars (Framer Motion), with reduced-motion preferences respected.
* Accessible by design: visible keyboard focus, text and icons next to colour, a live region for results, responsive layout.

### 5.2 API

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/predict` | Multipart field `file`. Returns the prediction below |
| `GET` | `/api/model` | Model name, build hash, threshold, input size, classes |
| `GET` | `/api/health` | Liveness (the app does not start without a model) |
| `GET` | `/api/docs` | Interactive OpenAPI documentation |

Example `POST /api/predict` response (illustrative values):

```json
{
  "verdict": "defect",
  "highest": "ok",
  "borderline": true,
  "probabilities": { "ok": 0.70, "defect": 0.30 },
  "threshold": 0.21,
  "warnings": [],
  "model": { "name": "effnet_b0_finetune", "version": "065136fdc4e6" },
  "inference_ms": 52.0
}
```

```bash
curl -F "file=@part.jpg" http://localhost:8000/api/predict
```

Error codes: `413` file too large, `422` unreadable or unsupported image, `429` rate limit exceeded.

### 5.3 Inference details

* The serving preprocessing is the notebook's evaluation pipeline: RGB → bilinear resize to 224 × 224 → divide by 255 → ImageNet normalisation. A test compares it against the torchvision transforms to guard against train/serve skew.
* The model is served with **ONNX Runtime** instead of PyTorch: the image stays small, memory use is lower and start-up is faster. `scripts/export_onnx.py` refuses to write an ONNX file whose probabilities differ from PyTorch by more than 1e-4.
* Class index 0 is `ok_front`, index 1 is `def_front` (defective), the same as in training.

### 5.4 Security and privacy

* Upload cap of 8 MB (checked from `Content-Length` and again while reading), a format whitelist (JPEG, PNG, WebP, BMP) verified by actually decoding the file, and a decompression-bomb guard (25 megapixels).
* Per-IP rate limit (default 30 per minute). It is in memory, per instance, and reset on restart, so it is a courtesy limit, not a security boundary.
* Security headers (Content-Security-Policy, `nosniff`, no-referrer, frame denial); `no-store` on API responses.
* Images are processed in memory, **never written to disk, and never logged**; only the verdict, probability, latency and byte size are logged.
* The container runs as a non-root user. There is **no authentication**; see section 10.

### 5.5 Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_KEY` | `effnet_b0_finetune` | Which model in `inference_config.json` to load (`<key>.onnx` must exist) |
| `MODEL_DIR` | `<repo>/model` | Folder with the `.onnx` and `inference_config.json` |
| `MAX_UPLOAD_MB` | `8` | Maximum upload size |
| `RATE_LIMIT_PER_MINUTE` | `30` | Requests per IP per minute on `/api/predict` |
| `CORS_ORIGINS` | *(empty)* | Comma-separated origins allowed for cross-origin calls (not needed when the frontend is served by the same app) |
| `STATIC_DIR` | `<repo>/static` | Built frontend to serve |
| `LOG_LEVEL` | `INFO` | Logging level |
| `PORT` | `10000` (Docker) | Port the container listens on |

---

## 6. Setup: run the app locally

### 6.1 Prerequisites

* Python 3.12 and Node.js 22 (older Node versions that Vite 6 supports also work)
* Git; Docker is optional (section 7)

### 6.2 Get the model files

The app will not start without the trained model. From the training run's Drive folder (`casting_v2/baseline/`), copy these two files into `model/`:

* `effnet_b0_finetune.pt`
* `inference_config.json`

Then create the runtime model:

```bash
python -m venv .venv && source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu    # CPU build, avoids multi-GB CUDA downloads
pip install -r backend/requirements-dev.txt
python scripts/export_onnx.py                            # or: make export
```

This writes `model/effnet_b0_finetune.onnx` after verifying that it matches the PyTorch model. Commit the three files in `model/` (about 16 MB each; no Git LFS needed) so that Docker and Render can use them.
See [`model/README.md`](model/README.md).

### 6.3 Run the backend

```bash
cd backend
uvicorn app.main:app --reload --port 8000                # or, from the repo root: make dev-api
```

Check <http://localhost:8000/api/health> and the interactive docs at <http://localhost:8000/api/docs>.

### 6.4 Run the frontend (development)

In a second terminal:

```bash
cd frontend
npm install
npm run dev                                              # or, from the repo root: make dev-web
```

Open <http://localhost:5173>. The Vite dev server proxies `/api` to `localhost:8000`.

### 6.5 Run the production build without Docker

```bash
cd frontend && npm ci && npm run build && cd ..
cp -r frontend/dist backend/static                       # FastAPI serves ./static when it exists
cd backend && uvicorn app.main:app --port 8000
```

Open <http://localhost:8000>.

### 6.6 Make targets

| Command | What it does |
|---|---|
| `make export` | Convert `model/*.pt` to ONNX and verify parity |
| `make test` | Run the backend tests |
| `make dev-api` | Start the API with auto-reload on port 8000 |
| `make dev-web` | Start the frontend dev server on port 5173 |
| `make build` | Build the Docker image (`castlens`) |
| `make run` | Run the image on port 10000 |

---

## 7. Docker

The `Dockerfile` is a two-stage build:

1. **`web` stage** (`node:22-alpine`): installs the frontend dependencies with `npm ci` and runs `npm run build`.
2. **Runtime stage** (`python:3.12-slim`): installs `backend/requirements.txt` (FastAPI, ONNX Runtime, Pillow, NumPy; **no PyTorch**), copies the app, the `model/` folder and the built frontend, and runs as a non-root user (`uid 10001`).

`.dockerignore` keeps the image lean: it excludes notebooks, scripts, tests, docs, `node_modules` and `model/*.pt` (the PyTorch checkpoint is not needed at runtime).

### 7.1 Build and run

```bash
docker build -t castlens .
docker run --rm -p 10000:10000 castlens
```

Open <http://localhost:10000>. The image contains a health check that polls `/api/health`.

### 7.2 Useful variations

```bash
# change a setting
docker run --rm -p 10000:10000 -e MAX_UPLOAD_MB=4 -e RATE_LIMIT_PER_MINUTE=10 castlens

# swap the model without rebuilding (bind-mount the model folder read-only)
docker run --rm -p 10000:10000 -v "$(pwd)/model:/srv/model:ro" castlens     # PowerShell: -v "${PWD}/model:/srv/model:ro"

# run on another port
docker run --rm -e PORT=8080 -p 8080:8080 castlens
```

### 7.3 Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Container exits with `Missing .../model/...` | `model/effnet_b0_finetune.onnx` or `inference_config.json` is missing from the build context. Run `make export` and check `git ls-files model` before building |
| `npm ci` fails during the build | `frontend/package-lock.json` is missing from the repository |
| Page loads but uploads fail with 429 | Rate limit reached; raise `RATE_LIMIT_PER_MINUTE` |
| Out-of-memory kills on small hosts | Make sure the runtime image is the ONNX one (no PyTorch); check the host's memory limit |
| Port in use | Map another host port, e.g. `-p 8080:10000` |

---

## 8. Deploying to Render

`render.yaml` is a Blueprint that builds the Dockerfile and uses `/api/health` as the health check.

1. Make sure the three model files are committed and pushed to GitHub (`git ls-files model` must list them).
2. In the Render dashboard choose **New → Blueprint**, connect the repository and apply.
3. The first build takes a few minutes. Open the service URL and check `/api/health`, then upload a photo. The live deployment of this project is at <https://castlens.onrender.com/>.

Notes on plans: the free plan has 512 MB of RAM and sleeps after about 15 minutes of inactivity, so the first request after a pause is slow. The ONNX runtime is meant to fit in 512 MB, but check the
**Metrics** tab after a few uploads. Plan names and limits change, so confirm them on Render's pricing page. `autoDeploy: true` redeploys on every push to the default branch.

---

## 9. Tests and CI

```bash
make test        # or: cd backend && PYTHONPATH=. pytest -q
```

The suite builds a small random-weight model, so it tests the **pipeline and API contract, not accuracy**:

* the threshold logic (`verdict`, `highest`, `borderline`);
* `/api/health`, `/api/model` and the `/api/predict` response contract (probabilities sum to 1, headers present);
* rejection of non-images (422) and oversized uploads (413), and warnings for small or non-square images;
* **preprocessing parity** with the notebook's torchvision transforms;
* **export parity** between PyTorch and ONNX (checked every time a model is exported).

GitHub Actions (`.github/workflows/ci.yml`) runs the backend tests, builds the frontend and builds the Docker image on every push and pull request.

---

## 10. Limitations and responsible use

* **Scope of the model.** It was trained on one part type and one camera setup. There is **no out-of-distribution check**, so a photo of anything else still receives a confident score. The API only warns on small or non-square images.
* **Evidence is strong but limited.** The test set is nearly saturated (1 error in 651), the false-alarm rate on genuinely new parts is the least certain number, and results come from a single split and seed (section 3.6).
* **A decision aid, not a replacement for inspection.** The threshold deliberately flags borderline parts for review. Keep human inspection and audited sampling in the loop, and track the missed-defect rate in use.
* **No authentication, and rate limiting is per instance.** Do not expose the service publicly without adding access control if the images are sensitive.
* **Privacy.** Uploaded photos are processed in memory and not stored or logged by this application.
* **Label noise.** A few images may be mislabelled (section 3.6); verify them before relying on the exact error counts.

---

## 11. Roadmap

* Out-of-distribution detection (feature statistics from the training set) to reject photos that are not castings.
* Grad-CAM heatmap in the UI to show where the model looked (the notebook already computes it).
* Seed variance study or grouped k-fold to quantify uncertainty between models.
* An independent test set from different batches, ideally with part identifiers.
* Authentication and a shared rate limiter for public deployments; batch upload.
* Monitoring of prediction distribution and audited miss rate in production, and threshold re-tuning when lighting or parts change.

---

## 12. Data, acknowledgements and license

* **Data:** [Real-life Industrial Dataset of Casting Product](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) on Kaggle. Please follow the dataset's own terms of use.
* **Built with:** PyTorch and torchvision (training), ONNX Runtime (inference), FastAPI, React, Vite, Framer Motion.
* **UI design process:** the checklist from the [ui-ux-pro-max-skill](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill) project was applied; see [`docs/design-system.md`](docs/design-system.md).
* **License:** add a `LICENSE` file to the repository to state the terms for this code.
