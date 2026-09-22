# Grass Segmentation Training Log

This document is the canonical experiment log for the grass segmentation model. Add an entry after every training
run, including failed runs, and keep the comparison table synchronized with the detailed records.

## Dataset

- Task: single-class (`grass`) instance segmentation
- Train: 2,724 images and 2,724 labels
- Validation: 1,170 images and 1,170 labels (1,355 instances)
- Config: `mower_grass_seg/data/yolo_seg/grass.yaml`
- Known risk: audit whether adjacent frames from the same recording are split across train and validation before
  treating validation metrics as deployment estimates.

## Comparison

| ID | Date | Status | Model | Epochs | Fraction | Mosaic | Device | Best epoch | Box mAP50-95 | Mask mAP50-95 | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| S00 | 2026-09-22 | Failed | YOLOv8n-Seg | 0 | 0.02 | 1.0 | RTX A6000 GPU 2 | - | - | - | AMP check could not find `ultralytics/assets/bus.jpg` |
| S01 | 2026-09-22 | Passed | YOLOv8n-Seg | 1 | 0.02 | 1.0 | RTX A6000 GPU 2 | 1 | 0.00072 | 0.00003 | Pipeline smoke test only; metrics are not meaningful |
| B01 | 2026-09-22 | Completed with warning | YOLOv8n-Seg | 100 | 1.0 | 0.0 | RTX A6000 GPU 2 | 97 | 0.94479 | 0.94882 | `train/box_loss=inf` at epochs 85 and 100 |

## S00 - Initial server GPU smoke test

- Purpose: verify the Linux CUDA training pipeline on physical GPU 2.
- Result: training stopped during the pre-training AMP check.
- Cause: this fork ignores `ultralytics/assets/`, so the AMP check could not open its required `bus.jpg` image.
- Resolution: downloaded the official Ultralytics `bus.jpg` into the ignored runtime assets directory. AMP remained
  enabled; no training logic was bypassed.
- Output directory: `mower_grass_seg/runs/segment/smoke`

## S01 - Successful server GPU smoke test

### Configuration

- Code commit: `bb45253`
- Ultralytics: 8.4.128 (editable repository install)
- Python: 3.10.21
- PyTorch: 2.6.0+cu124
- GPU: NVIDIA RTX A6000, physical GPU 2
- Model: `yolov8n-seg.pt` pretrained weights
- Epochs: 1
- Image size: 640
- Batch: 2
- Workers: 0
- Training fraction: 0.02 (54 training images)
- AMP: enabled and passed
- Optimizer: AdamW selected by `optimizer=auto`
- Mosaic: 1.0
- Seed: 0
- Deterministic: true

### Result

| Train box loss | Train seg loss | Train cls loss | Train DFL loss | Box mAP50-95 | Mask mAP50-95 | Duration |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.10921 | 4.95026 | 2.81278 | 1.36619 | 0.00072 | 0.00003 | 19.54 s |

- Conclusion: data loading, CUDA, AMP, training, validation, and checkpoint saving all completed successfully.
- Interpretation: quality metrics are intentionally ignored because the run used only 2% of the data for one epoch.
- Output directory: `mower_grass_seg/runs/segment/smoke-2`

## B01 - YOLOv8n-Seg baseline without Mosaic

### Configuration

- Code commit: `091ed93`
- Ultralytics: 8.4.128 (editable repository install)
- Python: 3.10.21
- PyTorch: 2.6.0+cu124
- CUDA runtime: 12.4
- NVIDIA driver: 550.120
- GPU: NVIDIA RTX A6000, physical GPU 2
- Model: `yolov8n-seg.pt` pretrained weights
- Epochs: 100
- Image size: 640
- Batch: 8
- Workers: 4
- Training fraction: 1.0
- AMP: enabled
- Optimizer: AdamW selected by `optimizer=auto`
- Mosaic: 0.0
- Seed: 0
- Deterministic: true
- Duration: 3,816.12 s (1 h 3 min 36 s)

### Selected checkpoints

Ultralytics selected epoch 97 for `best.pt` using the combined Box and Mask fitness. The highest Mask-only
mAP50-95 was 0.94891 at epoch 88, but its combined fitness was lower than epoch 97.

| Checkpoint | Epoch | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `best.pt` | 97 | 0.96192 | 0.95072 | 0.97969 | 0.94479 | 0.96266 | 0.95146 | 0.97999 | 0.94882 |
| `last.pt` | 100 | 0.96904 | 0.94465 | 0.97957 | 0.94325 | 0.96980 | 0.94539 | 0.97973 | 0.94727 |

### Training trend

| Epoch | Train box | Train seg | Train cls | Train DFL | Val box | Val seg | Mask mAP50-95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.51758 | 1.02069 | 1.31631 | 1.08500 | 0.59356 | 1.18779 | 0.78012 |
| 2 | 0.47367 | 0.65633 | 0.65951 | 1.04940 | 0.47188 | 0.72419 | 0.85561 |
| 85 | inf | 0.21076 | 0.13673 | 0.86885 | 0.23685 | 0.36383 | 0.94395 |
| 88 | 0.14178 | 0.21252 | 0.13382 | 0.87702 | 0.22723 | 0.35996 | 0.94891 |
| 97 | 0.12771 | 0.20334 | 0.11894 | 0.86719 | 0.22846 | 0.36194 | 0.94882 |
| 100 | inf | 0.19965 | 0.11895 | 0.86605 | 0.22300 | 0.36614 | 0.94727 |

### Artifacts

- Server output: `mower_grass_seg/runs/segment/yolov8n_seg`
- Best weight SHA-256: `a98d91bd96b29e15045d80c3b17b3fbf2ffd40cf5cdda266120d4261d957f711`
- Last weight SHA-256: `36c5f5c4573fdbcedfbfd8f03e28cd810fe6e39540a05a0643ad90a190a69fff`

### Findings

- Box and Mask validation metrics improved rapidly and remained stable near the end of training.
- `best.pt` is preferred over `last.pt` for evaluation and deployment comparisons.
- `train/box_loss` became infinite at epochs 85 and 100. No other CSV value was NaN or infinite, validation metrics
  remained finite, and epoch 97 was unaffected. Investigate the responsible batches or box-loss numerics before the
  next baseline rather than treating this run as fully clean.
- The unusually strong validation result from the first two epochs makes train/validation sequence leakage an
  important audit item.

### Next actions

1. Audit the split by source video or capture sequence and remove adjacent-frame leakage if present.
2. Diagnose the non-finite box loss at epochs 85 and 100.
3. Evaluate `best.pt` on representative videos and difficult day/night scenes.
4. Use B01 as the comparison reference only if the split audit passes.

## Entry template

### ID - Short experiment name

- Date:
- Objective:
- Code commit:
- Dataset/split version:
- Environment and GPU:
- Model and initialization:
- Epochs / image size / batch / workers:
- Optimizer / learning rate policy:
- Augmentations:
- Best epoch and selection criterion:
- Box P / R / mAP50 / mAP50-95:
- Mask P / R / mAP50 / mAP50-95:
- Train and validation loss behavior:
- Runtime and peak GPU memory:
- Warnings or failures:
- Artifact paths and weight checksums:
- Conclusion:
- Next action:
