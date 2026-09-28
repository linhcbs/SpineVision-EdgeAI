# HPE Model Evaluation Report -- Posture Classification

> **Generated:** 2026-09-28 10:52:28
> **Dataset:** Sitting Posture v4 (4-keypoint COCO format)
> **Task:** Binary Posture Classification (Good / Bad)
> **Splits evaluated:** train, valid, test
> **Positive class:** `Bad` (bad posture detection target)

---

## 1. Dataset Overview

| Split | Total Images | Good | Bad | Bad% |
|-------|-------------|------|-----|------|
| Train | 573 | 207 | 366 | 63.9% |
| Valid | 55 | 18 | 37 | 67.3% |
| Test | 27 | 10 | 17 | 63.0% |

> **Keypoints:** `bottom`, `shoulder`, `head`, `back` (4 custom anatomical points)
> **Labels:** category_id=1 -> `Bad`, category_id=2 -> `Good`

---

## 2. Models Under Evaluation

| # | Model | Keypoints | Weight Size | Backend | Input Size |
|---|-------|-----------|-------------|---------|------------|
| 1 | **MediaPipe Pose LITE** | 33 kps | 5642 KB | TFLite/MediaPipe | dynamic |
| 2 | **MediaPipe Pose FULL** | 33 kps | 9178 KB | TFLite/MediaPipe | dynamic |
| 3 | **MoveNet Lightning** | 17 kps | 4647 KB | TFLite | 192x192 |
| 4 | **MoveNet Thunder** | 17 kps | 12289 KB | TFLite | 256x256 |
| 5 | **YOLO26n-Pose (Nano)** | 17 kps | 6109 KB | PyTorch/Ultralytics | 640x640 |

> YOLO26m-Pose and YOLO26x-Pose are excluded per evaluation criteria.

---

## 3. Detailed Results by Model

### 3.1 MediaPipe Pose LITE

#### Classification Metrics

| Split | N | Accuracy | Precision | Recall | F1 | Specificity | TP | FP | FN | TN | No-Detection |
|-------|---|----------|-----------|--------|----|-------------|----|----|----|----|----|
| Train | 573 | 78.18% | 84.93% | 80.05% | 82.42% | 74.88% | 293 | 52 | 73 | 155 | 2 (0.3%) |
| Valid | 55 | 67.27% | 82.76% | 64.86% | 72.73% | 72.22% | 24 | 5 | 13 | 13 | 0 (0.0%) |
| Test | 27 | 70.37% | 76.47% | 76.47% | 76.47% | 60.00% | 13 | 4 | 4 | 6 | 0 (0.0%) |

#### Inference Latency (ms)

| Split | Mean | Std | P50 | P95 | P99 | Min | Max | FPS |
|-------|------|-----|-----|-----|-----|-----|-----|-----|
| Train | 26.31 | 4.66 | 25.71 | 31.52 | 40.56 | 15.73 | 86.82 | 38.0 |
| Valid | 26.82 | 2.73 | 26.08 | 31.9 | 32.97 | 23.58 | 33.89 | 37.3 |
| Test | 26.68 | 2.45 | 26.4 | 30.73 | 31.48 | 20.54 | 31.63 | 37.5 |

#### Per-Class Metrics -- Train

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Good | 67.98% | 74.88% | 71.26% | 207 |
| Bad | 84.93% | 80.05% | 82.42% | 366 |

#### Confusion Matrix -- Train

```
                 Predicted
                 Good    Bad
Actual  Good  [   155     52 ]
        Bad   [    73    293 ]
```

#### Per-Class Metrics -- Valid

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Good | 50.00% | 72.22% | 59.09% | 18 |
| Bad | 82.76% | 64.86% | 72.73% | 37 |

#### Confusion Matrix -- Valid

```
                 Predicted
                 Good    Bad
Actual  Good  [    13      5 ]
        Bad   [    13     24 ]
```

#### Per-Class Metrics -- Test

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Good | 60.00% | 60.00% | 60.00% | 10 |
| Bad | 76.47% | 76.47% | 76.47% | 17 |

#### Confusion Matrix -- Test

```
                 Predicted
                 Good    Bad
Actual  Good  [     6      4 ]
        Bad   [     4     13 ]
```

### 3.2 MediaPipe Pose FULL

#### Classification Metrics

| Split | N | Accuracy | Precision | Recall | F1 | Specificity | TP | FP | FN | TN | No-Detection |
|-------|---|----------|-----------|--------|----|-------------|----|----|----|----|----|
| Train | 573 | 78.36% | 89.80% | 74.59% | 81.49% | 85.02% | 273 | 31 | 93 | 176 | 2 (0.3%) |
| Valid | 55 | 67.27% | 82.76% | 64.86% | 72.73% | 72.22% | 24 | 5 | 13 | 13 | 0 (0.0%) |
| Test | 27 | 77.78% | 92.31% | 70.59% | 80.00% | 90.00% | 12 | 1 | 5 | 9 | 0 (0.0%) |

#### Inference Latency (ms)

| Split | Mean | Std | P50 | P95 | P99 | Min | Max | FPS |
|-------|------|-----|-----|-----|-----|-----|-----|-----|
| Train | 30.84 | 9.9 | 35.25 | 42.07 | 48.61 | 17.54 | 67.4 | 32.4 |
| Valid | 18.31 | 0.74 | 18.03 | 19.71 | 20.5 | 17.59 | 20.86 | 54.6 |
| Test | 19.38 | 1.61 | 18.79 | 23.56 | 24.57 | 17.96 | 24.6 | 51.6 |

#### Per-Class Metrics -- Train

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Good | 65.43% | 85.02% | 73.95% | 207 |
| Bad | 89.80% | 74.59% | 81.49% | 366 |

#### Confusion Matrix -- Train

```
                 Predicted
                 Good    Bad
Actual  Good  [   176     31 ]
        Bad   [    93    273 ]
```

#### Per-Class Metrics -- Valid

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Good | 50.00% | 72.22% | 59.09% | 18 |
| Bad | 82.76% | 64.86% | 72.73% | 37 |

#### Confusion Matrix -- Valid

```
                 Predicted
                 Good    Bad
Actual  Good  [    13      5 ]
        Bad   [    13     24 ]
```

#### Per-Class Metrics -- Test

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Good | 64.29% | 90.00% | 75.00% | 10 |
| Bad | 92.31% | 70.59% | 80.00% | 17 |

#### Confusion Matrix -- Test

```
                 Predicted
                 Good    Bad
Actual  Good  [     9      1 ]
        Bad   [     5     12 ]
```

### 3.3 MoveNet Lightning

#### Classification Metrics

| Split | N | Accuracy | Precision | Recall | F1 | Specificity | TP | FP | FN | TN | No-Detection |
|-------|---|----------|-----------|--------|----|-------------|----|----|----|----|----|
| Train | 573 | 67.02% | 66.86% | 95.90% | 78.79% | 15.94% | 351 | 174 | 15 | 33 | 0 (0.0%) |
| Valid | 55 | 67.27% | 67.92% | 97.30% | 80.00% | 5.56% | 36 | 17 | 1 | 1 | 0 (0.0%) |
| Test | 27 | 62.96% | 64.00% | 94.12% | 76.19% | 10.00% | 16 | 9 | 1 | 1 | 0 (0.0%) |

#### Inference Latency (ms)

| Split | Mean | Std | P50 | P95 | P99 | Min | Max | FPS |
|-------|------|-----|-----|-----|-----|-----|-----|-----|
| Train | 7.52 | 0.54 | 7.42 | 7.82 | 9.7 | 7.3 | 13.93 | 133.0 |
| Valid | 7.46 | 0.1 | 7.43 | 7.63 | 7.79 | 7.34 | 7.96 | 134.0 |
| Test | 7.43 | 0.05 | 7.41 | 7.52 | 7.57 | 7.36 | 7.59 | 134.6 |

#### Per-Class Metrics -- Train

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Good | 68.75% | 15.94% | 25.88% | 207 |
| Bad | 66.86% | 95.90% | 78.79% | 366 |

#### Confusion Matrix -- Train

```
                 Predicted
                 Good    Bad
Actual  Good  [    33    174 ]
        Bad   [    15    351 ]
```

#### Per-Class Metrics -- Valid

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Good | 50.00% | 5.56% | 10.00% | 18 |
| Bad | 67.92% | 97.30% | 80.00% | 37 |

#### Confusion Matrix -- Valid

```
                 Predicted
                 Good    Bad
Actual  Good  [     1     17 ]
        Bad   [     1     36 ]
```

#### Per-Class Metrics -- Test

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Good | 50.00% | 10.00% | 16.67% | 10 |
| Bad | 64.00% | 94.12% | 76.19% | 17 |

#### Confusion Matrix -- Test

```
                 Predicted
                 Good    Bad
Actual  Good  [     1      9 ]
        Bad   [     1     16 ]
```

### 3.4 MoveNet Thunder

#### Classification Metrics

| Split | N | Accuracy | Precision | Recall | F1 | Specificity | TP | FP | FN | TN | No-Detection |
|-------|---|----------|-----------|--------|----|-------------|----|----|----|----|----|
| Train | 573 | 66.32% | 70.35% | 81.69% | 75.60% | 39.13% | 299 | 126 | 67 | 81 | 0 (0.0%) |
| Valid | 55 | 67.27% | 70.21% | 89.19% | 78.57% | 22.22% | 33 | 14 | 4 | 4 | 0 (0.0%) |
| Test | 27 | 66.67% | 68.18% | 88.24% | 76.92% | 30.00% | 15 | 7 | 2 | 3 | 0 (0.0%) |

#### Inference Latency (ms)

| Split | Mean | Std | P50 | P95 | P99 | Min | Max | FPS |
|-------|------|-----|-----|-----|-----|-----|-----|-----|
| Train | 33.36 | 0.53 | 33.22 | 34.28 | 35.38 | 32.65 | 37.42 | 30.0 |
| Valid | 34.16 | 2.68 | 33.35 | 37.55 | 46.47 | 32.76 | 47.71 | 29.3 |
| Test | 33.29 | 0.33 | 33.24 | 34.02 | 34.06 | 32.83 | 34.06 | 30.0 |

#### Per-Class Metrics -- Train

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Good | 54.73% | 39.13% | 45.63% | 207 |
| Bad | 70.35% | 81.69% | 75.60% | 366 |

#### Confusion Matrix -- Train

```
                 Predicted
                 Good    Bad
Actual  Good  [    81    126 ]
        Bad   [    67    299 ]
```

#### Per-Class Metrics -- Valid

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Good | 50.00% | 22.22% | 30.77% | 18 |
| Bad | 70.21% | 89.19% | 78.57% | 37 |

#### Confusion Matrix -- Valid

```
                 Predicted
                 Good    Bad
Actual  Good  [     4     14 ]
        Bad   [     4     33 ]
```

#### Per-Class Metrics -- Test

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Good | 60.00% | 30.00% | 40.00% | 10 |
| Bad | 68.18% | 88.24% | 76.92% | 17 |

#### Confusion Matrix -- Test

```
                 Predicted
                 Good    Bad
Actual  Good  [     3      7 ]
        Bad   [     2     15 ]
```

### 3.5 YOLO26n-Pose (Nano)

#### Classification Metrics

| Split | N | Accuracy | Precision | Recall | F1 | Specificity | TP | FP | FN | TN | No-Detection |
|-------|---|----------|-----------|--------|----|-------------|----|----|----|----|----|
| Train | 573 | 71.73% | 93.97% | 59.56% | 72.91% | 93.24% | 218 | 14 | 148 | 193 | 5 (0.9%) |
| Valid | 55 | 60.00% | 89.47% | 45.95% | 60.71% | 88.89% | 17 | 2 | 20 | 16 | 0 (0.0%) |
| Test | 27 | 74.07% | 100.00% | 58.82% | 74.07% | 100.00% | 10 | 0 | 7 | 10 | 0 (0.0%) |

#### Inference Latency (ms)

| Split | Mean | Std | P50 | P95 | P99 | Min | Max | FPS |
|-------|------|-----|-----|-----|-----|-----|-----|-----|
| Train | 64.62 | 32.69 | 62.71 | 66.73 | 69.81 | 60.4 | 842.03 | 15.5 |
| Valid | 63.93 | 1.73 | 63.39 | 67.84 | 69.74 | 62.04 | 70.71 | 15.6 |
| Test | 63.71 | 1.07 | 63.42 | 65.89 | 66.5 | 62.37 | 66.64 | 15.7 |

#### Per-Class Metrics -- Train

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Good | 56.60% | 93.24% | 70.44% | 207 |
| Bad | 93.97% | 59.56% | 72.91% | 366 |

#### Confusion Matrix -- Train

```
                 Predicted
                 Good    Bad
Actual  Good  [   193     14 ]
        Bad   [   148    218 ]
```

#### Per-Class Metrics -- Valid

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Good | 44.44% | 88.89% | 59.26% | 18 |
| Bad | 89.47% | 45.95% | 60.71% | 37 |

#### Confusion Matrix -- Valid

```
                 Predicted
                 Good    Bad
Actual  Good  [    16      2 ]
        Bad   [    20     17 ]
```

#### Per-Class Metrics -- Test

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Good | 58.82% | 100.00% | 74.07% | 10 |
| Bad | 100.00% | 58.82% | 74.07% | 17 |

#### Confusion Matrix -- Test

```
                 Predicted
                 Good    Bad
Actual  Good  [    10      0 ]
        Bad   [     7     10 ]
```

---

## 4. Cross-Model Comparison

### 4.1 Train Split -- All Models

| Model | Accuracy | Precision | Recall | F1 | Specificity | Latency (ms) | FPS |
|-------|----------|-----------|--------|----|-------------|--------------|-----|
| MediaPipe Pose LITE | 78.18% | 84.93% | 80.05% | 82.42% | 74.88% | 26.31 | 38.0 |
| MediaPipe Pose FULL | 78.36% | 89.80% | 74.59% | 81.49% | 85.02% | 30.84 | 32.4 |
| MoveNet Lightning | 67.02% | 66.86% | 95.90% | 78.79% | 15.94% | 7.52 | 133.0 |
| MoveNet Thunder | 66.32% | 70.35% | 81.69% | 75.60% | 39.13% | 33.36 | 30.0 |
| YOLO26n-Pose (Nano) | 71.73% | 93.97% | 59.56% | 72.91% | 93.24% | 64.62 | 15.5 |

### 4.2 Valid Split -- All Models

| Model | Accuracy | Precision | Recall | F1 | Specificity | Latency (ms) | FPS |
|-------|----------|-----------|--------|----|-------------|--------------|-----|
| MediaPipe Pose LITE | 67.27% | 82.76% | 64.86% | 72.73% | 72.22% | 26.82 | 37.3 |
| MediaPipe Pose FULL | 67.27% | 82.76% | 64.86% | 72.73% | 72.22% | 18.31 | 54.6 |
| MoveNet Lightning | 67.27% | 67.92% | 97.30% | 80.00% | 5.56% | 7.46 | 134.0 |
| MoveNet Thunder | 67.27% | 70.21% | 89.19% | 78.57% | 22.22% | 34.16 | 29.3 |
| YOLO26n-Pose (Nano) | 60.00% | 89.47% | 45.95% | 60.71% | 88.89% | 63.93 | 15.6 |

### 4.3 Test Split -- All Models

| Model | Accuracy | Precision | Recall | F1 | Specificity | Latency (ms) | FPS |
|-------|----------|-----------|--------|----|-------------|--------------|-----|
| MediaPipe Pose LITE | 70.37% | 76.47% | 76.47% | 76.47% | 60.00% | 26.68 | 37.5 |
| MediaPipe Pose FULL | 77.78% | 92.31% | 70.59% | 80.00% | 90.00% | 19.38 | 51.6 |
| MoveNet Lightning | 62.96% | 64.00% | 94.12% | 76.19% | 10.00% | 7.43 | 134.6 |
| MoveNet Thunder | 66.67% | 68.18% | 88.24% | 76.92% | 30.00% | 33.29 | 30.0 |
| YOLO26n-Pose (Nano) | 74.07% | 100.00% | 58.82% | 74.07% | 100.00% | 63.71 | 15.7 |

---

## 5. Summary Rankings (Validation Split)

Models ranked by F1-score on the `valid` split:

| Rank | Model | F1 | Accuracy | Recall | Precision | Latency (ms) | FPS |
|------|-------|----|----------|--------|-----------|--------------|-----|
| 1 | **MoveNet Lightning** | **80.00%** | 67.27% | 97.30% | 67.92% | 7.46 | 134.0 |
| 2 | **MoveNet Thunder** | **78.57%** | 67.27% | 89.19% | 70.21% | 34.16 | 29.3 |
| 3 | **MediaPipe Pose LITE** | **72.73%** | 67.27% | 64.86% | 82.76% | 26.82 | 37.3 |
| 4 | **MediaPipe Pose FULL** | **72.73%** | 67.27% | 64.86% | 82.76% | 18.31 | 54.6 |
| 5 | **YOLO26n-Pose (Nano)** | **60.71%** | 60.00% | 45.95% | 89.47% | 63.93 | 15.6 |

> Fastest model: MoveNet Lightning @ 7.46 ms/image
> Best F1 on validation: MoveNet Lightning -- F1=80.00%, Accuracy=67.27%

---

## 6. Methodology & Notes

### Pipeline
```
Image (BGR) -> HPE Inference -> UnifiedKeypoints
           -> PostureMetricsCalculator (CVA, trunk angle, shoulder tilt, ...)
           -> LightweightPostureML.extract_features (8D feature vector)
           -> Hybrid Classifier (rule-based + softmax ML)
           -> Predicted Label: Good | Bad
           -> Compare with GT COCO label -> Classification Metrics
```

### Label Mapping
- PostureStatus.GOOD -> **Good**
- All other statuses (SLOUCHED, FORWARD_HEAD, LEANING_*, TOO_CLOSE, ...) -> **Bad**

### Positive Class
- Positive class = Bad (bad posture detection target)
- Precision: of all predicted Bad, how many truly are Bad
- Recall: of all truly Bad images, how many were correctly flagged

### No-Detection Cases
- When HPE fails to detect any person, the prediction defaults to Bad (safe-fail)

### Classifier Mode
- Hybrid mode: rule-based checks applied first, then lightweight ML softmax
- Uncalibrated (no personal baseline) for fair dataset-wide evaluation

### Latency
- Latency = HPE inference only (excludes image I/O and metric computation)
- Measured on CPU (no CUDA)
