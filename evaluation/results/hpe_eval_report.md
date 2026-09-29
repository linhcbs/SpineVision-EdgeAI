# HPE Model Evaluation Report -- Posture Classification

> **Generated:** 2026-09-28 23:20:02
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
| Train | 573 | 74.17% | 73.29% | 93.72% | 82.25% | 39.61% | 343 | 125 | 23 | 82 | 2 (0.3%) |
| Valid | 55 | 76.36% | 76.09% | 94.59% | 84.34% | 38.89% | 35 | 11 | 2 | 7 | 0 (0.0%) |
| Test | 27 | 70.37% | 69.57% | 94.12% | 80.00% | 30.00% | 16 | 7 | 1 | 3 | 0 (0.0%) |

#### Inference Latency (ms)

| Split | Mean | Std | P50 | P95 | P99 | Min | Max | FPS |
|-------|------|-----|-----|-----|-----|-----|-----|-----|
| Train | 13.87 | 1.66 | 13.53 | 16.23 | 19.21 | 12.03 | 33.04 | 72.1 |
| Valid | 13.8 | 1.07 | 13.52 | 15.57 | 16.81 | 12.37 | 18.02 | 72.5 |
| Test | 13.53 | 0.68 | 13.48 | 14.53 | 15.49 | 12.67 | 15.8 | 73.9 |

#### Per-Class Metrics -- Train

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Good | 78.10% | 39.61% | 52.56% | 207 |
| Bad | 73.29% | 93.72% | 82.25% | 366 |

#### Confusion Matrix -- Train

```
                 Predicted
                 Good    Bad
Actual  Good  [    82    125 ]
        Bad   [    23    343 ]
```

#### Per-Class Metrics -- Valid

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Good | 77.78% | 38.89% | 51.85% | 18 |
| Bad | 76.09% | 94.59% | 84.34% | 37 |

#### Confusion Matrix -- Valid

```
                 Predicted
                 Good    Bad
Actual  Good  [     7     11 ]
        Bad   [     2     35 ]
```

#### Per-Class Metrics -- Test

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Good | 75.00% | 30.00% | 42.86% | 10 |
| Bad | 69.57% | 94.12% | 80.00% | 17 |

#### Confusion Matrix -- Test

```
                 Predicted
                 Good    Bad
Actual  Good  [     3      7 ]
        Bad   [     1     16 ]
```

### 3.2 MediaPipe Pose FULL

#### Classification Metrics

| Split | N | Accuracy | Precision | Recall | F1 | Specificity | TP | FP | FN | TN | No-Detection |
|-------|---|----------|-----------|--------|----|-------------|----|----|----|----|----|
| Train | 573 | 79.23% | 77.63% | 94.81% | 85.36% | 51.69% | 347 | 100 | 19 | 107 | 2 (0.3%) |
| Valid | 55 | 80.00% | 80.95% | 91.89% | 86.08% | 55.56% | 34 | 8 | 3 | 10 | 0 (0.0%) |
| Test | 27 | 77.78% | 76.19% | 94.12% | 84.21% | 50.00% | 16 | 5 | 1 | 5 | 0 (0.0%) |

#### Inference Latency (ms)

| Split | Mean | Std | P50 | P95 | P99 | Min | Max | FPS |
|-------|------|-----|-----|-----|-----|-----|-----|-----|
| Train | 19.78 | 1.91 | 19.38 | 22.74 | 25.29 | 17.56 | 37.97 | 50.6 |
| Valid | 20.38 | 1.43 | 19.97 | 23.19 | 24.19 | 18.26 | 24.63 | 49.1 |
| Test | 18.42 | 0.87 | 18.11 | 20.25 | 20.59 | 17.51 | 20.68 | 54.3 |

#### Per-Class Metrics -- Train

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Good | 84.92% | 51.69% | 64.26% | 207 |
| Bad | 77.63% | 94.81% | 85.36% | 366 |

#### Confusion Matrix -- Train

```
                 Predicted
                 Good    Bad
Actual  Good  [   107    100 ]
        Bad   [    19    347 ]
```

#### Per-Class Metrics -- Valid

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Good | 76.92% | 55.56% | 64.52% | 18 |
| Bad | 80.95% | 91.89% | 86.08% | 37 |

#### Confusion Matrix -- Valid

```
                 Predicted
                 Good    Bad
Actual  Good  [    10      8 ]
        Bad   [     3     34 ]
```

#### Per-Class Metrics -- Test

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Good | 83.33% | 50.00% | 62.50% | 10 |
| Bad | 76.19% | 94.12% | 84.21% | 17 |

#### Confusion Matrix -- Test

```
                 Predicted
                 Good    Bad
Actual  Good  [     5      5 ]
        Bad   [     1     16 ]
```

### 3.3 MoveNet Lightning

#### Classification Metrics

| Split | N | Accuracy | Precision | Recall | F1 | Specificity | TP | FP | FN | TN | No-Detection |
|-------|---|----------|-----------|--------|----|-------------|----|----|----|----|----|
| Train | 573 | 78.88% | 97.67% | 68.58% | 80.58% | 97.10% | 251 | 6 | 115 | 201 | 0 (0.0%) |
| Valid | 55 | 70.91% | 100.00% | 56.76% | 72.41% | 100.00% | 21 | 0 | 16 | 18 | 0 (0.0%) |
| Test | 27 | 66.67% | 100.00% | 47.06% | 64.00% | 100.00% | 8 | 0 | 9 | 10 | 0 (0.0%) |

#### Inference Latency (ms)

| Split | Mean | Std | P50 | P95 | P99 | Min | Max | FPS |
|-------|------|-----|-----|-----|-----|-----|-----|-----|
| Train | 7.95 | 0.67 | 7.66 | 9.43 | 10.14 | 7.39 | 11.0 | 125.8 |
| Valid | 8.2 | 0.58 | 8.13 | 9.25 | 9.74 | 7.48 | 9.77 | 122.0 |
| Test | 7.92 | 0.4 | 7.87 | 8.67 | 8.97 | 7.39 | 9.04 | 126.3 |

#### Per-Class Metrics -- Train

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Good | 63.61% | 97.10% | 76.86% | 207 |
| Bad | 97.67% | 68.58% | 80.58% | 366 |

#### Confusion Matrix -- Train

```
                 Predicted
                 Good    Bad
Actual  Good  [   201      6 ]
        Bad   [   115    251 ]
```

#### Per-Class Metrics -- Valid

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Good | 52.94% | 100.00% | 69.23% | 18 |
| Bad | 100.00% | 56.76% | 72.41% | 37 |

#### Confusion Matrix -- Valid

```
                 Predicted
                 Good    Bad
Actual  Good  [    18      0 ]
        Bad   [    16     21 ]
```

#### Per-Class Metrics -- Test

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Good | 52.63% | 100.00% | 68.97% | 10 |
| Bad | 100.00% | 47.06% | 64.00% | 17 |

#### Confusion Matrix -- Test

```
                 Predicted
                 Good    Bad
Actual  Good  [    10      0 ]
        Bad   [     9      8 ]
```

### 3.4 MoveNet Thunder

#### Classification Metrics

| Split | N | Accuracy | Precision | Recall | F1 | Specificity | TP | FP | FN | TN | No-Detection |
|-------|---|----------|-----------|--------|----|-------------|----|----|----|----|----|
| Train | 573 | 85.17% | 99.65% | 77.05% | 86.90% | 99.52% | 282 | 1 | 84 | 206 | 0 (0.0%) |
| Valid | 55 | 74.55% | 100.00% | 62.16% | 76.67% | 100.00% | 23 | 0 | 14 | 18 | 0 (0.0%) |
| Test | 27 | 74.07% | 100.00% | 58.82% | 74.07% | 100.00% | 10 | 0 | 7 | 10 | 0 (0.0%) |

#### Inference Latency (ms)

| Split | Mean | Std | P50 | P95 | P99 | Min | Max | FPS |
|-------|------|-----|-----|-----|-----|-----|-----|-----|
| Train | 36.11 | 2.45 | 35.76 | 40.39 | 43.84 | 32.65 | 49.86 | 27.7 |
| Valid | 35.91 | 1.59 | 35.79 | 39.0 | 40.5 | 33.84 | 41.98 | 27.8 |
| Test | 35.87 | 0.99 | 35.99 | 37.18 | 38.13 | 33.9 | 38.46 | 27.9 |

#### Per-Class Metrics -- Train

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Good | 71.03% | 99.52% | 82.90% | 207 |
| Bad | 99.65% | 77.05% | 86.90% | 366 |

#### Confusion Matrix -- Train

```
                 Predicted
                 Good    Bad
Actual  Good  [   206      1 ]
        Bad   [    84    282 ]
```

#### Per-Class Metrics -- Valid

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Good | 56.25% | 100.00% | 72.00% | 18 |
| Bad | 100.00% | 62.16% | 76.67% | 37 |

#### Confusion Matrix -- Valid

```
                 Predicted
                 Good    Bad
Actual  Good  [    18      0 ]
        Bad   [    14     23 ]
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

### 3.5 YOLO26n-Pose (Nano)

#### Classification Metrics

| Split | N | Accuracy | Precision | Recall | F1 | Specificity | TP | FP | FN | TN | No-Detection |
|-------|---|----------|-----------|--------|----|-------------|----|----|----|----|----|
| Train | 573 | 87.09% | 94.24% | 84.97% | 89.37% | 90.82% | 311 | 19 | 55 | 188 | 5 (0.9%) |
| Valid | 55 | 80.00% | 96.43% | 72.97% | 83.08% | 94.44% | 27 | 1 | 10 | 17 | 0 (0.0%) |
| Test | 27 | 88.89% | 100.00% | 82.35% | 90.32% | 100.00% | 14 | 0 | 3 | 10 | 0 (0.0%) |

#### Inference Latency (ms)

| Split | Mean | Std | P50 | P95 | P99 | Min | Max | FPS |
|-------|------|-----|-----|-----|-----|-----|-----|-----|
| Train | 79.52 | 35.55 | 77.31 | 84.34 | 108.18 | 68.56 | 908.27 | 12.6 |
| Valid | 81.05 | 3.64 | 81.03 | 87.39 | 89.46 | 73.88 | 91.42 | 12.3 |
| Test | 82.43 | 5.27 | 80.82 | 93.22 | 94.13 | 76.27 | 94.41 | 12.1 |

#### Per-Class Metrics -- Train

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Good | 77.37% | 90.82% | 83.56% | 207 |
| Bad | 94.24% | 84.97% | 89.37% | 366 |

#### Confusion Matrix -- Train

```
                 Predicted
                 Good    Bad
Actual  Good  [   188     19 ]
        Bad   [    55    311 ]
```

#### Per-Class Metrics -- Valid

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Good | 62.96% | 94.44% | 75.56% | 18 |
| Bad | 96.43% | 72.97% | 83.08% | 37 |

#### Confusion Matrix -- Valid

```
                 Predicted
                 Good    Bad
Actual  Good  [    17      1 ]
        Bad   [    10     27 ]
```

#### Per-Class Metrics -- Test

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Good | 76.92% | 100.00% | 86.96% | 10 |
| Bad | 100.00% | 82.35% | 90.32% | 17 |

#### Confusion Matrix -- Test

```
                 Predicted
                 Good    Bad
Actual  Good  [    10      0 ]
        Bad   [     3     14 ]
```

---

## 4. Cross-Model Comparison

### 4.1 Train Split -- All Models

| Model | Accuracy | Precision | Recall | F1 | Specificity | Latency (ms) | FPS |
|-------|----------|-----------|--------|----|-------------|--------------|-----|
| MediaPipe Pose LITE | 74.17% | 73.29% | 93.72% | 82.25% | 39.61% | 13.87 | 72.1 |
| MediaPipe Pose FULL | 79.23% | 77.63% | 94.81% | 85.36% | 51.69% | 19.78 | 50.6 |
| MoveNet Lightning | 78.88% | 97.67% | 68.58% | 80.58% | 97.10% | 7.95 | 125.8 |
| MoveNet Thunder | 85.17% | 99.65% | 77.05% | 86.90% | 99.52% | 36.11 | 27.7 |
| YOLO26n-Pose (Nano) | 87.09% | 94.24% | 84.97% | 89.37% | 90.82% | 79.52 | 12.6 |

### 4.2 Valid Split -- All Models

| Model | Accuracy | Precision | Recall | F1 | Specificity | Latency (ms) | FPS |
|-------|----------|-----------|--------|----|-------------|--------------|-----|
| MediaPipe Pose LITE | 76.36% | 76.09% | 94.59% | 84.34% | 38.89% | 13.8 | 72.5 |
| MediaPipe Pose FULL | 80.00% | 80.95% | 91.89% | 86.08% | 55.56% | 20.38 | 49.1 |
| MoveNet Lightning | 70.91% | 100.00% | 56.76% | 72.41% | 100.00% | 8.2 | 122.0 |
| MoveNet Thunder | 74.55% | 100.00% | 62.16% | 76.67% | 100.00% | 35.91 | 27.8 |
| YOLO26n-Pose (Nano) | 80.00% | 96.43% | 72.97% | 83.08% | 94.44% | 81.05 | 12.3 |

### 4.3 Test Split -- All Models

| Model | Accuracy | Precision | Recall | F1 | Specificity | Latency (ms) | FPS |
|-------|----------|-----------|--------|----|-------------|--------------|-----|
| MediaPipe Pose LITE | 70.37% | 69.57% | 94.12% | 80.00% | 30.00% | 13.53 | 73.9 |
| MediaPipe Pose FULL | 77.78% | 76.19% | 94.12% | 84.21% | 50.00% | 18.42 | 54.3 |
| MoveNet Lightning | 66.67% | 100.00% | 47.06% | 64.00% | 100.00% | 7.92 | 126.3 |
| MoveNet Thunder | 74.07% | 100.00% | 58.82% | 74.07% | 100.00% | 35.87 | 27.9 |
| YOLO26n-Pose (Nano) | 88.89% | 100.00% | 82.35% | 90.32% | 100.00% | 82.43 | 12.1 |

---

## 5. Summary Rankings (Validation Split)

Models ranked by F1-score on the `valid` split:

| Rank | Model | F1 | Accuracy | Recall | Precision | Latency (ms) | FPS |
|------|-------|----|----------|--------|-----------|--------------|-----|
| 1 | **MediaPipe Pose FULL** | **86.08%** | 80.00% | 91.89% | 80.95% | 20.38 | 49.1 |
| 2 | **MediaPipe Pose LITE** | **84.34%** | 76.36% | 94.59% | 76.09% | 13.8 | 72.5 |
| 3 | **YOLO26n-Pose (Nano)** | **83.08%** | 80.00% | 72.97% | 96.43% | 81.05 | 12.3 |
| 4 | **MoveNet Thunder** | **76.67%** | 74.55% | 62.16% | 100.00% | 35.91 | 27.8 |
| 5 | **MoveNet Lightning** | **72.41%** | 70.91% | 56.76% | 100.00% | 8.2 | 122.0 |

> Fastest model: MoveNet Lightning @ 8.2 ms/image
> Best F1 on validation: MediaPipe Pose FULL -- F1=86.08%, Accuracy=80.00%

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
