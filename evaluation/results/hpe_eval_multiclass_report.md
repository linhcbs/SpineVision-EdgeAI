# Báo Cáo Đánh Giá Mô Hình HPE -- Phân Loại Tư Thế Ngồi (Sitting Posture Multiclass)

> **Thời gian chạy:** 2026-09-28 23:15:01
> **Bộ dữ liệu:** Sitting Posture v2 Multiclass (`Sitting Posture.v2i.multiclass`)
> **Nhiệm vụ:** Phân loại tư thế nhị phân (Good Posture vs Bad Posture)
> **Tập dữ liệu đánh giá:** train, valid (eval), test
> **Lớp mục tiêu Positive:** `Bad` (Bad Posture Detection)

---

## 1. Tổng Quan Bộ Dữ Liệu (Dataset Overview)

Bộ dữ liệu phân loại đa lớp Roboflow được quy về bài toán nhị phân chuẩn:
- **Good Posture:** Gốc `goodposture` / file `good_posture-*`.
- **Bad Posture:** Gộp `backwardbadposture` (tư thế ngả sau) và `forwardbadposture` (tư thế chúi trước / gù lưng) thành một lớp **Bad** duy nhất.

### Phân bố nhãn nhị phân (Good vs Bad)

| Tập dữ liệu (Split) | Tổng số ảnh | Good Posture | Bad Posture | Tỷ lệ Bad (%) |
|---------------------|-------------|--------------|-------------|---------------|
| Train | 1314 | 684 | 630 | 47.9% |
| Valid | 187 | 98 | 89 | 47.6% |
| Test | 94 | 43 | 51 | 54.3% |

### Phân bố chi tiết theo loại tư thế gốc

| Tập (Split) | Good Posture | Forward Lean (Bad) | Backward Lean (Bad) | Tổng |
|-------------|--------------|-------------------|--------------------|------|
| Train | 650 | 300 | 364 | 1314 |
| Valid | 95 | 44 | 48 | 187 |
| Test | 41 | 24 | 29 | 94 |

---

## 2. Danh Sách 5 Mô Hình HPE Nhẹ Nhất (Plug-and-Play)

| # | Mô hình | Keypoints | Kích thước trọng số | Backend | Kích thước Input |
|---|---------|-----------|----------------------|---------|-------------------|
| 1 | **MediaPipe Pose LITE** | 33 kps | 5642 KB | TFLite/MediaPipe | dynamic |
| 2 | **MediaPipe Pose FULL** | 33 kps | 9178 KB | TFLite/MediaPipe | dynamic |
| 3 | **MoveNet Lightning** | 17 kps | 4647 KB | TFLite | 192x192 |
| 4 | **MoveNet Thunder** | 17 kps | 12289 KB | TFLite | 256x256 |
| 5 | **YOLO26n-Pose (Nano)** | 17 kps | 6109 KB | PyTorch/Ultralytics | 640x640 |

> *Ghi chú:* Các mô hình nặng hơn như YOLO-m (`yolo26m_pose`) và YOLO-x (`yolo26x_pose`) được loại trừ theo yêu cầu.

---

## 3. Kết Quả Chi Tiết Từng Mô Hình

### 3.1 MediaPipe Pose LITE

#### Chỉ Số Phân Loại (Classification Metrics)

| Split | N | Accuracy | Precision (Bad) | Recall (Bad) | F1-Score | Specificity (Good) | TP | FP | FN | TN | Không nhận diện (No-Det) |
|-------|---|----------|-----------------|--------------|----------|--------------------|----|----|----|----|--------------------------|
| Train | 1314 | 59.28% | 54.82% | 85.71% | 66.87% | 34.94% | 540 | 445 | 90 | 239 | 103 (7.8%) |
| Valid | 187 | 72.19% | 68.69% | 76.40% | 72.34% | 68.37% | 68 | 31 | 21 | 67 | 0 (0.0%) |
| Test | 94 | 71.28% | 71.43% | 78.43% | 74.77% | 62.79% | 40 | 16 | 11 | 27 | 0 (0.0%) |

#### Độ Trễ Suy Luận (Inference Latency & FPS trên CPU)

| Split | Mean (ms) | Std (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Min (ms) | Max (ms) | FPS |
|-------|-----------|----------|----------|----------|----------|----------|----------|-----|
| Train | 14.85 | 3.83 | 13.71 | 26.55 | 29.21 | 11.86 | 31.47 | 67.3 |
| Valid | 13.74 | 1.12 | 13.57 | 15.83 | 17.2 | 11.97 | 17.96 | 72.8 |
| Test | 14.03 | 1.02 | 13.92 | 15.83 | 16.17 | 12.25 | 16.91 | 71.3 |

#### Chi Tiết Theo Lớp -- Tập Valid

| Lớp (Class) | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Good | 76.14% | 68.37% | 72.04% | 98 |
| Bad | 68.69% | 76.40% | 72.34% | 89 |

**Độ nhạy theo từng dạng tư thế thực tế (Valid):**

| Dạng tư thế gốc | Tổng ảnh | Đoán là Bad (Tư thế xấu) | Đoán là Good (Tư thế tốt) | Tỷ lệ nhận diện đúng |
|-----------------|----------|--------------------------|---------------------------|----------------------|
| Forward Lean (Bad) | 44 | 40 | 4 | **90.91%** |
| Backward Lean (Bad) | 48 | 31 | 17 | **64.58%** |
| Good Posture (Good) | 95 | 28 | 67 | **70.53%** |

**Confusion Matrix (Valid):**
```
                       Predicted
                     Good     Bad
Actual   Good  [      67       31   ]
         Bad   [      21       68   ]
```

#### Chi Tiết Theo Lớp -- Tập Test

| Lớp (Class) | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Good | 71.05% | 62.79% | 66.67% | 43 |
| Bad | 71.43% | 78.43% | 74.77% | 51 |

**Độ nhạy theo từng dạng tư thế thực tế (Test):**

| Dạng tư thế gốc | Tổng ảnh | Đoán là Bad (Tư thế xấu) | Đoán là Good (Tư thế tốt) | Tỷ lệ nhận diện đúng |
|-----------------|----------|--------------------------|---------------------------|----------------------|
| Forward Lean (Bad) | 24 | 22 | 2 | **91.67%** |
| Backward Lean (Bad) | 29 | 20 | 9 | **68.97%** |
| Good Posture (Good) | 41 | 14 | 27 | **65.85%** |

**Confusion Matrix (Test):**
```
                       Predicted
                     Good     Bad
Actual   Good  [      27       16   ]
         Bad   [      11       40   ]
```

### 3.2 MediaPipe Pose FULL

#### Chỉ Số Phân Loại (Classification Metrics)

| Split | N | Accuracy | Precision (Bad) | Recall (Bad) | F1-Score | Specificity (Good) | TP | FP | FN | TN | Không nhận diện (No-Det) |
|-------|---|----------|-----------------|--------------|----------|--------------------|----|----|----|----|--------------------------|
| Train | 1314 | 58.68% | 54.27% | 87.78% | 67.07% | 31.87% | 553 | 466 | 77 | 218 | 81 (6.2%) |
| Valid | 187 | 70.05% | 65.14% | 79.78% | 71.72% | 61.22% | 71 | 38 | 18 | 60 | 0 (0.0%) |
| Test | 94 | 71.28% | 69.35% | 84.31% | 76.11% | 55.81% | 43 | 19 | 8 | 24 | 0 (0.0%) |

#### Độ Trễ Suy Luận (Inference Latency & FPS trên CPU)

| Split | Mean (ms) | Std (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Min (ms) | Max (ms) | FPS |
|-------|-----------|----------|----------|----------|----------|----------|----------|-----|
| Train | 21.03 | 3.54 | 20.14 | 32.18 | 35.27 | 17.48 | 37.94 | 47.6 |
| Valid | 20.42 | 1.99 | 20.02 | 23.2 | 29.13 | 17.67 | 30.61 | 49.0 |
| Test | 20.51 | 1.22 | 20.42 | 22.43 | 23.74 | 18.16 | 23.89 | 48.8 |

#### Chi Tiết Theo Lớp -- Tập Valid

| Lớp (Class) | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Good | 76.92% | 61.22% | 68.18% | 98 |
| Bad | 65.14% | 79.78% | 71.72% | 89 |

**Độ nhạy theo từng dạng tư thế thực tế (Valid):**

| Dạng tư thế gốc | Tổng ảnh | Đoán là Bad (Tư thế xấu) | Đoán là Good (Tư thế tốt) | Tỷ lệ nhận diện đúng |
|-----------------|----------|--------------------------|---------------------------|----------------------|
| Forward Lean (Bad) | 44 | 41 | 3 | **93.18%** |
| Backward Lean (Bad) | 48 | 33 | 15 | **68.75%** |
| Good Posture (Good) | 95 | 35 | 60 | **63.16%** |

**Confusion Matrix (Valid):**
```
                       Predicted
                     Good     Bad
Actual   Good  [      60       38   ]
         Bad   [      18       71   ]
```

#### Chi Tiết Theo Lớp -- Tập Test

| Lớp (Class) | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Good | 75.00% | 55.81% | 64.00% | 43 |
| Bad | 69.35% | 84.31% | 76.11% | 51 |

**Độ nhạy theo từng dạng tư thế thực tế (Test):**

| Dạng tư thế gốc | Tổng ảnh | Đoán là Bad (Tư thế xấu) | Đoán là Good (Tư thế tốt) | Tỷ lệ nhận diện đúng |
|-----------------|----------|--------------------------|---------------------------|----------------------|
| Forward Lean (Bad) | 24 | 23 | 1 | **95.83%** |
| Backward Lean (Bad) | 29 | 22 | 7 | **75.86%** |
| Good Posture (Good) | 41 | 17 | 24 | **58.54%** |

**Confusion Matrix (Test):**
```
                       Predicted
                     Good     Bad
Actual   Good  [      24       19   ]
         Bad   [       8       43   ]
```

### 3.3 MoveNet Lightning

#### Chỉ Số Phân Loại (Classification Metrics)

| Split | N | Accuracy | Precision (Bad) | Recall (Bad) | F1-Score | Specificity (Good) | TP | FP | FN | TN | Không nhận diện (No-Det) |
|-------|---|----------|-----------------|--------------|----------|--------------------|----|----|----|----|--------------------------|
| Train | 1314 | 79.00% | 79.40% | 75.87% | 77.60% | 81.87% | 478 | 124 | 152 | 560 | 0 (0.0%) |
| Valid | 187 | 86.63% | 95.71% | 75.28% | 84.28% | 96.94% | 67 | 3 | 22 | 95 | 0 (0.0%) |
| Test | 94 | 86.17% | 93.18% | 80.39% | 86.32% | 93.02% | 41 | 3 | 10 | 40 | 0 (0.0%) |

#### Độ Trễ Suy Luận (Inference Latency & FPS trên CPU)

| Split | Mean (ms) | Std (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Min (ms) | Max (ms) | FPS |
|-------|-----------|----------|----------|----------|----------|----------|----------|-----|
| Train | 8.39 | 0.79 | 8.18 | 9.96 | 10.7 | 7.41 | 11.88 | 119.2 |
| Valid | 8.55 | 0.92 | 8.36 | 10.14 | 10.91 | 7.41 | 13.36 | 117.0 |
| Test | 8.5 | 0.91 | 8.25 | 10.35 | 11.04 | 7.44 | 11.35 | 117.6 |

#### Chi Tiết Theo Lớp -- Tập Valid

| Lớp (Class) | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Good | 81.20% | 96.94% | 88.37% | 98 |
| Bad | 95.71% | 75.28% | 84.28% | 89 |

**Độ nhạy theo từng dạng tư thế thực tế (Valid):**

| Dạng tư thế gốc | Tổng ảnh | Đoán là Bad (Tư thế xấu) | Đoán là Good (Tư thế tốt) | Tỷ lệ nhận diện đúng |
|-----------------|----------|--------------------------|---------------------------|----------------------|
| Forward Lean (Bad) | 44 | 34 | 10 | **77.27%** |
| Backward Lean (Bad) | 48 | 36 | 12 | **75.00%** |
| Good Posture (Good) | 95 | 0 | 95 | **100.00%** |

**Confusion Matrix (Valid):**
```
                       Predicted
                     Good     Bad
Actual   Good  [      95        3   ]
         Bad   [      22       67   ]
```

#### Chi Tiết Theo Lớp -- Tập Test

| Lớp (Class) | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Good | 80.00% | 93.02% | 86.02% | 43 |
| Bad | 93.18% | 80.39% | 86.32% | 51 |

**Độ nhạy theo từng dạng tư thế thực tế (Test):**

| Dạng tư thế gốc | Tổng ảnh | Đoán là Bad (Tư thế xấu) | Đoán là Good (Tư thế tốt) | Tỷ lệ nhận diện đúng |
|-----------------|----------|--------------------------|---------------------------|----------------------|
| Forward Lean (Bad) | 24 | 20 | 4 | **83.33%** |
| Backward Lean (Bad) | 29 | 23 | 6 | **79.31%** |
| Good Posture (Good) | 41 | 1 | 40 | **97.56%** |

**Confusion Matrix (Test):**
```
                       Predicted
                     Good     Bad
Actual   Good  [      40        3   ]
         Bad   [      10       41   ]
```

### 3.4 MoveNet Thunder

#### Chỉ Số Phân Loại (Classification Metrics)

| Split | N | Accuracy | Precision (Bad) | Recall (Bad) | F1-Score | Specificity (Good) | TP | FP | FN | TN | Không nhận diện (No-Det) |
|-------|---|----------|-----------------|--------------|----------|--------------------|----|----|----|----|--------------------------|
| Train | 1314 | 80.21% | 81.68% | 75.71% | 78.58% | 84.36% | 477 | 107 | 153 | 577 | 0 (0.0%) |
| Valid | 187 | 89.30% | 96.00% | 80.90% | 87.80% | 96.94% | 72 | 3 | 17 | 95 | 0 (0.0%) |
| Test | 94 | 91.49% | 95.74% | 88.24% | 91.84% | 95.35% | 45 | 2 | 6 | 41 | 0 (0.0%) |

#### Độ Trễ Suy Luận (Inference Latency & FPS trên CPU)

| Split | Mean (ms) | Std (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Min (ms) | Max (ms) | FPS |
|-------|-----------|----------|----------|----------|----------|----------|----------|-----|
| Train | 36.46 | 1.53 | 36.27 | 38.88 | 41.54 | 33.4 | 54.08 | 27.4 |
| Valid | 36.72 | 1.33 | 36.62 | 39.36 | 40.2 | 33.79 | 40.93 | 27.2 |
| Test | 36.7 | 1.39 | 36.59 | 39.0 | 40.6 | 33.89 | 41.38 | 27.2 |

#### Chi Tiết Theo Lớp -- Tập Valid

| Lớp (Class) | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Good | 84.82% | 96.94% | 90.48% | 98 |
| Bad | 96.00% | 80.90% | 87.80% | 89 |

**Độ nhạy theo từng dạng tư thế thực tế (Valid):**

| Dạng tư thế gốc | Tổng ảnh | Đoán là Bad (Tư thế xấu) | Đoán là Good (Tư thế tốt) | Tỷ lệ nhận diện đúng |
|-----------------|----------|--------------------------|---------------------------|----------------------|
| Forward Lean (Bad) | 44 | 31 | 13 | **70.45%** |
| Backward Lean (Bad) | 48 | 44 | 4 | **91.67%** |
| Good Posture (Good) | 95 | 0 | 95 | **100.00%** |

**Confusion Matrix (Valid):**
```
                       Predicted
                     Good     Bad
Actual   Good  [      95        3   ]
         Bad   [      17       72   ]
```

#### Chi Tiết Theo Lớp -- Tập Test

| Lớp (Class) | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Good | 87.23% | 95.35% | 91.11% | 43 |
| Bad | 95.74% | 88.24% | 91.84% | 51 |

**Độ nhạy theo từng dạng tư thế thực tế (Test):**

| Dạng tư thế gốc | Tổng ảnh | Đoán là Bad (Tư thế xấu) | Đoán là Good (Tư thế tốt) | Tỷ lệ nhận diện đúng |
|-----------------|----------|--------------------------|---------------------------|----------------------|
| Forward Lean (Bad) | 24 | 21 | 3 | **87.50%** |
| Backward Lean (Bad) | 29 | 26 | 3 | **89.66%** |
| Good Posture (Good) | 41 | 0 | 41 | **100.00%** |

**Confusion Matrix (Test):**
```
                       Predicted
                     Good     Bad
Actual   Good  [      41        2   ]
         Bad   [       6       45   ]
```

### 3.5 YOLO26n-Pose (Nano)

#### Chỉ Số Phân Loại (Classification Metrics)

| Split | N | Accuracy | Precision (Bad) | Recall (Bad) | F1-Score | Specificity (Good) | TP | FP | FN | TN | Không nhận diện (No-Det) |
|-------|---|----------|-----------------|--------------|----------|--------------------|----|----|----|----|--------------------------|
| Train | 1314 | 82.50% | 77.25% | 90.00% | 83.14% | 75.58% | 567 | 167 | 63 | 517 | 2 (0.2%) |
| Valid | 187 | 95.19% | 94.44% | 95.51% | 94.97% | 94.90% | 85 | 5 | 4 | 93 | 0 (0.0%) |
| Test | 94 | 91.49% | 90.57% | 94.12% | 92.31% | 88.37% | 48 | 5 | 3 | 38 | 0 (0.0%) |

#### Độ Trễ Suy Luận (Inference Latency & FPS trên CPU)

| Split | Mean (ms) | Std (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Min (ms) | Max (ms) | FPS |
|-------|-----------|----------|----------|----------|----------|----------|----------|-----|
| Train | 77.53 | 24.63 | 76.06 | 84.21 | 90.43 | 67.08 | 932.69 | 12.9 |
| Valid | 81.34 | 4.78 | 80.42 | 88.93 | 99.0 | 74.16 | 102.76 | 12.3 |
| Test | 84.1 | 13.25 | 81.81 | 89.52 | 112.02 | 73.68 | 203.06 | 11.9 |

#### Chi Tiết Theo Lớp -- Tập Valid

| Lớp (Class) | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Good | 95.88% | 94.90% | 95.38% | 98 |
| Bad | 94.44% | 95.51% | 94.97% | 89 |

**Độ nhạy theo từng dạng tư thế thực tế (Valid):**

| Dạng tư thế gốc | Tổng ảnh | Đoán là Bad (Tư thế xấu) | Đoán là Good (Tư thế tốt) | Tỷ lệ nhận diện đúng |
|-----------------|----------|--------------------------|---------------------------|----------------------|
| Forward Lean (Bad) | 44 | 42 | 2 | **95.45%** |
| Backward Lean (Bad) | 48 | 46 | 2 | **95.83%** |
| Good Posture (Good) | 95 | 2 | 93 | **97.89%** |

**Confusion Matrix (Valid):**
```
                       Predicted
                     Good     Bad
Actual   Good  [      93        5   ]
         Bad   [       4       85   ]
```

#### Chi Tiết Theo Lớp -- Tập Test

| Lớp (Class) | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Good | 92.68% | 88.37% | 90.48% | 43 |
| Bad | 90.57% | 94.12% | 92.31% | 51 |

**Độ nhạy theo từng dạng tư thế thực tế (Test):**

| Dạng tư thế gốc | Tổng ảnh | Đoán là Bad (Tư thế xấu) | Đoán là Good (Tư thế tốt) | Tỷ lệ nhận diện đúng |
|-----------------|----------|--------------------------|---------------------------|----------------------|
| Forward Lean (Bad) | 24 | 24 | 0 | **100.00%** |
| Backward Lean (Bad) | 29 | 26 | 3 | **89.66%** |
| Good Posture (Good) | 41 | 3 | 38 | **92.68%** |

**Confusion Matrix (Test):**
```
                       Predicted
                     Good     Bad
Actual   Good  [      38        5   ]
         Bad   [       3       48   ]
```

---

## 4. So Sánh Tổng Thể Các Mô Hình (Cross-Model Comparison)

### 4.1 Tập Train -- Toàn Bộ 5 Mô Hình

| Mô hình | Accuracy | Precision (Bad) | Recall (Bad) | F1-Score | Specificity | Mean Latency (ms) | FPS (CPU) |
|---------|----------|-----------------|--------------|----------|-------------|-------------------|-----------|
| MediaPipe Pose LITE | 59.28% | 54.82% | 85.71% | 66.87% | 34.94% | 14.85 | 67.3 |
| MediaPipe Pose FULL | 58.68% | 54.27% | 87.78% | 67.07% | 31.87% | 21.03 | 47.6 |
| MoveNet Lightning | 79.00% | 79.40% | 75.87% | 77.60% | 81.87% | 8.39 | 119.2 |
| MoveNet Thunder | 80.21% | 81.68% | 75.71% | 78.58% | 84.36% | 36.46 | 27.4 |
| YOLO26n-Pose (Nano) | 82.50% | 77.25% | 90.00% | 83.14% | 75.58% | 77.53 | 12.9 |

### 4.2 Tập Valid -- Toàn Bộ 5 Mô Hình

| Mô hình | Accuracy | Precision (Bad) | Recall (Bad) | F1-Score | Specificity | Mean Latency (ms) | FPS (CPU) |
|---------|----------|-----------------|--------------|----------|-------------|-------------------|-----------|
| MediaPipe Pose LITE | 72.19% | 68.69% | 76.40% | 72.34% | 68.37% | 13.74 | 72.8 |
| MediaPipe Pose FULL | 70.05% | 65.14% | 79.78% | 71.72% | 61.22% | 20.42 | 49.0 |
| MoveNet Lightning | 86.63% | 95.71% | 75.28% | 84.28% | 96.94% | 8.55 | 117.0 |
| MoveNet Thunder | 89.30% | 96.00% | 80.90% | 87.80% | 96.94% | 36.72 | 27.2 |
| YOLO26n-Pose (Nano) | 95.19% | 94.44% | 95.51% | 94.97% | 94.90% | 81.34 | 12.3 |

### 4.3 Tập Test -- Toàn Bộ 5 Mô Hình

| Mô hình | Accuracy | Precision (Bad) | Recall (Bad) | F1-Score | Specificity | Mean Latency (ms) | FPS (CPU) |
|---------|----------|-----------------|--------------|----------|-------------|-------------------|-----------|
| MediaPipe Pose LITE | 71.28% | 71.43% | 78.43% | 74.77% | 62.79% | 14.03 | 71.3 |
| MediaPipe Pose FULL | 71.28% | 69.35% | 84.31% | 76.11% | 55.81% | 20.51 | 48.8 |
| MoveNet Lightning | 86.17% | 93.18% | 80.39% | 86.32% | 93.02% | 8.5 | 117.6 |
| MoveNet Thunder | 91.49% | 95.74% | 88.24% | 91.84% | 95.35% | 36.7 | 27.2 |
| YOLO26n-Pose (Nano) | 91.49% | 90.57% | 94.12% | 92.31% | 88.37% | 84.1 | 11.9 |

---

## 5. Bảng Xếp Hạng Đánh Giá (Summary Rankings)

### 5.1 Xếp hạng trên tập Valid (theo F1-Score)

| Hạng | Mô hình | F1-Score | Accuracy | Recall (Bad) | Precision (Bad) | Độ trễ (ms) | FPS (CPU) |
|------|---------|----------|----------|--------------|-----------------|-------------|-----------|
| 1 | **YOLO26n-Pose (Nano)** | **94.97%** | 95.19% | 95.51% | 94.44% | 81.34 | 12.3 |
| 2 | **MoveNet Thunder** | **87.80%** | 89.30% | 80.90% | 96.00% | 36.72 | 27.2 |
| 3 | **MoveNet Lightning** | **84.28%** | 86.63% | 75.28% | 95.71% | 8.55 | 117.0 |
| 4 | **MediaPipe Pose LITE** | **72.34%** | 72.19% | 76.40% | 68.69% | 13.74 | 72.8 |
| 5 | **MediaPipe Pose FULL** | **71.72%** | 70.05% | 79.78% | 65.14% | 20.42 | 49.0 |

> 🏆 **Mô hình tốt nhất trên tập Valid:** **YOLO26n-Pose (Nano)** (F1 = 94.97%, Accuracy = 95.19%)
> ⚡ **Mô hình nhanh nhất:** **MoveNet Lightning** (8.55 ms/ảnh ~ 117.0 FPS)

### 5.2 Xếp hạng trên tập Test (theo F1-Score)

| Hạng | Mô hình | F1-Score | Accuracy | Recall (Bad) | Precision (Bad) | Độ trễ (ms) | FPS (CPU) |
|------|---------|----------|----------|--------------|-----------------|-------------|-----------|
| 1 | **YOLO26n-Pose (Nano)** | **92.31%** | 91.49% | 94.12% | 90.57% | 84.1 | 11.9 |
| 2 | **MoveNet Thunder** | **91.84%** | 91.49% | 88.24% | 95.74% | 36.7 | 27.2 |
| 3 | **MoveNet Lightning** | **86.32%** | 86.17% | 80.39% | 93.18% | 8.5 | 117.6 |
| 4 | **MediaPipe Pose FULL** | **76.11%** | 71.28% | 84.31% | 69.35% | 20.51 | 48.8 |
| 5 | **MediaPipe Pose LITE** | **74.77%** | 71.28% | 78.43% | 71.43% | 14.03 | 71.3 |

> 🏆 **Mô hình tốt nhất trên tập Test:** **YOLO26n-Pose (Nano)** (F1 = 92.31%, Accuracy = 91.49%)
> ⚡ **Mô hình nhanh nhất:** **MoveNet Lightning** (8.5 ms/ảnh ~ 117.6 FPS)

---

## 6. Phương Pháp Đánh Giá & Nguyên Lý Kỹ Thuật (Methodology)

### Pipeline Xử Lý Hoàn Chỉnh
```
Ảnh đầu vào (BGR) -> Mô hình HPE (MediaPipe / MoveNet / YOLO)
                 -> Trích xuất UnifiedKeypoints chuẩn hóa
                 -> PostureMetricsCalculator (Góc nghiêng CVA, góc thân, độ lệch vai)
                 -> PostureClassifier (Chế độ Hybrid: Rule-based + ML)
                 -> Nhãn dự đoán: Good | Bad
                 -> Đối chiếu với Ground Truth -> Tính toán ma trận lỗi & chỉ số
```

### Quy tắc gộp lớp theo yêu cầu:
- Nhãn **Good Posture** (`Good`): Các trường hợp tư thế ngồi đúng, thẳng lưng, công thái học tốt.
- Nhãn **Bad Posture** (`Bad`): Gộp tất cả các dạng sai lệch tư thế bao gồm cả chúi đầu/nghiêng trước (`forwardbadposture`) và ngửa sau (`backwardbadposture`).
- Safe-fail khi không phát hiện người (`no_detection`): Mặc định gán cảnh báo `Bad` để bảo đảm an toàn công thái học.
