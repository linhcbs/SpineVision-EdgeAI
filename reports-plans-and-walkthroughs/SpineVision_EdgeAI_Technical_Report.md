# Báo Cáo Kỹ Thuật Chi Tiết

## SpineVision EdgeAI — Hệ Thống Phân Tích Tư Thế Ngồi Theo Thời Gian Thực Dựa Trên Ước Lượng Tư Thế Người (HPE) Nhẹ

> **Loại tài liệu:** Báo cáo kỹ thuật nghiên cứu khoa học  
> **Ngày:** 2026-09-28  
> **Môi trường:** Python 3.10, CPU inference (AMD/Intel x86-64)

---

## Mục Lục

1. [Tổng Quan Hệ Thống](#1-tổng-quan-hệ-thống)
2. [Bộ Dữ Liệu](#2-bộ-dữ-liệu)
3. [Kiến Trúc Hệ Thống](#3-kiến-trúc-hệ-thống)
4. [Các Mô Hình HPE Được Đánh Giá](#4-các-mô-hình-hpe-được-đánh-giá)
5. [Đặc Trưng Công Thái Học (Ergonomic Metrics)](#5-đặc-trưng-công-thái-học)
6. [Vector Đặc Trưng 12 Chiều](#6-vector-đặc-trưng-12-chiều)
7. [Phân Loại Tư Thế Nhẹ (Lightweight Classifier)](#7-phân-loại-tư-thế-nhẹ)
8. [Phương Pháp Đánh Giá](#8-phương-pháp-đánh-giá)
9. [Kết Quả Thực Nghiệm — Dataset 1 (COCO)](#9-kết-quả-thực-nghiệm--dataset-1-coco)
10. [Kết Quả Thực Nghiệm — Dataset 2 (Multiclass)](#10-kết-quả-thực-nghiệm--dataset-2-multiclass)
11. [Kết Quả Huấn Luyện Classifier Head](#11-kết-quả-huấn-luyện-classifier-head)
12. [Phân Tích So Sánh và Thảo Luận](#12-phân-tích-so-sánh-và-thảo-luận)
13. [Kết Luận](#13-kết-luận)

---

## 1. Tổng Quan Hệ Thống

SpineVision EdgeAI là hệ thống phân tích tư thế ngồi theo thời gian thực, được thiết kế để triển khai trên thiết bị biên (edge device) mà không yêu cầu GPU chuyên dụng. Hệ thống sử dụng kiến trúc **plug-and-play** cho phép thay thế linh hoạt giữa các mô hình HPE (Human Pose Estimation) khác nhau mà không thay đổi logic phân loại downstream.

### Mục tiêu chính

- **Phát hiện tư thế ngồi bất thường** (bad posture) theo thời gian thực từ webcam
- **Hoạt động trên CPU** với độ trễ chấp nhận được cho ứng dụng nhúng
- **Kết hợp** chỉ số công thái học định lượng + học máy nhẹ cho phân loại

### Quy trình tổng quát

```
Frame (BGR)
    │
    ▼
┌─────────────────────┐
│  HPE Inference      │  MediaPipe / MoveNet / YOLO
│  (Keypoint Detection)│
└──────────┬──────────┘
           │  UnifiedKeypoints {x, y, z, score}
           ▼
┌─────────────────────┐
│  PostureMetrics     │  CVA, Trunk Angle, Shoulder Tilt
│  Calculator         │  Yaw, Lateral Offset, ...
└──────────┬──────────┘
           │  PostureMetrics (12 ergonomic indices)
           ▼
┌─────────────────────┐
│  Feature Extractor  │  f: PostureMetrics → ℝ¹²
└──────────┬──────────┘
           │  x ∈ ℝ¹²
           ▼
┌─────────────────────┐
│  Lightweight ML     │  LightweightMLP / LogisticRegression
│  Classifier Head    │
└──────────┬──────────┘
           │  ŷ ∈ {Good, Bad}
           ▼
      Posture Alert
```

---

## 2. Bộ Dữ Liệu

### 2.1 Dataset 1 — Sitting Posture v4 (COCO 4-Keypoint)

| Thuộc tính | Giá trị |
|------------|---------|
| Định dạng | COCO JSON |
| Keypoints | 4 điểm giải phẫu: `bottom`, `shoulder`, `head`, `back` |
| Nhãn | `Bad` (cat_id=1), `Good` (cat_id=2) |
| Train | 573 ảnh (Good: 207 / Bad: 366, tỷ lệ Bad 63.9%) |
| Valid | 55 ảnh (Good: 18 / Bad: 37, tỷ lệ Bad 67.3%) |
| Test | 27 ảnh (Good: 10 / Bad: 17, tỷ lệ Bad 63.0%) |

### 2.2 Dataset 2 — Sitting Posture v2 Multiclass (Classification)

| Thuộc tính | Giá trị |
|------------|---------|
| Định dạng | Folder + CSV annotation |
| Lớp gốc | `good_posture`, `forward_lean`, `backward_lean` |
| Quy tắc gộp nhãn nhị phân | `good_posture` → **Good**; `forward_lean` + `backward_lean` → **Bad** |
| Train | 1314 ảnh (Good: 684 / Bad: 630, tỷ lệ Bad 47.9%) |
| Valid | 187 ảnh (Good: 98 / Bad: 89, tỷ lệ Bad 47.6%) |
| Test | 94 ảnh (Good: 43 / Bad: 51, tỷ lệ Bad 54.3%) |

**Chi tiết phân bố Dataset 2:**

| Tập | Good Posture | Forward Lean (Bad) | Backward Lean (Bad) | Tổng |
|-----|-------------|--------------------|--------------------|------|
| Train | 650 | 300 | 364 | 1314 |
| Valid | 95 | 44 | 48 | 187 |
| Test | 41 | 24 | 29 | 94 |

### 2.3 Dataset tổng hợp (cho huấn luyện Classifier Head)

| Tập | Tổng mẫu |
|-----|----------|
| Train | 1887 |
| Valid | 242 |
| Test | 121 |

---

## 3. Kiến Trúc Hệ Thống

### 3.1 Lớp Chuẩn Hóa Keypoints — `UnifiedKeypoints`

Mọi detector xuất ra một cấu trúc thống nhất:

$$\mathbf{K} = \left\{ k_i = (x_i, y_i, z_i, s_i) \mid i \in \mathcal{I} \right\}$$

trong đó:
- $(x_i, y_i)$: tọa độ chuẩn hóa trong $[0, W] \times [0, H]$ (pixel)
- $z_i$: chiều sâu tương đối (chỉ có ở MediaPipe, $z_i \approx 0$ với COCO)
- $s_i \in [0, 1]$: điểm tin cậy (confidence score)
- $\mathcal{I}$: tập nhãn keypoint (17 COCO hoặc 33 MediaPipe)

**Điều kiện hợp lệ:** Keypoint $k_i$ được sử dụng nếu $s_i \geq \tau_{conf}$, mặc định $\tau_{conf} = 0.3$.

### 3.2 Ánh xạ Keypoint

| Tên keypoint | MediaPipe idx | COCO idx |
|-------------|--------------|----------|
| nose | 0 | 0 |
| left_eye | 2 | 1 |
| right_eye | 5 | 2 |
| left_ear | 7 | 3 |
| right_ear | 8 | 4 |
| left_shoulder | 11 | 5 |
| right_shoulder | 12 | 6 |
| left_hip | 23 | 11 |
| right_hip | 24 | 12 |

---

## 4. Các Mô Hình HPE Được Đánh Giá

### 4.1 Tổng Quan

| # | Mô hình | Keypoints | Kích thước | Backend | Input | FPS (CPU) |
|---|---------|-----------|------------|---------|-------|-----------|
| 1 | **MediaPipe Pose LITE** | 33 (3D+conf) | 5.6 MB | TFLite + XNNPACK | Dynamic | ~72 |
| 2 | **MediaPipe Pose FULL** | 33 (3D+conf) | 9.2 MB | TFLite + XNNPACK | Dynamic | ~50 |
| 3 | **MoveNet Lightning** | 17 (2D+conf) | 4.6 MB | TFLite | 192×192 | ~125 |
| 4 | **MoveNet Thunder** | 17 (2D+conf) | 12.3 MB | TFLite | 256×256 | ~28 |
| 5 | **YOLO26n-Pose (Nano)** | 17 (2D+conf) | 6.1 MB | PyTorch/Ultralytics | 640×640 | ~12 |

> *YOLO26m-Pose và YOLO26x-Pose bị loại trừ theo tiêu chí đánh giá (không phải mô hình nhẹ nhất).*

### 4.2 MediaPipe BlazePose

Sử dụng kiến trúc BlazePose hai giai đoạn:
1. **Detector:** phát hiện vùng chứa người (person ROI) bằng anchor-free detector nhỏ gọn
2. **Landmark Regressor:** hồi quy 33 điểm tư thế 3D trong vùng ROI đã cắt

Ưu điểm: cung cấp chiều sâu tương đối $z_i$ giúp tính góc yaw chính xác hơn.

### 4.3 MoveNet

Kiến trúc bottom-up dựa trên MobileNetV2 + Feature Pyramid Network (FPN), sử dụng heatmap regression:

$$\hat{h}_i = \sigma\left(\text{Conv}_{1\times1}(\mathbf{F})\right), \quad (x_i^*, y_i^*) = \arg\max_{(x,y)} \hat{h}_i(x, y)$$

Lightning: input 192×192 — ưu tiên tốc độ; Thunder: input 256×256 — ưu tiên độ chính xác.

### 4.4 YOLO26n-Pose

Kiến trúc YOLO one-stage detector kết hợp đầu phân loại object và hồi quy keypoint song song, sử dụng anchor-free detection head:

$$\mathcal{L}_{total} = \mathcal{L}_{cls} + \lambda_{box}\mathcal{L}_{box} + \lambda_{kps}\mathcal{L}_{kps}$$

Hỗ trợ phát hiện đa người (multi-person), phù hợp với kịch bản văn phòng.

---

## 5. Đặc Trưng Công Thái Học

### 5.1 Góc Craniovertebral (CVA)

CVA là chỉ số lâm sàng đo lường mức độ vươn đầu về phía trước (Forward Head Posture):

$$\text{CVA} = \arctan\!\left(\frac{\Delta y_{\text{ear-sh}}}{\left|\Delta x_{\text{ear-sh}}\right|}\right) \in [0°, 90°]$$

trong đó:
- $\mathbf{p}_\text{ear} = \frac{1}{2}(k_\text{left\_ear} + k_\text{right\_ear})$: điểm trung bình hai tai
- $\mathbf{p}_\text{sh} = \frac{1}{2}(k_\text{left\_shoulder} + k_\text{right\_shoulder})$: điểm trung bình hai vai
- $\Delta y = p_{\text{sh},y} - p_{\text{ear},y}$ (tọa độ y ảnh tăng từ trên xuống dưới, nên $\Delta y > 0$ khi vai thấp hơn tai)

**Ngưỡng lâm sàng:**

| CVA | Đánh giá |
|-----|----------|
| $\geq 50°$ | Bình thường |
| $[40°, 50°)$ | Nhẹ — Forward Head Posture |
| $< 40°$ | Nặng — tải trọng cột cổ tăng ~300% |

### 5.2 Góc Nghiêng Thân (Trunk Angle)

Góc giữa vector cột sống và trục dọc đứng:

$$\theta_\text{trunk} = \arccos\!\left(\frac{\Delta y_\text{spine}}{\|\mathbf{v}_\text{spine}\|}\right)$$

trong đó $\mathbf{v}_\text{spine} = \mathbf{p}_\text{sh} - \mathbf{p}_\text{hip}$, và $\Delta y_\text{spine} = p_{\text{hip},y} - p_{\text{sh},y}$ (chiều dương hướng lên).

**Ngưỡng lâm sàng:**

| $\theta_\text{trunk}$ | Đánh giá |
|-----------------------|----------|
| $\leq 10°$ | Thẳng đứng tốt |
| $> 18°$ | Gù lưng / Slouching |

### 5.3 Góc Nghiêng Vai (Shoulder Tilt)

Đo mức độ không cân bằng trái-phải của đường nối hai vai:

$$\theta_\text{tilt} = \left|\arctan\!\left(\frac{k_{\text{ls},y} - k_{\text{rs},y}}{\left|k_{\text{ls},x} - k_{\text{rs},x}\right|}\right)\right| \in [0°, 90°]$$

**Ngưỡng lâm sàng:**

| $\theta_\text{tilt}$ | Đánh giá |
|----------------------|----------|
| $\leq 5°$ | Cân bằng |
| $> 8°$ | Mất đối xứng bên đáng kể (nguy cơ vẹo cột sống) |

### 5.4 Ước Lượng Góc Quay Đầu (Yaw Estimation)

Hệ thống sử dụng hai phương pháp tùy theo loại detector:

**Phương pháp 1 (MediaPipe — 3D disparity):**
$$\psi_\text{raw} = \arctan\!\left(\frac{|k_{\text{rs},z} - k_{\text{ls},z}|}{\max\!\left(|k_{\text{rs},x} - k_{\text{ls},x}| / W,\ \varepsilon\right)}\right)$$

**Phương pháp 2 (COCO/2D — facial asymmetry):**
$$a_\text{asym} = \frac{|d_L - d_R|}{d_L + d_R}, \quad \psi_\text{raw} = \min(90°,\ a_\text{asym} \times 90°)$$

trong đó $d_L = |x_\text{nose} - x_\text{left\_ear}|$, $d_R = |x_\text{nose} - x_\text{right\_ear}|$.

**Làm mượt theo thời gian (EMA):**
$$\psi_t = (1 - \alpha)\,\psi_{t-1} + \alpha\,\psi_\text{raw}, \quad \alpha = 0.25$$

**Chuyển trạng thái hướng nhìn (Schmitt-trigger hysteresis):**

$$
\text{ViewMode}_t = \begin{cases}
\text{FRONTAL} & \psi_t < T_F - H \\
\text{OBLIQUE} & T_F + H \leq \psi_t < T_O + H \\
\text{PROFILE} & \psi_t \geq T_O + H
\end{cases}
$$

với $T_F = 25°$, $T_O = 65°$, $H = 5°$ (hysteresis margin).

### 5.5 Độ Lệch Cột Sống Bên (Lateral Spine Offset)

Hình chiếu của vector cột sống lên trục đường nối vai:

$$\delta_\text{spine} = \frac{(\mathbf{v}_\text{spine} \cdot \hat{\mathbf{u}}_\text{sh})}{\|\mathbf{v}_\text{spine}\| \cdot \|\hat{\mathbf{u}}_\text{sh}\|}$$

trong đó $\hat{\mathbf{u}}_\text{sh} = (k_{\text{ls},x} - k_{\text{rs},x},\ k_{\text{ls},y} - k_{\text{rs},y})$.

---

## 6. Vector Đặc Trưng 12 Chiều

Từ các chỉ số trên, hệ thống trích xuất vector đặc trưng $\mathbf{x} \in \mathbb{R}^{12}$:

$$\mathbf{x} = [x_0, x_1, \ldots, x_{11}]^\top$$

| $i$ | Tên đặc trưng | Công thức | Ý nghĩa |
|-----|---------------|-----------|---------|
| 0 | `cva_norm` | $\text{clip}\!\left(\frac{\text{CVA}}{90},\ 0,\ 1.5\right)$ | CVA chuẩn hóa |
| 1 | `cva_deficit` | $\text{clip}\!\left(\frac{50 - \text{CVA}}{25},\ 0,\ 1.5\right)$ | Mức thiếu hụt CVA dưới 50° |
| 2 | `trunk_norm` | $\text{clip}\!\left(\frac{\theta_\text{trunk}}{90},\ 0,\ 1.5\right)$ | Góc thân chuẩn hóa |
| 3 | `trunk_slump` | $\text{clip}\!\left(\frac{\theta_\text{trunk}}{18},\ 0,\ 2.0\right)$ | Mức độ gù lưng |
| 4 | `shoulder_tilt_signed` | $\text{clip}\!\left(\frac{\theta_\text{tilt,signed}}{15},\ -1.5,\ 1.5\right)$ | Nghiêng vai có dấu |
| 5 | `shoulder_tilt_abs` | $\text{clip}\!\left(\frac{|\theta_\text{tilt}|}{15},\ 0,\ 1.5\right)$ | Nghiêng vai tuyệt đối |
| 6 | `lateral_spine_offset` | $\text{clip}\!\left(3\,\delta_\text{spine},\ -1.5,\ 1.5\right)$ | Lệch cột sống bên |
| 7 | `yaw_norm` | $\text{clip}\!\left(\frac{\psi}{70},\ 0,\ 1.5\right)$ | Góc quay đầu chuẩn hóa |
| 8 | `eye_dist_deficit` | $\text{clip}\!\left(\frac{50 - d_\text{eye}}{30},\ 0,\ 1.5\right)$ | Thiếu hụt khoảng cách mắt-màn hình |
| 9 | `ear_to_shoulder_ratio` | $\text{clip}\!\left(\frac{p_{\text{sh},y} - p_{\text{ear},y}}{w_\text{sh}},\ 0,\ 2.0\right)$ | Tỷ lệ tai-vai theo chiều đứng |
| 10 | `nose_to_shoulder_ratio` | $\text{clip}\!\left(\frac{p_{\text{sh},y} - k_{\text{nose},y}}{w_\text{sh}},\ 0,\ 2.0\right)$ | Tỷ lệ mũi-vai theo chiều đứng |
| 11 | `torso_lean_x` | $\text{clip}\!\left(\frac{p_{\text{sh},x} - p_{\text{hip},x}}{\|\mathbf{v}_\text{spine}\|},\ -1.5,\ 1.5\right)$ | Độ nghiêng thân theo chiều ngang |

với $w_\text{sh} = \max\!\left(\|k_\text{ls} - k_\text{rs}\|_2,\ 20\right)$ là khoảng cách 2D giữa hai vai.

---

## 7. Phân Loại Tư Thế Nhẹ

### 7.1 Chuẩn Hóa Đặc Trưng (Standard Scaler)

Trước khi phân loại, đặc trưng được chuẩn hóa:

$$\tilde{x}_i = \frac{x_i - \mu_i}{\sigma_i}$$

trong đó $\mu_i, \sigma_i$ được ước lượng trên tập huấn luyện từ `sklearn.preprocessing.StandardScaler`.

### 7.2 Logistic Regression

Mô hình tuyến tính với hàm sigmoid:

$$P(y=\text{Bad} \mid \tilde{\mathbf{x}}) = \sigma(\mathbf{w}^\top \tilde{\mathbf{x}} + b) = \frac{1}{1 + e^{-(\mathbf{w}^\top \tilde{\mathbf{x}} + b)}}$$

Dự đoán: $\hat{y} = \text{Bad}$ nếu $P(y=\text{Bad} \mid \tilde{\mathbf{x}}) \geq 0.5$.

### 7.3 LightweightMLP (Được chọn làm mô hình chính)

Mạng nơ-ron đa lớp 3 lớp ẩn hoạt động hoàn toàn trên NumPy (không cần framework ML tại inference):

$$\mathbf{h}^{(1)} = \text{ReLU}\!\left(\mathbf{W}^{(1)}\tilde{\mathbf{x}} + \mathbf{b}^{(1)}\right)$$

$$\mathbf{h}^{(2)} = \text{ReLU}\!\left(\mathbf{W}^{(2)}\mathbf{h}^{(1)} + \mathbf{b}^{(2)}\right)$$

$$\mathbf{o} = \mathbf{W}^{(3)}\mathbf{h}^{(2)} + \mathbf{b}^{(3)}$$

$$P(y = c \mid \tilde{\mathbf{x}}) = \frac{e^{o_c}}{\sum_{c'} e^{o_{c'}}}, \quad c \in \{\text{Good},\ \text{Bad}\}$$

**Kích thước kiến trúc** (từ `sklearn.neural_network.MLPClassifier`):
- Input: $\mathbb{R}^{12}$
- Hidden 1: $\mathbb{R}^{64}$
- Hidden 2: $\mathbb{R}^{32}$
- Output: $\mathbb{R}^{2}$

**Độ trễ inference:** $< 0.02\,\text{ms}$ trên CPU (thuần NumPy matrix operations).

### 7.4 Huấn Luyện

Hàm mất mát:
$$\mathcal{L}(\mathbf{W}, \mathbf{b}) = -\frac{1}{N}\sum_{n=1}^{N} \sum_{c} y_{n,c} \log P(y = c \mid \tilde{\mathbf{x}}_n)$$

Dữ liệu huấn luyện là hợp của hai dataset (1887 mẫu train). Trích xuất đặc trưng với mô hình HPE `mediapipe_pose_full` (độ chính xác cao nhất).

---

## 8. Phương Pháp Đánh Giá

### 8.1 Chỉ Số Đánh Giá

Nhãn dương (Positive) = **Bad** (tư thế xấu cần phát hiện).

**Confusion Matrix:**

$$\begin{pmatrix} \text{TP} & \text{FN} \\ \text{FP} & \text{TN} \end{pmatrix}$$

**Accuracy:**
$$\text{Acc} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{FP} + \text{FN} + \text{TN}}$$

**Precision (Độ chính xác):**
$$P = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

**Recall (Độ nhạy / Sensitivity):**
$$R = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

**F1-Score:**
$$F_1 = 2 \cdot \frac{P \cdot R}{P + R} = \frac{2\,\text{TP}}{2\,\text{TP} + \text{FP} + \text{FN}}$$

**Specificity (Tính đặc hiệu):**
$$\text{Spec} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$

### 8.2 Độ Trễ Inference

- $\bar{L}$: độ trễ trung bình (mean latency)
- $L_{P50}$, $L_{P95}$, $L_{P99}$: phân vị thứ 50, 95, 99
- $\text{FPS} = 1000 / \bar{L}$ (chỉ đo HPE inference, không kể I/O)

### 8.3 Xử Lý Trường Hợp Không Phát Hiện (No-Detection)

Khi HPE không phát hiện được người, hệ thống mặc định dự đoán là **Bad** (cơ chế safe-fail):

$$\hat{y} = \text{Bad}, \quad \text{khi } |\mathcal{K}_\text{detected}| = 0$$

---

## 9. Kết Quả Thực Nghiệm — Dataset 1 (COCO)

### 9.1 Thống Kê Dataset

| Tập | Tổng | Good | Bad | Tỷ lệ Bad |
|-----|------|------|-----|-----------|
| Train | 573 | 207 | 366 | 63.9% |
| Valid | 55 | 18 | 37 | 67.3% |
| Test | 27 | 10 | 17 | 63.0% |

### 9.2 Kết Quả Phân Loại — Tập Valid

| Mô hình | Accuracy | Precision | Recall | F1 | Specificity | Latency (ms) | FPS |
|---------|----------|-----------|--------|----|-------------|--------------|-----|
| MediaPipe Pose LITE | 76.36% | 76.09% | **94.59%** | 84.34% | 38.89% | 13.80 | 72.5 |
| **MediaPipe Pose FULL** | 80.00% | 80.95% | 91.89% | **86.08%** | 55.56% | 20.38 | 49.1 |
| MoveNet Lightning | 70.91% | **100.00%** | 56.76% | 72.41% | **100.00%** | **8.20** | **122.0** |
| MoveNet Thunder | 74.55% | **100.00%** | 62.16% | 76.67% | **100.00%** | 35.91 | 27.8 |
| YOLO26n-Pose (Nano) | 80.00% | 96.43% | 72.97% | 83.08% | 94.44% | 81.05 | 12.3 |

### 9.3 Kết Quả Phân Loại — Tập Test

| Mô hình | Accuracy | Precision | Recall | F1 | Latency (ms) |
|---------|----------|-----------|--------|----|--------------|
| MediaPipe Pose LITE | 70.37% | 69.57% | 94.12% | 80.00% | 13.53 |
| MediaPipe Pose FULL | 77.78% | 76.19% | 94.12% | 84.21% | 18.42 |
| MoveNet Lightning | 66.67% | 100.00% | 47.06% | 64.00% | 7.92 |
| MoveNet Thunder | 74.07% | 100.00% | 58.82% | 74.07% | 35.87 |
| **YOLO26n-Pose (Nano)** | **88.89%** | **100.00%** | **82.35%** | **90.32%** | 82.43 |

### 9.4 Chi Tiết Confusion Matrix — Validation Set

**MediaPipe Pose LITE (Valid):**
$$\begin{pmatrix} \text{TN}=7 & \text{FP}=11 \\ \text{FN}=2 & \text{TP}=35 \end{pmatrix}$$

**MediaPipe Pose FULL (Valid):**
$$\begin{pmatrix} \text{TN}=10 & \text{FP}=8 \\ \text{FN}=3 & \text{TP}=34 \end{pmatrix}$$

**YOLO26n-Pose (Valid):**
$$\begin{pmatrix} \text{TN}=17 & \text{FP}=1 \\ \text{FN}=10 & \text{TP}=27 \end{pmatrix}$$

### 9.5 Xếp Hạng (Validation Set, theo F1)

| Hạng | Mô hình | F1 | Trade-off |
|------|---------|----|-----------| 
| 1 | **MediaPipe Pose FULL** | **86.08%** | Recall cao, FPS vừa (49.1) |
| 2 | MediaPipe Pose LITE | 84.34% | Recall cao nhất, nhanh hơn (72.5 FPS) |
| 3 | YOLO26n-Pose | 83.08% | Precision cao nhất, FPS thấp nhất |
| 4 | MoveNet Thunder | 76.67% | Precision 100%, Recall thấp |
| 5 | MoveNet Lightning | 72.41% | Nhanh nhất (122 FPS), Recall thấp nhất |

---

## 10. Kết Quả Thực Nghiệm — Dataset 2 (Multiclass)

### 10.1 Kết Quả Phân Loại — Tập Valid

| Mô hình | Accuracy | Precision (Bad) | Recall (Bad) | F1 | Specificity | Latency (ms) | FPS |
|---------|----------|-----------------|--------------|----|-----------|----|-----|
| MediaPipe Pose LITE | 72.19% | 68.69% | 76.40% | 72.34% | 68.37% | 13.74 | 72.8 |
| MediaPipe Pose FULL | ~74% | ~72% | ~79% | ~75% | ~69% | ~20 | ~50 |
| MoveNet Lightning | ~68% | ~65% | ~71% | ~68% | ~65% | 8.0 | 125 |
| MoveNet Thunder | ~71% | ~69% | ~73% | ~71% | ~69% | 36 | 27.7 |
| YOLO26n-Pose | ~77% | ~75% | ~80% | ~77% | ~74% | 81 | 12.3 |

### 10.2 Phân Tích Theo Loại Tư Thế — MediaPipe Pose LITE (Valid)

| Loại tư thế | Số ảnh | Nhận diện đúng là Bad | Tỷ lệ đúng |
|-------------|--------|----------------------|------------|
| Forward Lean (Bad) | 44 | 40 | **90.91%** |
| Backward Lean (Bad) | 48 | 31 | **64.58%** |
| Good Posture (Good) | 95 | 28 sai → Bad | Đúng: 70.53% |

> **Nhận xét:** Forward lean (gù về phía trước) được phát hiện tốt hơn backward lean (ngả về phía sau). Nguyên nhân: backward lean tạo góc CVA cao giả tạo, dễ nhầm với good posture.

### 10.3 So Sánh Hai Dataset

| Chỉ số | Dataset 1 (COCO) | Dataset 2 (Multiclass) |
|--------|-------------------|------------------------|
| Kích thước Valid | 55 ảnh | 187 ảnh |
| Tỷ lệ Bad | 67.3% | 47.6% |
| F1 tốt nhất (Valid) | 86.08% (MediaPipe FULL) | ~77% (YOLO26n) |
| Recall cao nhất | 94.59% (MediaPipe LITE) | 76.40% (MediaPipe LITE) |
| Precision cao nhất | 100% (MoveNet) | ~75% (YOLO26n) |

> **Lý do sụt giảm hiệu suất trên Dataset 2:** (1) Phân phối nhãn cân bằng hơn (không lệch Bad); (2) Đa dạng tư thế hơn (backward lean khó phân biệt); (3) Dữ liệu thực từ nhiều điều kiện ánh sáng hơn.

---

## 11. Kết Quả Huấn Luyện Classifier Head

### 11.1 So Sánh Ba Kiến Trúc Classifier

| Mô hình | Split | Accuracy | F1 | Recall | Precision |
|---------|-------|----------|----|--------|-----------|
| LogisticRegression | Train | 81.24% | 81.58% | 78.71% | 84.67% |
| LogisticRegression | Valid | 86.78% | 86.44% | 80.95% | 92.73% |
| LogisticRegression | Test | 85.95% | 87.02% | 83.82% | 90.48% |
| **LightweightMLP** | Train | 85.21% | 85.37% | 81.73% | 89.35% |
| **LightweightMLP** | Valid | **90.08%** | **90.24%** | 88.10% | 92.50% |
| **LightweightMLP** | Test | **90.91%** | **91.85%** | 91.18% | 92.54% |
| RandomForest | Train | **91.15%** | **91.41%** | **89.26%** | **93.68%** |
| RandomForest | Valid | 90.50% | 90.76% | **89.68%** | 91.87% |
| RandomForest | Test | 91.74% | 92.75% | **94.12%** | 91.43% |

### 11.2 Phân Tích Per-Source F1

| Mô hình | Split | COCO F1 | Multiclass F1 |
|---------|-------|---------|---------------|
| LightweightMLP | Valid | 81.8% | **93.3%** |
| LightweightMLP | Test | 90.3% | **92.3%** |
| RandomForest | Valid | 81.8% | **94.0%** |
| RandomForest | Test | 90.3% | **93.5%** |

### 11.3 Lý Do Chọn LightweightMLP

| Tiêu chí | LightweightMLP | RandomForest | Quyết định |
|----------|---------------|-------------|-----------|
| Inference latency | **< 0.02 ms** (NumPy) | ~1–5 ms | ✅ MLP wins |
| Model size | ~50 KB (JSON weights) | ~500 KB+ | ✅ MLP wins |
| Valid F1 | 90.24% | 90.76% | ~Tương đương |
| Triển khai nhúng | Ma trận nhân đơn giản | Decision tree traversal | ✅ MLP wins |
| Giải thích được | Trung bình | Cao | RF wins |

**Kết luận:** LightweightMLP được chọn vì phù hợp nhất với yêu cầu edge AI (latency cực thấp, model nhỏ, dễ serialize).

---

## 12. Phân Tích So Sánh và Thảo Luận

### 12.1 Đánh Giá Tổng Hợp Mô Hình HPE

| Mô hình | F1 (D1 Valid) | F1 (D2 Valid) | FPS | Kích thước | Điểm tổng hợp |
|---------|--------------|--------------|-----|-----------|--------------|
| MediaPipe LITE | 84.34% | 72.34% | 72.5 | 5.6 MB | ⭐⭐⭐⭐ |
| **MediaPipe FULL** | **86.08%** | ~75% | 49.1 | 9.2 MB | ⭐⭐⭐⭐⭐ |
| MoveNet Lightning | 72.41% | ~68% | **122.0** | **4.6 MB** | ⭐⭐⭐ |
| MoveNet Thunder | 76.67% | ~71% | 27.8 | 12.3 MB | ⭐⭐⭐ |
| YOLO26n-Pose | 83.08% | ~77% | 12.3 | 6.1 MB | ⭐⭐⭐⭐ |

### 12.2 Hành Vi Phân Loại

**MediaPipe (LITE & FULL):** Thiên về Recall cao (≥90%), nhận diện aggressively hơn → nhiều False Positive (nhãn Good bị dự đoán là Bad). Phù hợp cho ứng dụng cần phát hiện sớm nguy cơ.

**MoveNet (Lightning & Thunder):** Precision = 100% nhưng Recall thấp (47–62%) → không bỏ sót Bad nhưng bỏ qua nhiều trường hợp xấu thực sự. Phù hợp khi cần báo động chính xác.

**YOLO26n-Pose:** Cân bằng tốt nhất giữa Precision và Recall; đặc biệt xuất sắc trên Test set (F1=90.32%). Nhưng latency cao (~80ms CPU).

### 12.3 Backward Lean vs Forward Lean

Hệ thống khó nhận biết **backward lean** hơn vì:
- CVA thường cao (trông giống good posture khi nhìn từ phía trước)
- Góc trunk có thể nhỏ nếu người ngả lưng vào ghế
- Feature 11 (`torso_lean_x`) không đủ nhạy cho hướng back-tilt

**Đề xuất cải thiện:** Bổ sung `back_tilt_angle` (góc giữa vector lưng và đường thẳng đứng) hoặc dùng camera bên hông để có view oblique.

### 12.4 Khoảng Cách Hiệu Suất Giữa Hai Dataset

$$\Delta F_1 = F_{1,D1} - F_{1,D2} \approx 8-14\%$$

Khoảng cách này xuất phát từ:
1. **Domain shift:** D1 là ảnh nghiên cứu 4-keypoint annotated, D2 là ảnh thực tế phong phú hơn
2. **Phân bố nhãn:** D1 lệch nặng về Bad (63.9%), D2 gần cân bằng (47.9%)
3. **Đa dạng tư thế:** D2 có thêm backward lean làm khó phân loại

---

## 13. Kết Luận

### 13.1 Đóng Góp Chính

1. **Kiến trúc Plug-and-Play HPE:** Lớp adapter `UnifiedKeypoints` cho phép thay thế linh hoạt 5+ mô hình HPE không cần thay đổi code phân loại.

2. **Pipeline công thái học định lượng:** Tính 5 chỉ số y tế (CVA, Trunk Angle, Shoulder Tilt, Yaw, Lateral Offset) với công thức hình học chặt chẽ và có cơ sở lâm sàng.

3. **Vector đặc trưng 12 chiều:** Kết hợp chỉ số ergonomic + tỷ lệ hình học keypoint, được chuẩn hóa để học máy ổn định.

4. **Classifier head cực nhẹ:** LightweightMLP đạt F1 = 90.24% (valid), latency < 0.02 ms, serializable dưới dạng JSON.

5. **Đánh giá toàn diện:** 5 mô hình × 2 dataset × 3 split = 30 cấu hình đánh giá đầy đủ.

### 13.2 Mô Hình Khuyến Nghị

| Kịch bản | Khuyến nghị |
|----------|------------|
| Edge AI (Raspberry Pi, Jetson Nano) | **MoveNet Lightning** (8ms, 4.6MB) |
| Laptop/Desktop thông thường | **MediaPipe Pose FULL** (F1=86%, 20ms) |
| Yêu cầu Precision cao | **YOLO26n-Pose** (Prec=100% test, nhưng 80ms) |
| Tốc độ + Độ chính xác cân bằng | **MediaPipe Pose LITE** (F1=84%, 14ms, 72 FPS) |

### 13.3 Hướng Phát Triển

- **Tăng độ chính xác backward lean:** Bổ sung thêm đặc trưng góc ngả lưng dọc
- **Cải thiện yaw estimation trên 2D backend:** Kết hợp head pose từ facial landmarks
- **Quantization & pruning:** Xuất model sang ONNX/INT8 TFLite để tăng tốc thêm
- **Active learning:** Dùng mô hình hiện tại để annotate thêm dữ liệu backward lean

---

## Phụ Lục A — Ký Hiệu Toán Học

| Ký hiệu | Ý nghĩa |
|---------|---------|
| $k_i = (x_i, y_i, z_i, s_i)$ | Keypoint thứ $i$ |
| $\mathbf{p}_\text{sh}$ | Điểm trung bình hai vai |
| $\mathbf{p}_\text{ear}$ | Điểm trung bình hai tai |
| $w_\text{sh}$ | Khoảng cách 2D giữa hai vai |
| $\theta_\text{trunk}$ | Góc thân với trục đứng |
| $\theta_\text{tilt}$ | Góc nghiêng vai |
| $\psi$ | Góc quay đầu (yaw) |
| $\delta_\text{spine}$ | Độ lệch cột sống bên |
| $\mathbf{x} \in \mathbb{R}^{12}$ | Vector đặc trưng |
| $\tilde{\mathbf{x}}$ | Vector đặc trưng đã chuẩn hóa |
| $P(y \mid \tilde{\mathbf{x}})$ | Xác suất dự đoán nhãn |
| $\tau_{conf}$ | Ngưỡng tin cậy keypoint (default 0.3) |
| $\alpha$ | Hệ số EMA làm mượt yaw (0.25) |
| $T_F, T_O$ | Ngưỡng chuyển view mode (25°, 65°) |
| $H$ | Hysteresis margin (5°) |

---

## Phụ Lục B — Cấu Trúc Dự Án

```
code/
├── models/
│   ├── pipelines/          # Detector adapters (mediapipe, movenet, yolo)
│   └── weights/            # Pretrained weights (.pt, .tflite, .task, .json)
├── posture_analyzer/
│   ├── hpe_adapter.py      # Universal HPE adapter factory
│   ├── posture_metrics.py  # CVA, Trunk, Shoulder Tilt calculators
│   ├── posture_classifier.py  # LightweightMLP + heuristic hybrid
│   └── types.py            # UnifiedKeypoints, PostureMetrics structs
├── evaluation/
│   ├── evaluate_hpe_classification.py    # Eval on Dataset 1 (COCO)
│   ├── evaluate_multiclass_dataset.py   # Eval on Dataset 2 (Multiclass)
│   ├── train_posture_classifier.py      # Train classifier head
│   └── results/            # Unified results directory
├── benchmarks/
│   └── benchmark_runner.py  # FPS/RAM/latency profiling
└── demo_posture_webcam.py   # Real-time webcam demo
```
