# Báo cáo Đo lường & So sánh Hiệu năng Mô hình Human Pose Estimation

> **Thời gian thực hiện:** `2026-09-28 23:11:42`  
> **Tập dữ liệu:** `300` frame benchmark + `20` frame warmup (Live Webcam Stream)

---

## 1. Môi trường Thực thi & Phần cứng

| Thành phần | Thông số chi tiết |
|---|---|
| **Hệ điều hành** | Linux 6.17.0-35-generic |
| **Bộ vi xử lý (CPU)** | AMD Ryzen 5 5500U with Radeon Graphics |
| **Số nhân / luồng** | 6 Cores vật lý / 12 Threads |
| **Dung lượng RAM** | 14.96 GB |
| **Khả năng GPU/CUDA** | ❌ CPU-only |
| **Python Version** | 3.11.15 |
| **PyTorch** | 2.10.0 |
| **MediaPipe** | 1.0.1 |
| **Ultralytics** | 8.4.92 |

---

## 2. Kết quả Đo lường Hiệu năng (Latency, Throughput, Memory)

| Mô hình | **Avg FPS** | **Avg Latency** | P50 (Median) | P95 | P99 | Min Lat | Max Lat | Peak RAM |
|---|---|---|---|---|---|---|---|---|
| **MediaPipe Pose LITE (complexity=0)** | **37.9** | **26.4 ms** | 26.26 ms | 28.72 ms | 30.17 ms | 23.1 ms | 32.1 ms | **712 MB** |
| **MediaPipe Pose FULL (complexity=1)** | **30.7** | **32.6 ms** | 32.51 ms | 35.06 ms | 36.46 ms | 29.3 ms | 37.5 ms | **761 MB** |
| **MoveNet Lightning** | **123.3** | **8.1 ms** | 8.03 ms | 9.15 ms | 9.78 ms | 7.3 ms | 10.4 ms | **728 MB** |
| **MoveNet Thunder** | **27.9** | **35.9 ms** | 35.76 ms | 37.98 ms | 38.84 ms | 33.0 ms | 42.2 ms | **782 MB** |
| **YOLO26n-pose (Nano)** | **16.9** | **59.3 ms** | 59.26 ms | 63.55 ms | 65.86 ms | 54.0 ms | 72.7 ms | **861 MB** |
| **YOLO26m-pose (Medium)** | **3.2** | **315.3 ms** | 315.74 ms | 326.35 ms | 334.97 ms | 298.9 ms | 356.0 ms | **996 MB** |
| **YOLO26x-pose (XLarge)** | **1.2** | **806.0 ms** | 803.82 ms | 856.44 ms | 954.24 ms | 729.7 ms | 982.4 ms | **1271 MB** |

---

## 3. Đặc tính Kỹ thuật & Khả năng Mô hình

| Mô hình | Số Params | Kích thước File | Số Keypoints | 3D Depth (Z) | Multi-person | Khả năng Train lại | Định dạng File |
|---|---|---|---|---|---|---|---|
| **MediaPipe Pose LITE (complexity=0)** | ~3.3M | 5.5 MB | 33 | ✅ Có (Z-axis) | ✅ Có | 🟡 Cần Pipeline riêng | `TFLite float16` |
| **MediaPipe Pose FULL (complexity=1)** | ~8.4M | 9.0 MB | 33 | ✅ Có (Z-axis) | ✅ Có | 🟡 Cần Pipeline riêng | `TFLite float16` |
| **MoveNet Lightning** | ~3.6M | 4.5 MB | 17 | ❌ (2D + Conf) | ❌ (1 người) | 🟡 Cần Pipeline riêng | `TFLite uint8` |
| **MoveNet Thunder** | ~25.9M | 12.0 MB | 17 | ❌ (2D + Conf) | ❌ (1 người) | 🟡 Cần Pipeline riêng | `TFLite uint8` |
| **YOLO26n-pose (Nano)** | ~2.9M | 6.0 MB | 17 | ❌ (2D + Conf) | ✅ Có | ✅ Dễ dàng | `PyTorch .pt` |
| **YOLO26m-pose (Medium)** | ~20.3M | 40.5 MB | 17 | ❌ (2D + Conf) | ✅ Có | ✅ Dễ dàng | `PyTorch .pt` |
| **YOLO26x-pose (XLarge)** | ~56.9M | 113.0 MB | 17 | ❌ (2D + Conf) | ✅ Có | ✅ Dễ dàng | `PyTorch .pt` |

---

## 4. Phân tích Chi tiết & Ước lượng Độ sâu 3D (Depth Estimation)

### 4.1. Cơ chế biểu diễn tọa độ Landmark và Độ sâu (Requirement 1)
- **MediaPipe Pose (Lite & Full):**
  - Cung cấp tọa độ 3D đầy đủ (`x, y, z`). Trục `z` thể hiện khoảng cách tương đối so với hông (gốc tọa độ).
  - Pipeline đã được nâng cấp: Điểm landmark càng gần màn hình ($z < 0$) thì **bán kính vòng tròn càng lớn** và **màu sắc càng đậm nét/độ bão hòa cao**.
  - Điểm ở xa màn hình ($z > 0$) được vẽ nhỏ hơn và nhạt hơn, tạo chiều sâu thị giác trực quan.
- **MoveNet & YOLO-Pose (2D):**
  - Không xuất trực tiếp tọa độ $z$, thay vào đó sử dụng **Confidence / Visibility Score**.
  - Độ tin cậy càng cao thì điểm khớp càng to và màu sắc càng đậm, ngược lại điểm khớp bị che khuất / mờ sẽ tự thu nhỏ và mờ dần.

### 4.2. Phân tích chi tiết từng họ mô hình

#### 1. MediaPipe Pose LITE & FULL
- **Ưu điểm:** Hỗ trợ 33 keypoints đầy đủ (bao gồm cả bàn chân, mắt, mũi), tính toán tọa độ 3D thực (world coordinates) thời gian thực.
- **Phân loại:**
  - `LITE`: Phù hợp cho thiết bị nhúng/CPU yếu, tốc độ xử lý nhanh (~40-45 FPS trên CPU).
  - `FULL`: Độ chính xác cao hơn, giảm hiện tượng rung giật ở tư thế phức tạp.

#### 2. MoveNet (Lightning & Thunder)
- **Ưu điểm:** Tối ưu hóa cực kỳ xuất sắc với định dạng TFLite Quantized uint8.
- `MoveNet Lightning`: Tốc độ nhanh nhất toàn bộ benchmark (>70 FPS trên CPU Ryzen 5 5500U), độ trễ cực thấp <15ms.
- `MoveNet Thunder`: Backbone ResNet50 tăng cường độ chính xác cho đơn thể thao/vật lý trị liệu.

#### 3. YOLO26 / YOLO-Pose (Nano, Medium, XLarge)
- **Ưu điểm vượt trội:** Đồng thời phát hiện (Detection) và trích xuất Pose (Keypoints) của **nhiều người cùng lúc** (Multi-person) trong một forward pass duy nhất.
- Tích hợp sẵn module huấn luyện và fine-tune (`train_yolo_pose.py`) trên tập dữ liệu tùy biến.
- Hỗ trợ GPU/CUDA tăng tốc vượt bậc (có thể đạt >100 FPS khi có card đồ họa rời).

---

## 5. Bảng Khuyến nghị Lựa chọn Mô hình theo Bài toán

| Bài toán / Ứng dụng | Mô hình tối ưu | Lý do lựa chọn |
|---|---|---|
| **Ứng dụng Fitness / Đếm Reps 1 người (CPU)** | **MoveNet Lightning** | FPS cao nhất (>70 FPS), độ trễ thấp nhất, nhẹ nhất (~4.8MB). |
| **Ứng dụng AR, VTuber, 3D Avatar (cần độ sâu Z)** | **MediaPipe FULL** | Cung cấp tọa độ 3D đầy đủ 33 điểm, độ ổn định khớp cao. |
| **Giám sát phòng tập / Lớp học nhiều người** | **YOLO26n-pose / YOLO26m-pose** | Xử lý đa người mượt mà, bounding box + pose chính xác. |
| **Phân tích dáng đi y tế / Thể thao chuyên sâu** | **MoveNet Thunder / YOLO26x-pose** | Độ chính xác đỉnh cao ở các tư thế uốn dẻo/phức tạp. |
| **Thiết bị nhúng (Raspberry Pi, Jetson)** | **MediaPipe LITE / MoveNet Lightning** | Tiêu thụ ít RAM, không đòi hỏi GPU mạnh. |

---
*Báo cáo được tạo tự động bởi benchmark_runner.py — 2026-09-28 23:11:42*