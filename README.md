# SpineVision-EdgeAI: Human Pose Estimation Pipeline & Benchmark

Dự án triển khai, trực quan hóa và đo lường hiệu năng chuyên sâu các mô hình **Human Pose Estimation (HPE)** hàng đầu trên môi trường Edge AI / CPU / GPU.

---

## 🚀 Tính năng & Nâng cấp Mới

1. **Trực quan hóa Độ sâu 3D (3D Depth-Aware Rendering):**
   - Tọa độ $z$ của **MediaPipe Pose** được ánh xạ trực quan: Landmark càng gần màn hình ($z < 0$) thì **bán kính điểm càng lớn** và **màu sắc càng đậm nét/độ bão hòa cao**; landmark càng xa thì càng nhỏ và nhạt.
   - Đối với các mô hình 2D (**MoveNet, YOLO**): Tự động điều biến kích thước điểm và độ đậm màu theo **Confidence / Visibility Score**.

2. **Cấu hình Tập trung & Đồng nhất (`pose_models_config.json`):**
   - Toàn bộ siêu tham số, threshold, confidence, input resolution, camera config, benchmark settings và training hyper-parameters được lưu trong [`benchmarks/configs/pose_models_config.json`](file:///media/linhcbs/DATA/NCKH/NCKH%20H%C3%88/code/benchmarks/configs/pose_models_config.json).

3. **Hệ thống 7 Mô hình Toàn diện:**
   - **MediaPipe Pose LITE** (complexity=0, 33 3D Keypoints)
   - **MediaPipe Pose FULL** (complexity=1, 33 3D Keypoints)
   - **MoveNet Lightning** (17 Keypoints, siêu nhẹ, >120 FPS)
   - **MoveNet Thunder** (17 Keypoints, ResNet backbone)
   - **YOLO26n-pose (Nano)** (17 Keypoints, Multi-person)
   - **YOLO26m-pose (Medium)** (17 Keypoints, Multi-person cân bằng)
   - **YOLO26x-pose (XLarge)** (17 Keypoints, Multi-person độ chính xác cực đại)

4. **Benchmark Runner Chuyên nghiệp (`benchmark_runner.py`):**
   - Thu thập thống kê đầy đủ: FPS trung bình, min, max; Latency (Mean, Std, P25, P50, P75, P90, P95, P99); RAM Average/Peak.
   - Hỗ trợ chế độ **Synthetic Frames** tự sinh dữ liệu kiểm thử chuẩn hóa và lưu ảnh mẫu tại `benchmark_results/synthetic_samples/`.
   - Tự động xuất báo cáo Markdown [`report.md`](file:///media/linhcbs/DATA/NCKH/NCKH%20H%C3%88/code/benchmarks/benchmark_results/report.md) và file [`results.json`](file:///media/linhcbs/DATA/NCKH/NCKH%20H%C3%88/code/benchmarks/benchmark_results/results.json).

5. **Dễ dàng Mở rộng, Suy luận & Huấn luyện:**
   - Mỗi mô hình được đóng gói thành một Detector Class chuẩn hóa (`MediaPipePoseDetector`, `MoveNetPoseDetector`, `YOLOPoseDetector`).
   - Tích hợp script huấn luyện / fine-tune: [`train_yolo_pose.py`](file:///media/linhcbs/DATA/NCKH/NCKH%20H%C3%88/code/benchmarks/train_yolo_pose.py).

---

## 📁 Cấu trúc Thư mục

```
code/
├── README.md
└── benchmarks/
    ├── benchmark_runner.py          # Benchmark Runner đánh giá 7 mô hình
    ├── train_yolo_pose.py           # Script train/fine-tune YOLO-Pose
    ├── configs/
    │   └── pose_models_config.json  # File cấu hình tập trung
    ├── weights/                     # Thư mục chứa toàn bộ file trọng số (.task, .tflite, .pt)
    │   ├── pose_landmarker_lite.task
    │   ├── pose_landmarker_full.task
    │   ├── movenet_lightning.tflite
    │   ├── movenet_thunder.tflite
    │   ├── yolo11n-pose.pt / yolo26n-pose.pt
    │   ├── yolo11m-pose.pt / yolo26m-pose.pt
    │   └── yolo11x-pose.pt / yolo26x-pose.pt
    ├── benchmark_results/
    │   ├── report.md                # Báo cáo Markdown chi tiết
    │   ├── results.json             # Dữ liệu benchmark dạng JSON
    │   └── synthetic_samples/       # Ảnh frame giả lập được sinh ra
    └── models/
        ├── drawing_utils.py         # Utility vẽ 3D depth, HUD, load config & weights
        ├── mediapipe_pose_lite.py   # Module MP Lite (complexity 0)
        ├── mediapipe_pose_full.py   # Module MP Full (complexity 1)
        ├── movenet_lightning.py     # Module MoveNet Lightning
        ├── movenet_thunder.py       # Module MoveNet Thunder
        ├── yolo26n_pose.py          # Module YOLO26n Pose (Nano)
        ├── yolo26m_pose.py          # Module YOLO26m Pose (Medium)
        └── yolo26x_pose.py          # Module YOLO26x Pose (XLarge)
```

---

## 💻 Hướng dẫn Sử dụng

### 1. Chạy Benchmark
```bash
# Benchmark với Synthetic Frames chuẩn hóa (50 frames):
conda run -n ceoai2026 python benchmarks/benchmark_runner.py --no-webcam --frames 50

# Benchmark trực tiếp với Webcam:
conda run -n ceoai2026 python benchmarks/benchmark_runner.py --cam 1 --frames 100
```

### 2. Chạy Trực quan hóa Webcam từng Mô hình
```bash
# MediaPipe Pose Lite (3D Depth)
conda run -n ceoai2026 python benchmarks/models/mediapipe_pose_lite.py --cam 1

# MoveNet Lightning (Siêu nhanh)
conda run -n ceoai2026 python benchmarks/models/movenet_lightning.py --cam 1

# YOLO26 Pose (Nano / Medium / XLarge - Đa người)
conda run -n ceoai2026 python benchmarks/models/yolo26n_pose.py --cam 1
conda run -n ceoai2026 python benchmarks/models/yolo26m_pose.py --cam 1
conda run -n ceoai2026 python benchmarks/models/yolo26x_pose.py --cam 1
```

### 3. Huấn luyện / Fine-tune Mô hình YOLO-Pose
```bash
conda run -n ceoai2026 python benchmarks/train_yolo_pose.py --model yolo26n_pose --dataset coco8-pose.yaml --epochs 50
```

### 4. Chạy Phân Tích Tư Thế Thời Gian Thực (Ergonomic Posture Analysis)
```bash
# Chạy demo webcam với MediaPipe Pose + One-Euro Filter + HUD công thái học:
conda run -n ceoai2026 python demo_posture_webcam.py --camera 0

# Phím tắt trong khi chạy:
# [C]: Bắt đầu Auto-Calibration cá nhân hóa (ngồi thẳng 3 giây)
# [R]: Reset Calibration về ngưỡng lâm sàng mặc định
# [S]: Bật/Tắt bộ lọc One-Euro Filter làm mượt Landmark
# [W]: Bật/Tắt lớp hiển thị phát hiện mặt bàn & màn hình
# [Q] / [ESC]: Thoát
```

### 5. Chạy Kiểm Thử Tự Động (Unit & Integration Tests)
```bash
conda run -n ceoai2026 python -m pytest tests/test_posture_analyzer.py -v
```
### 6. Chạy Trực quan hóa Webcam từng Mô hình với Tốc độ & RAM
# 1. MediaPipe Pose Full (Mặc định)
/home/linhcbs/anaconda3/envs/ceoai2026/bin/python demo_posture_webcam.py --model mediapipe_pose_full

# 2. MediaPipe Pose Lite
/home/linhcbs/anaconda3/envs/ceoai2026/bin/python demo_posture_webcam.py --model mediapipe_pose_lite

# 3. MoveNet Lightning (Siêu nhẹ)
/home/linhcbs/anaconda3/envs/ceoai2026/bin/python demo_posture_webcam.py --model movenet_lightning

# 4. MoveNet Thunder
/home/linhcbs/anaconda3/envs/ceoai2026/bin/python demo_posture_webcam.py --model movenet_thunder

# 5. YOLO26n-pose (Nano Multi-person)
/home/linhcbs/anaconda3/envs/ceoai2026/bin/python demo_posture_webcam.py --model yolo26n_pose
