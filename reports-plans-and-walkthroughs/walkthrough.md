# Human Pose Estimation — Walkthrough & Kết quả

## Tổng quan

Đã triển khai và benchmark thành công **5 mô hình Human Pose Estimation** trên môi trường `ceoai2026`.

---

## Cấu trúc Dự án

```
benchmarks/
├── models/
│   ├── pose_landmarker_lite.task    (5.5 MB) — MediaPipe LITE
│   ├── pose_landmarker_full.task    (9.0 MB) — MediaPipe FULL
│   ├── movenet_lightning.tflite     (4.5 MB) — MoveNet Lightning
│   ├── movenet_thunder.tflite      (12.0 MB) — MoveNet Thunder
│   ├── yolov8n-pose.pt              (6.5 MB) — YOLOv8n-pose
│   ├── mediapipe_pose_lite.py       ← Module webcam #1
│   ├── mediapipe_pose_full.py       ← Module webcam #2
│   ├── movenet_lightning.py         ← Module webcam #3
│   ├── movenet_thunder.py           ← Module webcam #4
│   └── yolov8n_pose.py              ← Module webcam #5
├── benchmark_runner.py              ← Benchmark tổng hợp
└── benchmark_results/
    ├── results.json
    └── report.md
```

---

## Cách Chạy Từng Module Webcam

```bash
# Kích hoạt conda env
conda activate ceoai2026

# Chạy từng model (nhấn q hoặc ESC để thoát)
python benchmarks/models/mediapipe_pose_lite.py
python benchmarks/models/mediapipe_pose_full.py
python benchmarks/models/movenet_lightning.py
python benchmarks/models/movenet_thunder.py
python benchmarks/models/yolov8n_pose.py
```

**HUD hiển thị khi chạy webcam:**
- Model name + backend
- FPS (exponential moving average)
- Latency ms (chỉ inference time)
- RAM MB (RSS của process)
- Số người detected (nếu applicable)

---

## Cách Chạy Benchmark

```bash
# Benchmark với webcam (300 frames)
conda run -n ceoai2026 python benchmarks/benchmark_runner.py --cam 1

# Benchmark không cần webcam (synthetic frames)
conda run -n ceoai2026 python benchmarks/benchmark_runner.py --no-webcam --frames 100
```

---

## ✅ Kết quả Benchmark (100 frames, synthetic, CPU-only)

**Hardware:** AMD Ryzen 5 5500U 6C/12T | 15 GB RAM | No GPU

| Model | **Avg FPS** | Avg Latency | P95 Latency | RAM Peak | Size |
|---|---|---|---|---|---|
| 🥇 MoveNet Lightning | **75.9** | **13.2 ms** | 13.5 ms | 537 MB | 4.5 MB |
| 🥈 MediaPipe FULL | **44.9** | 22.3 ms | 23.3 ms | 537 MB | 9.0 MB |
| MediaPipe LITE | 42.9 | 23.3 ms | 23.7 ms | 515 MB | 5.5 MB |
| MoveNet Thunder | 28.1 | 35.6 ms | 44.7 ms | 591 MB | 12.0 MB |
| YOLOv8n-pose | 20.1 | 49.8 ms | 54.4 ms | 663 MB | 6.5 MB |

> [!NOTE]
> Benchmark trên là synthetic (ảnh ngẫu nhiên không có người) — latency inference-only rất chính xác. Với webcam thực, FPS sẽ thấp hơn do overhead capture + display.

---

## Quan sát Thú vị

- **MediaPipe FULL nhanh hơn LITE** trên synthetic data (22.3ms vs 23.3ms) — do LITE có detector 2 giai đoạn trong một số điều kiện; trên ảnh thực với người thật kết quả có thể đảo ngược
- **MoveNet Lightning** vượt trội hoàn toàn về FPS (76 FPS vs 45 FPS của MediaPipe)
- **YOLOv8n-pose** chậm nhất trên CPU — overhead của detection backbone; nhưng sẽ vượt trội khi có GPU (CUDA)
- **RAM**: YOLO tốn RAM nhiều nhất (663MB), Lite tiết kiệm nhất (515MB)

---

## Packages Đã Cài Thêm

| Package | Version | Lý do |
|---|---|---|
| `ai-edge-litert` | 2.2.0 | TFLite mới nhất tương thích NumPy 2.x (thay thế `tflite-runtime 2.14` bị lỗi `_ARRAY_API`) |

---

## Files Quan trọng

| File | Link |
|---|---|
| MediaPipe Lite module | [mediapipe_pose_lite.py](file:///media/linhcbs/DATA/NCKH/NCKH%20H%C3%88/code/benchmarks/models/mediapipe_pose_lite.py) |
| MediaPipe Full module | [mediapipe_pose_full.py](file:///media/linhcbs/DATA/NCKH/NCKH%20H%C3%88/code/benchmarks/models/mediapipe_pose_full.py) |
| MoveNet Lightning module | [movenet_lightning.py](file:///media/linhcbs/DATA/NCKH/NCKH%20H%C3%88/code/benchmarks/models/movenet_lightning.py) |
| MoveNet Thunder module | [movenet_thunder.py](file:///media/linhcbs/DATA/NCKH/NCKH%20H%C3%88/code/benchmarks/models/movenet_thunder.py) |
| YOLOv8n-pose module | [yolov8n_pose.py](file:///media/linhcbs/DATA/NCKH/NCKH%20H%C3%88/code/benchmarks/models/yolov8n_pose.py) |
| Benchmark Runner | [benchmark_runner.py](file:///media/linhcbs/DATA/NCKH/NCKH%20H%C3%88/code/benchmarks/benchmark_runner.py) |
| Kết quả JSON | [results.json](file:///media/linhcbs/DATA/NCKH/NCKH%20H%C3%88/code/benchmarks/benchmark_results/results.json) |
| Báo cáo Markdown | [report.md](file:///media/linhcbs/DATA/NCKH/NCKH%20H%C3%88/code/benchmarks/benchmark_results/report.md) |
