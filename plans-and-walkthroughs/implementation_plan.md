# Human Pose Estimation — Triển khai & Benchmark 5 Mô hình

## Tổng quan

Triển khai 5 module inference với webcam + vẽ skeleton, kèm bản báo cáo so sánh chi tiết cho các mô hình:
1. **MediaPipe Pose (complexity 0)** — Lite/Blazepose, Tasks API mới nhất
2. **MediaPipe Pose (complexity 1)** — Full/Blazepose, Tasks API mới nhất
3. **MoveNet Lightning** — TFLite qua `tflite-runtime` hoặc TF
4. **MoveNet Thunder** — TFLite qua `tflite-runtime` hoặc TF
5. **YOLOv8n-pose** — `ultralytics` (đã cài v8.4.92)

**Hardware:** AMD Ryzen 5 5500U (6C/12T), 15 GB RAM, No GPU (CPU-only inference)  
**Environment:** `ceoai2026` conda (Python 3.11.15)

---

## Packages đã có

| Package | Version |
|---|---|
| `mediapipe` | 1.0.1 |
| `ultralytics` | 8.4.92 |
| `torch` | 2.10.0 |
| `opencv-python` | 4.13.0 |
| `psutil` | 7.0.0 |
| `numpy` | 2.4.6 |

> [!IMPORTANT]
> TensorFlow/TFLite **chưa cài**. MoveNet chạy qua TFLite. Sẽ cài `tflite-runtime` (nhẹ hơn full TF) khi execute.
> MediaPipe Pose (complexity 0/1) trong Tasks API mới dùng file `.task` (Blazepose lite / full). Đã có `pose_landmarker_full.task` (9.4 MB), cần tải thêm `pose_landmarker_lite.task`.

---

## Cấu trúc thư mục đề xuất

```
benchmarks/
├── models/
│   ├── pose_landmarker_lite.task       # MediaPipe Pose complexity=0
│   ├── pose_landmarker_full.task       # MediaPipe Pose complexity=1 (đã có)
│   ├── movenet_lightning.tflite        # MoveNet Lightning
│   ├── movenet_thunder.tflite          # MoveNet Thunder
│   ├── yolov8n-pose.pt                 # YOLOv8n-pose (auto-download)
│   ├── mediapipe_pose_lite.py          # Module 1
│   ├── mediapipe_pose_full.py          # Module 2
│   ├── movenet_lightning.py            # Module 3
│   ├── movenet_thunder.py              # Module 4
│   ├── yolov8n_pose.py                 # Module 5
│   └── test.py                         # File hiện tại (giữ nguyên)
├── benchmark_runner.py                 # Script chạy benchmark tổng hợp
├── benchmark_results/                  # Thư mục lưu kết quả
│   ├── results.json
│   └── report.md
└── configs/
```

---

## Nội dung mỗi module (webcam runner)

Mỗi file `*_pose.py` sẽ có:
- **HUD overlay** (góc trên trái): FPS, Latency (ms), RAM (MB), Model name
- **Skeleton drawing** với màu sắc phân biệt per-model
- **Auto camera fallback** (quét index 0–5)
- **Graceful exit** khi nhấn `q` hoặc `ESC`

Các chỉ số runtime được tính:
- `FPS` = exponential moving average (α=0.1)
- `Latency (ms)` = thời gian inference only (không tính capture/display)
- `RAM (MB)` = `psutil.Process().memory_info().rss / 1024**2`

---

## Xử lý MoveNet (TFLite)

MoveNet chỉ hỗ trợ single-person. Sẽ dùng `tflite-runtime` (không cần full TF):

```bash
pip install tflite-runtime
```

Nếu không khả dụng trên Python 3.11 Linux, fallback sang `tensorflow` (chỉ import `tf.lite.Interpreter`).

Model URLs:
- Lightning: `https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4`
- Thunder: `https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4`

---

## Benchmark Runner

`benchmark_runner.py` sẽ:
1. Mở webcam, đọc N frames (mặc định 300)
2. Chạy inference cho từng model, thu thập metrics
3. Xuất `results.json` + in bảng so sánh ra terminal + lưu `report.md`

**Metrics thu thập:**
| Metric | Mô tả |
|---|---|
| Avg FPS | FPS trung bình |
| Avg Latency (ms) | Latency inference trung bình |
| P95 Latency (ms) | Latency percentile 95 |
| Peak RAM (MB) | RAM tối đa trong session |
| Avg RAM (MB) | RAM trung bình |
| Model size (MB) | Dung lượng file model |

---

## Báo cáo so sánh (report.md)

Bao gồm:
- Bảng hiệu năng (FPS, Latency, RAM)
- Bảng thông tin model (params, size, multi-person support)
- Phân tích: hoạt động tốt/kém khi nào & tại sao
- Thông tin phần cứng test

---

## Open Questions

> [!IMPORTANT]
> **MoveNet TFLite**: `tflite-runtime` có thể không build sẵn cho Python 3.11 trên Linux x86_64 mới nhất. Nếu không cài được, sẽ dùng `tensorflow` (nặng hơn ~500 MB). Bạn có muốn tôi thử `tflite-runtime` trước không, hay dùng luôn `tensorflow`?

> [!NOTE]
> **Camera index**: Code hiện tại dùng `CAM_INDEX = 1`. Benchmark runner sẽ dùng index 1 làm default, tự động fallback. Bạn có muốn thay đổi không?

> [!NOTE]
> **Số frames benchmark**: Mặc định 300 frames (~10-15 giây mỗi model). Bạn muốn nhiều hơn/ít hơn?

---

## Verification Plan

### Automated
```bash
# Kiểm tra import tất cả module
conda run -n ceoai2026 python -c "import mediapipe, ultralytics, cv2, psutil, numpy; print('OK')"

# Chạy benchmark (không cần webcam, dùng synthetic frames)
conda run -n ceoai2026 python benchmark_runner.py --no-webcam --frames 50
```

### Manual
- Chạy từng module riêng, kiểm tra webcam + skeleton display + HUD
- Xem kết quả `report.md`
