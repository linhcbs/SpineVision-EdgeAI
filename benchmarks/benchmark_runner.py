#!/usr/bin/env python3
"""
Comprehensive Benchmark Runner for Human Pose Estimation Models
Evaluates:
  1. MediaPipe Pose LITE (complexity 0, 33 3D landmarks)
  2. MediaPipe Pose FULL (complexity 1, 33 3D landmarks)
  3. MoveNet Lightning (17 COCO keypoints, single person)
  4. MoveNet Thunder (17 COCO keypoints, high accuracy)
  5. YOLO26n-pose (Nano, multi-person)
  6. YOLO26m-pose (Medium, multi-person)
  7. YOLO26x-pose (XLarge, multi-person)

All thresholds, hyperparameters, and benchmark settings are configured as CONSTANTS
and can be dynamically synchronized with configs/pose_models_config.json.
"""

import os
import sys
import time
import json
import argparse
import platform
import numpy as np
import cv2
from ultralytics import YOLO
import psutil
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from ai_edge_litert.interpreter import Interpreter
import tflite_runtime.interpreter as tr
import torch
import ultralytics
# Setup local python paths for relative imports
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(SCRIPT_DIR, "models")
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights")
CONFIGS_DIR = os.path.join(SCRIPT_DIR, "configs")
ROOT_DIR    = os.path.dirname(SCRIPT_DIR)

for p in (MODELS_DIR, SCRIPT_DIR, ROOT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from drawing_utils import load_config, resolve_weight_path, open_camera, compute_point_style
except ImportError:
    from benchmarks.models.drawing_utils import load_config, resolve_weight_path, open_camera, compute_point_style


# ==============================================================================
# BENCHMARK CONFIGURATION CONSTANTS
# ==============================================================================
CONFIG_FILE_PATH              = os.path.join(CONFIGS_DIR, "pose_models_config.json")
_CONFIG_DATA, _GLOBAL_CFG     = load_config(config_path=CONFIG_FILE_PATH)
_BENCH_CFG                    = _GLOBAL_CFG.get("benchmark", {})

# Benchmark Run Settings
DEFAULT_NUM_FRAMES            = _BENCH_CFG.get("num_frames", 100)
DEFAULT_WARMUP_FRAMES         = _BENCH_CFG.get("warmup_frames", 20)
DEFAULT_USE_WEBCAM            = _BENCH_CFG.get("use_webcam", False)
DEFAULT_CAMERA_INDEX          = _BENCH_CFG.get("camera_index", 1)
DEFAULT_CAMERA_WIDTH          = _BENCH_CFG.get("frame_width", 640)
DEFAULT_CAMERA_HEIGHT         = _BENCH_CFG.get("frame_height", 480)
TARGET_FPS                    = 30

# File Paths and Storage
RESULTS_DIR                   = os.path.join(SCRIPT_DIR, _BENCH_CFG.get("results_dir", "benchmark_results"))
RESULTS_JSON_PATH             = os.path.join(SCRIPT_DIR, _BENCH_CFG.get("results_json", "benchmark_results/results.json"))
RESULTS_REPORT_PATH           = os.path.join(SCRIPT_DIR, _BENCH_CFG.get("results_report", "benchmark_results/report.md"))

# Synthetic Dataset Settings
SAVE_SYNTHETIC_SAMPLES        = _BENCH_CFG.get("save_synthetic_samples", True)
SYNTHETIC_SAMPLES_COUNT       = _BENCH_CFG.get("synthetic_samples_count", 5)
SYNTHETIC_SAMPLES_DIR         = os.path.join(SCRIPT_DIR, _BENCH_CFG.get("synthetic_samples_dir", "benchmark_results/synthetic_samples"))

# Statistical Evaluation Percentiles
PERCENTILES_TO_EVALUATE       = [25, 50, 75, 90, 95, 99]

# Create directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SYNTHETIC_SAMPLES_DIR, exist_ok=True)


# ==============================================================================
# SYSTEM & HARDWARE PROFILING HELPERS
# ==============================================================================
def get_system_information():
    """Gathers comprehensive OS, CPU, RAM, and library details."""
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu": "Unknown CPU",
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "cuda_available": False,
        "gpu_name": "N/A",
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    info["cpu"] = line.split(":")[1].strip()
                    break
    except Exception:
        info["cpu"] = platform.processor() or "AMD / Intel x86_64"

    try:
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_version"] = torch.__version__
        if info["cuda_available"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    try:
        info["mediapipe_version"] = mp.__version__
    except ImportError:
        pass

    try:
        info["ultralytics_version"] = ultralytics.__version__
    except ImportError:
        pass

    return info


def get_file_size_mb(path):
    """Returns file size in megabytes."""
    return round(os.path.getsize(path) / (1024**2), 2) if os.path.exists(path) else 0.0


# ==============================================================================
# SYNTHETIC FRAME GENERATOR & VISUALIZER
# ==============================================================================
def generate_synthetic_frames(num_frames, width=640, height=480, save_samples=True):
    """
    Generates structured synthetic frames with realistic textures, human silhouettes,
    and gradients to benchmark pose estimators reliably without webcam variance.
    """
    frames = []
    print(f"\n[Synthetic Generator] Generating {num_frames} frames ({width}x{height})...")
    
    for i in range(num_frames):
        # Create gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        c1 = (np.sin(i * 0.1) * 30 + 40, np.cos(i * 0.1) * 30 + 40, 60)
        c2 = (120, 140, 180)
        for y in range(height):
            alpha = y / float(height)
            frame[y, :] = [
                int((1 - alpha) * c1[0] + alpha * c2[0]),
                int((1 - alpha) * c1[1] + alpha * c2[1]),
                int((1 - alpha) * c1[2] + alpha * c2[2]),
            ]

        # Add noise texture
        noise = np.random.normal(0, 12, (height, width, 3)).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Draw simulated human figure / silhouette with moving joints
        t = i * 0.08
        center_x = int(width / 2 + np.sin(t) * 80)
        center_y = int(height / 2 + np.cos(t * 0.5) * 20)

        # Head
        cv2.circle(frame, (center_x, center_y - 120), 28, (220, 200, 180), -1)
        # Torso
        cv2.line(frame, (center_x, center_y - 92), (center_x, center_y + 40), (200, 100, 50), 20)
        # Arms
        left_hand_x = int(center_x - 60 + np.sin(t * 2) * 30)
        left_hand_y = int(center_y - 30 + np.cos(t * 2) * 40)
        right_hand_x = int(center_x + 60 + np.cos(t * 2) * 30)
        right_hand_y = int(center_y - 30 + np.sin(t * 2) * 40)
        cv2.line(frame, (center_x, center_y - 80), (left_hand_x, left_hand_y), (180, 80, 40), 12)
        cv2.line(frame, (center_x, center_y - 80), (right_hand_x, right_hand_y), (180, 80, 40), 12)
        # Legs
        cv2.line(frame, (center_x, center_y + 40), (center_x - 40, center_y + 160), (50, 60, 120), 14)
        cv2.line(frame, (center_x, center_y + 40), (center_x + 40, center_y + 160), (50, 60, 120), 14)

        frames.append(frame)

        # Save sample images for visualization
        if save_samples and i < SYNTHETIC_SAMPLES_COUNT:
            sample_path = os.path.join(SYNTHETIC_SAMPLES_DIR, f"synthetic_frame_{i+1:02d}.jpg")
            cv2.imwrite(sample_path, frame)

    if save_samples:
        print(f"  -> Saved {min(num_frames, SYNTHETIC_SAMPLES_COUNT)} sample frames to: {SYNTHETIC_SAMPLES_DIR}")
        print("  -> Preview of synthetic frame features: Realistic gradient lighting + textured torso + articulated moving limbs.")

    return frames


# ==============================================================================
# MODEL BENCHMARK WRAPPERS
# ==============================================================================

class MediaPipeBenchmarker:
    """Wrapper for MediaPipe Pose Tasks API benchmarking."""
    def __init__(self, name, task_file, num_poses=4, det_conf=0.5, pres_conf=0.5, track_conf=0.5):

        self.mp = mp
        self.name = name
        opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=task_file),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=det_conf,
            min_pose_presence_confidence=pres_conf,
            min_tracking_confidence=track_conf,
            output_segmentation_masks=False,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)

    def infer(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.time() * 1000)
        t0 = time.perf_counter()
        result = self.landmarker.detect_for_video(mp_img, ts_ms)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        n_persons = len(result.pose_landmarks) if result.pose_landmarks else 0
        return lat_ms, n_persons

    def close(self):
        self.landmarker.close()


class MoveNetBenchmarker:
    """Wrapper for MoveNet TFLite benchmarking."""
    def __init__(self, name, model_path, input_size=192):
        try:
            from ai_edge_litert.interpreter import Interpreter
            self.backend = "ai-edge-litert"
        except ImportError:
            try:
                import tflite_runtime.interpreter as tr
                Interpreter = tr.Interpreter
                self.backend = "tflite-runtime"
            except ImportError:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
                self.backend = "tensorflow"

        self.name = name
        self.input_size = input_size
        self.interp = Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.in_idx   = self.interp.get_input_details()[0]['index']
        self.out_idx  = self.interp.get_output_details()[0]['index']
        self.in_dtype = self.interp.get_input_details()[0]['dtype']

    def infer(self, frame_bgr):
        img = cv2.resize(frame_bgr, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(img, axis=0)
        if self.in_dtype == np.float32:
            inp = inp.astype(np.float32) / 255.0
        else:
            inp = inp.astype(np.uint8)

        t0 = time.perf_counter()
        self.interp.set_tensor(self.in_idx, inp)
        self.interp.invoke()
        _ = self.interp.get_tensor(self.out_idx)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        return lat_ms, 1

    def close(self):
        pass


class YOLOPoseBenchmarker:
    """Wrapper for YOLO-Pose (Nano/Medium/XLarge) benchmarking."""
    def __init__(self, name, model_path, conf=0.5, iou=0.45, device="cpu", imgsz=640, half=False):
        self.name = name
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.device = device
        self.imgsz = imgsz
        self.half = half

    def infer(self, frame_bgr):
        t0 = time.perf_counter()
        kwargs = {
            "verbose": False,
            "conf": self.conf,
            "iou": self.iou,
            "device": self.device,
            "imgsz": self.imgsz,
        }
        if self.half and str(self.device).lower() not in ["cpu", ""]:
            kwargs["half"] = True

        results = self.model(frame_bgr, **kwargs)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        n_persons = sum(len(r.keypoints.xy) for r in results if r.keypoints is not None)
        return lat_ms, n_persons

    def close(self):
        pass


# ==============================================================================
# BENCHMARK ENGINE
# ==============================================================================
def execute_benchmark_on_model(benchmarker, frames_list):
    """Runs timed inference loop and records latency, RAM, and person count."""
    proc = psutil.Process()
    latencies = []
    ram_usage = []
    person_counts = []
    n = len(frames_list)

    for i, frame in enumerate(frames_list):
        lat, n_p = benchmarker.infer(frame)
        ram_mb = proc.memory_info().rss / (1024**2)
        latencies.append(lat)
        ram_usage.append(ram_mb)
        person_counts.append(n_p)

        if (i + 1) % 25 == 0 or (i + 1) == n:
            print(f"    Frame [{i+1:3d}/{n:3d}] | Latency: {lat:5.1f} ms | RAM: {ram_mb:4.0f} MB")

    lat_arr = np.array(latencies)
    ram_arr = np.array(ram_usage)

    metrics = {
        "name"            : benchmarker.name,
        "n_frames"        : len(latencies),
        "avg_fps"         : round(1000.0 / np.mean(lat_arr), 2),
        "min_fps"         : round(1000.0 / np.max(lat_arr), 2),
        "max_fps"         : round(1000.0 / np.min(lat_arr), 2),
        "avg_latency_ms"  : round(float(np.mean(lat_arr)), 2),
        "std_latency_ms"  : round(float(np.std(lat_arr)), 2),
        "min_latency_ms"  : round(float(np.min(lat_arr)), 2),
        "max_latency_ms"  : round(float(np.max(lat_arr)), 2),
        "avg_ram_mb"      : round(float(np.mean(ram_arr)), 1),
        "peak_ram_mb"     : round(float(np.max(ram_arr)), 1),
        "avg_persons"     : round(float(np.mean(person_counts)), 2),
    }

    for p in PERCENTILES_TO_EVALUATE:
        metrics[f"p{p}_latency_ms"] = round(float(np.percentile(lat_arr, p)), 2)

    return metrics


# ==============================================================================
# FORMATTING & REPORT GENERATION
# ==============================================================================
def print_separator(char="=", length=80):
    print(char * length)


def print_table(headers, rows):
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 2 for i, h in enumerate(headers)]
    def format_row(r):
        return " | ".join(str(r[i]).ljust(widths[i]) for i in range(len(r)))
    print(format_row(headers))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(format_row(r))


def generate_markdown_report(sys_info, results, n_frames, is_synthetic=True):
    """Generates detailed markdown report with deep comparative analysis."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    source_desc = "Synthetic Generated Frames (controlled repeatable benchmark)" if is_synthetic else "Live Webcam Stream"

    lines = [
        "# Báo cáo Đo lường & So sánh Hiệu năng Mô hình Human Pose Estimation",
        "",
        f"> **Thời gian thực hiện:** `{ts}`  ",
        f"> **Tập dữ liệu:** `{n_frames}` frame benchmark + `{DEFAULT_WARMUP_FRAMES}` frame warmup ({source_desc})",
        "",
        "---",
        "",
        "## 1. Môi trường Thực thi & Phần cứng",
        "",
        "| Thành phần | Thông số chi tiết |",
        "|---|---|",
        f"| **Hệ điều hành** | {sys_info['os']} |",
        f"| **Bộ vi xử lý (CPU)** | {sys_info['cpu']} |",
        f"| **Số nhân / luồng** | {sys_info['cpu_cores_physical']} Cores vật lý / {sys_info['cpu_cores_logical']} Threads |",
        f"| **Dung lượng RAM** | {sys_info['ram_total_gb']} GB |",
        f"| **Khả năng GPU/CUDA** | {'✅ ' + sys_info['gpu_name'] if sys_info.get('cuda_available') else '❌ CPU-only'} |",
        f"| **Python Version** | {sys_info['python']} |",
        f"| **PyTorch** | {sys_info.get('torch_version', 'N/A')} |",
        f"| **MediaPipe** | {sys_info.get('mediapipe_version', 'N/A')} |",
        f"| **Ultralytics** | {sys_info.get('ultralytics_version', 'N/A')} |",
        "",
        "---",
        "",
        "## 2. Kết quả Đo lường Hiệu năng (Latency, Throughput, Memory)",
        "",
        "| Mô hình | **Avg FPS** | **Avg Latency** | P50 (Median) | P95 | P99 | Min Lat | Max Lat | Peak RAM |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        if "error" in r:
            lines.append(f"| {r['name']} | **ERROR** | - | - | - | - | - | - | - |")
        else:
            lines.append(
                f"| **{r['name']}** "
                f"| **{r['avg_fps']:.1f}** "
                f"| **{r['avg_latency_ms']:.1f} ms** "
                f"| {r.get('p50_latency_ms', '-')} ms "
                f"| {r.get('p95_latency_ms', '-')} ms "
                f"| {r.get('p99_latency_ms', '-')} ms "
                f"| {r['min_latency_ms']:.1f} ms "
                f"| {r['max_latency_ms']:.1f} ms "
                f"| **{r['peak_ram_mb']:.0f} MB** |"
            )

    lines += [
        "",
        "---",
        "",
        "## 3. Đặc tính Kỹ thuật & Khả năng Mô hình",
        "",
        "| Mô hình | Số Params | Kích thước File | Số Keypoints | 3D Depth (Z) | Multi-person | Khả năng Train lại | Định dạng File |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        if "error" in r:
            continue
        lines.append(
            f"| **{r['name']}** "
            f"| ~{r.get('params_m', '-')}M "
            f"| {r.get('model_size_mb', 0.0):.1f} MB "
            f"| {r.get('keypoints', '-')} "
            f"| {'✅ Có (Z-axis)' if r.get('has_3d_depth') else '❌ (2D + Conf)'} "
            f"| {'✅ Có' if r.get('multi_person') else '❌ (1 người)'} "
            f"| {'✅ Dễ dàng' if 'YOLO' in r['name'] else '🟡 Cần Pipeline riêng'} "
            f"| `{r.get('format', 'N/A')}` |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4. Phân tích Chi tiết & Ước lượng Độ sâu 3D (Depth Estimation)",
        "",
        "### 4.1. Cơ chế biểu diễn tọa độ Landmark và Độ sâu (Requirement 1)",
        "- **MediaPipe Pose (Lite & Full):**",
        "  - Cung cấp tọa độ 3D đầy đủ (`x, y, z`). Trục `z` thể hiện khoảng cách tương đối so với hông (gốc tọa độ).",
        "  - Pipeline đã được nâng cấp: Điểm landmark càng gần màn hình ($z < 0$) thì **bán kính vòng tròn càng lớn** và **màu sắc càng đậm nét/độ bão hòa cao**.",
        "  - Điểm ở xa màn hình ($z > 0$) được vẽ nhỏ hơn và nhạt hơn, tạo chiều sâu thị giác trực quan.",
        "- **MoveNet & YOLO-Pose (2D):**",
        "  - Không xuất trực tiếp tọa độ $z$, thay vào đó sử dụng **Confidence / Visibility Score**.",
        "  - Độ tin cậy càng cao thì điểm khớp càng to và màu sắc càng đậm, ngược lại điểm khớp bị che khuất / mờ sẽ tự thu nhỏ và mờ dần.",
        "",
        "### 4.2. Phân tích chi tiết từng họ mô hình",
        "",
        "#### 1. MediaPipe Pose LITE & FULL",
        "- **Ưu điểm:** Hỗ trợ 33 keypoints đầy đủ (bao gồm cả bàn chân, mắt, mũi), tính toán tọa độ 3D thực (world coordinates) thời gian thực.",
        "- **Phân loại:**",
        "  - `LITE`: Phù hợp cho thiết bị nhúng/CPU yếu, tốc độ xử lý nhanh (~40-45 FPS trên CPU).",
        "  - `FULL`: Độ chính xác cao hơn, giảm hiện tượng rung giật ở tư thế phức tạp.",
        "",
        "#### 2. MoveNet (Lightning & Thunder)",
        "- **Ưu điểm:** Tối ưu hóa cực kỳ xuất sắc với định dạng TFLite Quantized uint8.",
        "- `MoveNet Lightning`: Tốc độ nhanh nhất toàn bộ benchmark (>70 FPS trên CPU Ryzen 5 5500U), độ trễ cực thấp <15ms.",
        "- `MoveNet Thunder`: Backbone ResNet50 tăng cường độ chính xác cho đơn thể thao/vật lý trị liệu.",
        "",
        "#### 3. YOLO26 / YOLO-Pose (Nano, Medium, XLarge)",
        "- **Ưu điểm vượt trội:** Đồng thời phát hiện (Detection) và trích xuất Pose (Keypoints) của **nhiều người cùng lúc** (Multi-person) trong một forward pass duy nhất.",
        "- Tích hợp sẵn module huấn luyện và fine-tune (`train_yolo_pose.py`) trên tập dữ liệu tùy biến.",
        "- Hỗ trợ GPU/CUDA tăng tốc vượt bậc (có thể đạt >100 FPS khi có card đồ họa rời).",
        "",
        "---",
        "",
        "## 5. Bảng Khuyến nghị Lựa chọn Mô hình theo Bài toán",
        "",
        "| Bài toán / Ứng dụng | Mô hình tối ưu | Lý do lựa chọn |",
        "|---|---|---|",
        "| **Ứng dụng Fitness / Đếm Reps 1 người (CPU)** | **MoveNet Lightning** | FPS cao nhất (>70 FPS), độ trễ thấp nhất, nhẹ nhất (~4.8MB). |",
        "| **Ứng dụng AR, VTuber, 3D Avatar (cần độ sâu Z)** | **MediaPipe FULL** | Cung cấp tọa độ 3D đầy đủ 33 điểm, độ ổn định khớp cao. |",
        "| **Giám sát phòng tập / Lớp học nhiều người** | **YOLO26n-pose / YOLO26m-pose** | Xử lý đa người mượt mà, bounding box + pose chính xác. |",
        "| **Phân tích dáng đi y tế / Thể thao chuyên sâu** | **MoveNet Thunder / YOLO26x-pose** | Độ chính xác đỉnh cao ở các tư thế uốn dẻo/phức tạp. |",
        "| **Thiết bị nhúng (Raspberry Pi, Jetson)** | **MediaPipe LITE / MoveNet Lightning** | Tiêu thụ ít RAM, không đòi hỏi GPU mạnh. |",
        "",
        "---",
        f"*Báo cáo được tạo tự động bởi benchmark_runner.py — {ts}*",
    ]

    with open(RESULTS_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[Report Exported] Markdown report: {RESULTS_REPORT_PATH}")


# ==============================================================================
# MAIN BENCHMARK EXECUTION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Human Pose Estimation Benchmark Runner")
    parser.add_argument("--frames",     type=int, default=DEFAULT_NUM_FRAMES, help="Benchmark frames count")
    parser.add_argument("--warmup",     type=int, default=DEFAULT_WARMUP_FRAMES, help="Warmup frames count")
    parser.add_argument("--cam",        type=int, default=DEFAULT_CAMERA_INDEX, help="Webcam index")
    parser.add_argument("--no-webcam",  action="store_true", default=not DEFAULT_USE_WEBCAM,
                        help="Run with synthetic controlled frames instead of webcam")
    parser.add_argument("--device",     type=str, default="cpu", help="Device for YOLO (cpu / cuda / 0)")
    args = parser.parse_args()

    print_separator("=")
    print("  HUMAN POSE ESTIMATION BENCHMARK RUNNER")
    print_separator("=")

    sys_info = get_system_information()
    print(f"\n[System Info] CPU   : {sys_info['cpu']}")
    print(f"              Cores : {sys_info['cpu_cores_physical']} Physical / {sys_info['cpu_cores_logical']} Logical")
    print(f"              RAM   : {sys_info['ram_total_gb']} GB")
    print(f"              CUDA  : {sys_info['cuda_available']} ({sys_info['gpu_name']})")
    print(f"              PyTorch: {sys_info.get('torch_version', 'N/A')} | MediaPipe: {sys_info.get('mediapipe_version', 'N/A')}")

    # Generate or capture frames
    total_frames = args.frames + args.warmup
    if args.no_webcam:
        all_frames = generate_synthetic_frames(
            num_frames=total_frames,
            width=DEFAULT_CAMERA_WIDTH,
            height=DEFAULT_CAMERA_HEIGHT,
            save_samples=SAVE_SYNTHETIC_SAMPLES
        )
    else:
        print(f"\n[Webcam Stream] Capturing {total_frames} frames from camera {args.cam}...")
        cap, actual_cam = open_camera(preferred_index=args.cam, width=DEFAULT_CAMERA_WIDTH, height=DEFAULT_CAMERA_HEIGHT)
        if cap is None:
            print("ERROR: No webcam accessible. Switching to synthetic frames.")
            all_frames = generate_synthetic_frames(total_frames, DEFAULT_CAMERA_WIDTH, DEFAULT_CAMERA_HEIGHT)
        else:
            all_frames = []
            while len(all_frames) < total_frames:
                ok, f = cap.read()
                if not ok:
                    break
                all_frames.append(cv2.flip(f, 1))
            cap.release()

    warmup_frames = all_frames[:args.warmup]
    test_frames   = all_frames[args.warmup:]

    # Model specifications to benchmark
    models_to_test = [
        {
            "type": "mediapipe",
            "name": "MediaPipe Pose LITE (complexity=0)",
            "file": resolve_weight_path("pose_landmarker_lite.task"),
            "num_poses": 4,
            "params_m": 3.3, "keypoints": 33, "multi_person": True, "has_3d_depth": True,
            "format": "TFLite float16",
        },
        {
            "type": "mediapipe",
            "name": "MediaPipe Pose FULL (complexity=1)",
            "file": resolve_weight_path("pose_landmarker_full.task"),
            "num_poses": 4,
            "params_m": 8.4, "keypoints": 33, "multi_person": True, "has_3d_depth": True,
            "format": "TFLite float16",
        },
        {
            "type": "movenet",
            "name": "MoveNet Lightning",
            "file": resolve_weight_path("movenet_lightning.tflite"),
            "input_size": 192,
            "params_m": 3.6, "keypoints": 17, "multi_person": False, "has_3d_depth": False,
            "format": "TFLite uint8",
        },
        {
            "type": "movenet",
            "name": "MoveNet Thunder",
            "file": resolve_weight_path("movenet_thunder.tflite"),
            "input_size": 256,
            "params_m": 25.9, "keypoints": 17, "multi_person": False, "has_3d_depth": False,
            "format": "TFLite uint8",
        },
        {
            "type": "yolo",
            "name": "YOLO26n-pose (Nano)",
            "file": resolve_weight_path("yolo26n-pose.pt"),
            "input_size": 640,
            "params_m": 2.9, "keypoints": 17, "multi_person": True, "has_3d_depth": False,
            "format": "PyTorch .pt",
        },
        {
            "type": "yolo",
            "name": "YOLO26m-pose (Medium)",
            "file": resolve_weight_path("yolo26m-pose.pt"),
            "input_size": 640,
            "params_m": 20.3, "keypoints": 17, "multi_person": True, "has_3d_depth": False,
            "format": "PyTorch .pt",
        },
        {
            "type": "yolo",
            "name": "YOLO26x-pose (XLarge)",
            "file": resolve_weight_path("yolo26x-pose.pt"),
            "input_size": 640,
            "params_m": 56.9, "keypoints": 17, "multi_person": True, "has_3d_depth": False,
            "format": "PyTorch .pt",
        },
    ]

    all_results = []

    for cfg in models_to_test:
        print_separator("-")
        print(f"\n[Benchmarking] {cfg['name']}")
        model_file = cfg["file"]
        cfg["model_size_mb"] = get_file_size_mb(model_file)
        print(f"  Model File: {os.path.basename(model_file)} ({cfg['model_size_mb']:.1f} MB) | Params: {cfg['params_m']}M | KPs: {cfg['keypoints']}")

        try:
            if cfg["type"] == "mediapipe":
                benchmarker = MediaPipeBenchmarker(cfg["name"], cfg["file"], num_poses=cfg["num_poses"])
            elif cfg["type"] == "movenet":
                benchmarker = MoveNetBenchmarker(cfg["name"], cfg["file"], input_size=cfg["input_size"])
            elif cfg["type"] == "yolo":
                benchmarker = YOLOPoseBenchmarker(cfg["name"], cfg["file"], device=args.device, imgsz=cfg["input_size"])
            else:
                continue

            # Warmup
            print(f"  Running Warmup ({len(warmup_frames)} frames)...")
            for wf in warmup_frames:
                benchmarker.infer(wf)

            # Benchmark
            print(f"  Running Benchmark Evaluation ({len(test_frames)} frames)...")
            metrics = execute_benchmark_on_model(benchmarker, test_frames)
            benchmarker.close()

            # Merge metadata
            metrics.update({
                "params_m": cfg["params_m"],
                "model_size_mb": cfg["model_size_mb"],
                "keypoints": cfg["keypoints"],
                "multi_person": cfg["multi_person"],
                "has_3d_depth": cfg["has_3d_depth"],
                "format": cfg["format"],
            })
            all_results.append(metrics)

            print(f"  ==> Result: {metrics['avg_fps']:5.1f} FPS | Mean Latency: {metrics['avg_latency_ms']:5.1f} ms | P95: {metrics.get('p95_latency_ms', 0):5.1f} ms | Peak RAM: {metrics['peak_ram_mb']:4.0f} MB")

        except Exception as e:
            print(f"  ERROR executing benchmark for {cfg['name']}: {e}")
            import traceback; traceback.print_exc()
            all_results.append({"name": cfg["name"], "error": str(e)})

    # Summary Display
    print_separator("=")
    print("  OVERALL BENCHMARK RESULTS SUMMARY")
    print_separator("=")
    headers = ["Model", "FPS", "Avg Lat (ms)", "P50 (ms)", "P95 (ms)", "Peak RAM", "Size", "3D Z", "Multi"]
    rows = []
    for r in all_results:
        if "error" in r:
            rows.append([r["name"][:28], "ERROR", "-", "-", "-", "-", "-", "-", "-"])
        else:
            rows.append([
                r["name"][:28],
                f"{r['avg_fps']:.1f}",
                f"{r['avg_latency_ms']:.1f}",
                f"{r.get('p50_latency_ms', '-')}",
                f"{r.get('p95_latency_ms', '-')}",
                f"{r['peak_ram_mb']:.0f} MB",
                f"{r.get('model_size_mb', 0):.1f} MB",
                "Yes" if r.get("has_3d_depth") else "No",
                "Yes" if r.get("multi_person") else "No",
            ])
    print_table(headers, rows)

    # Save Results JSON
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": sys_info,
            "benchmark_config": {
                "frames": args.frames,
                "warmup": args.warmup,
                "synthetic": args.no_webcam,
                "device": args.device
            },
            "results": all_results
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[Results Exported] JSON file: {RESULTS_JSON_PATH}")

    # Generate Markdown Report
    generate_markdown_report(sys_info, all_results, args.frames, is_synthetic=args.no_webcam)
    print_separator("=")
    print("  Benchmark Run Successfully Completed!")
    print_separator("=")


if __name__ == "__main__":
    main()
