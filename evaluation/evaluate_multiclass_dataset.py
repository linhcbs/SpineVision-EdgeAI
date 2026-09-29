#!/usr/bin/env python3
"""
Evaluation of Lightweight HPE Models on Sitting Posture Multiclass Dataset.
Dataset: /media/linhcbs/DATA/NCKH/NCKH HÈ/DATA/Sitting Posture.v2i.multiclass

Plug-and-play evaluation of the 5 lightest HPE models (excluding YOLO-m and YOLO-x):
  1. MediaPipe Pose LITE      (complexity=0, 33 kps, 5.6 MB, TFLite)
  2. MediaPipe Pose FULL      (complexity=1, 33 kps, 9.2 MB, TFLite)
  3. MoveNet Lightning        (192x192, 17 kps, 4.6 MB, TFLite)
  4. MoveNet Thunder          (256x256, 17 kps, 12.3 MB, TFLite)
  5. YOLO26n-Pose (Nano)      (640x640, 17 kps, 6.1 MB, PyTorch/Ultralytics)

Task: Binary Posture Classification (Good vs Bad)
  - Target classes: 'Good' (good_posture) vs 'Bad' (bad_posture)
  - Sub-types such as backward_lean and forward_lean are mapped into 'Bad'
  - Splits evaluated: train, valid (eval), test

Outputs:
  - Markdown Report: results/hpe_eval_multiclass_report.md
  - Summary JSON:    results/hpe_eval_multiclass_summary.json
  - Full JSON:       results/hpe_eval_multiclass_full.json
"""

import os
import sys
import time
import json
import logging
import argparse
import datetime
import traceback
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from posture_analyzer import (
    create_hpe_detector,
    PostureStatus,
)
from posture_analyzer.posture_metrics import PostureMetricsCalculator
from posture_analyzer.posture_classifier import PostureClassifier
from posture_analyzer.types import PostureMetrics, CalibrationData

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DATASET_ROOT = Path(
    "/media/linhcbs/DATA/NCKH/NCKH HÈ/DATA/Sitting Posture.v2i.multiclass"
)

DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"

MODELS_TO_EVALUATE = [
    "mediapipe_pose_lite",
    "mediapipe_pose_full",
    "movenet_lightning",
    "movenet_thunder",
    "yolo26n_pose",
]

SPLITS = ["train", "valid", "test"]

MODEL_INFO = {
    "mediapipe_pose_lite": {
        "display_name": "MediaPipe Pose LITE",
        "keypoints": 33,
        "weight_kb": 5642,
        "backend": "TFLite/MediaPipe",
        "input_size": "dynamic",
    },
    "mediapipe_pose_full": {
        "display_name": "MediaPipe Pose FULL",
        "keypoints": 33,
        "weight_kb": 9178,
        "backend": "TFLite/MediaPipe",
        "input_size": "dynamic",
    },
    "movenet_lightning": {
        "display_name": "MoveNet Lightning",
        "keypoints": 17,
        "weight_kb": 4647,
        "backend": "TFLite",
        "input_size": "192x192",
    },
    "movenet_thunder": {
        "display_name": "MoveNet Thunder",
        "keypoints": 17,
        "weight_kb": 12289,
        "backend": "TFLite",
        "input_size": "256x256",
    },
    "yolo26n_pose": {
        "display_name": "YOLO26n-Pose (Nano)",
        "keypoints": 17,
        "weight_kb": 6109,
        "backend": "PyTorch/Ultralytics",
        "input_size": "640x640",
    },
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("multiclass_eval")


# ===========================================================================
# Helper functions
# ===========================================================================

def posture_status_to_label(status: PostureStatus) -> str:
    """Map system PostureStatus enum to binary Good / Bad label."""
    if status == PostureStatus.GOOD:
        return "Good"
    return "Bad"


def format_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


# ===========================================================================
# Multiclass Posture Dataset Loader
# ===========================================================================

class MulticlassPostureDataset:
    """
    Loads images and labels from Roboflow multiclass export folder containing _classes.csv.
    Maps:
      - goodposture == 1 -> 'Good'
      - backwardbadposture == 1 or forwardbadposture == 1 -> 'Bad'
    Resolves ties/unlabeled via filename prefixes:
      - good_posture-* -> 'Good'
      - backward_lean_bad_posture-* or forward_lean_bad_posture-* -> 'Bad'
    """

    def __init__(self, split_dir: Path):
        self.split_dir = split_dir
        csv_path = split_dir / "_classes.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Classes CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]

        self.samples: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            filename = str(row["filename"]).strip()
            img_path = split_dir / filename
            if not img_path.exists():
                continue

            # Original multiclass flags
            is_unlabeled = int(row.get("Unlabeled", 0)) == 1
            is_back = int(row.get("backwardbadposture", 0)) == 1
            is_fwd = int(row.get("forwardbadposture", 0)) == 1
            is_good = int(row.get("goodposture", 0)) == 1

            # Sub-category tracking
            if filename.startswith("backward_lean") or is_back:
                sub_type = "backward_lean"
            elif filename.startswith("forward_lean") or is_fwd:
                sub_type = "forward_lean"
            else:
                sub_type = "good_posture"

            # Binary Label Mapping (Good vs Bad)
            if is_good and not is_back and not is_fwd:
                label = "Good"
            elif (is_back or is_fwd) and not is_good:
                label = "Bad"
            else:
                # Ambiguous / tie / unlabeled: use filename prefix as canonical source
                if filename.startswith("good_posture"):
                    label = "Good"
                elif filename.startswith("backward_lean_bad_posture") or filename.startswith("forward_lean_bad_posture"):
                    label = "Bad"
                else:
                    label = "Good" if is_good else "Bad"

            self.samples.append({
                "filename": filename,
                "path": img_path,
                "label": label,
                "sub_type": sub_type,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def items(self):
        for sample in self.samples:
            yield sample["path"], sample["label"], sample["sub_type"]

    def class_distribution(self) -> Dict[str, int]:
        counts: Counter = Counter()
        for s in self.samples:
            counts[s["label"]] += 1
        return dict(counts)

    def subtype_distribution(self) -> Dict[str, int]:
        counts: Counter = Counter()
        for s in self.samples:
            counts[s["sub_type"]] += 1
        return dict(counts)


# ===========================================================================
# Posture Predictor
# ===========================================================================

class PosturePredictor:
    """Wraps the full posture pipeline: image -> HPE -> metrics -> classifier -> label."""

    def __init__(self, model_key: str, device: str = "cpu"):
        self.model_key = model_key
        log.info(f"Loading HPE model: {model_key}")
        t0 = time.perf_counter()
        self.detector = create_hpe_detector(model_key, device=device)
        self.load_time_s = time.perf_counter() - t0
        log.info(f"  -> Loaded in {self.load_time_s:.2f}s")

        self.metrics_calc = PostureMetricsCalculator()
        self.classifier = PostureClassifier(mode="hybrid")
        self._last_ts_ms = 0

    def predict_image(self, img_bgr: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        h, w = img_bgr.shape[:2]

        now_ms = int(time.time() * 1000)
        if now_ms <= self._last_ts_ms:
            now_ms = self._last_ts_ms + 1
        self._last_ts_ms = now_ms

        # 1. HPE Inference
        raw_result, latency_ms = self.detector.infer(img_bgr, timestamp_ms=now_ms)
        kps = self.detector.to_unified(raw_result, frame_width=w, frame_height=h)

        if kps is None:
            return "Bad", latency_ms, {"no_detection": True, "latency_ms": latency_ms}

        # 2. Compute posture metrics
        metrics: PostureMetrics = self.metrics_calc.compute_all(kps)

        # 3. Classify posture using hybrid classifier (uncalibrated)
        calib = CalibrationData()
        posture_state = self.classifier.classify(
            metrics=metrics,
            calibration=calib,
            kps=kps,
            timestamp_s=time.time(),
        )

        pred_status = posture_state.status
        pred_label = posture_status_to_label(pred_status)

        details = {
            "no_detection": False,
            "latency_ms": latency_ms,
            "pred_status": pred_status.value,
            "neck_cva_deg": round(metrics.neck_cva_deg, 2),
            "trunk_angle_deg": round(metrics.trunk_angle_deg, 2),
            "shoulder_tilt_deg": round(metrics.shoulder_tilt_deg, 2),
            "kyphosis_risk_pct": round(posture_state.kyphosis_risk_pct, 1),
            "myopia_risk_pct": round(posture_state.myopia_risk_pct, 1),
            "is_valid": metrics.is_valid,
        }
        return pred_label, latency_ms, details

    def close(self):
        self.detector.close()


# ===========================================================================
# Classification Metrics
# ===========================================================================

def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    sub_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    classes = ["Good", "Bad"]
    cm = {c: {"Good": 0, "Bad": 0} for c in classes}
    for t, p in zip(y_true, y_pred):
        if t in cm and p in cm:
            cm[t][p] += 1

    TP = cm["Bad"]["Bad"]
    FP = cm["Good"]["Bad"]
    FN = cm["Bad"]["Good"]
    TN = cm["Good"]["Good"]
    total = TP + FP + FN + TN

    accuracy = (TP + TN) / max(total, 1)
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    specificity = TN / max(TN + FP, 1)

    per_class = {}
    for cls in classes:
        tp_c = cm[cls][cls]
        fp_c = sum(cm[other][cls] for other in classes if other != cls)
        fn_c = sum(cm[cls][other] for other in classes if other != cls)
        prec_c = tp_c / max(tp_c + fp_c, 1)
        rec_c = tp_c / max(tp_c + fn_c, 1)
        f1_c = 2 * prec_c * rec_c / max(prec_c + rec_c, 1e-9)
        support = sum(cm[cls].values())
        per_class[cls] = {
            "precision": round(prec_c, 4),
            "recall": round(rec_c, 4),
            "f1": round(f1_c, 4),
            "support": support,
        }

    # Sub-type breakdown (how well backward_lean, forward_lean, good_posture are recognized)
    subtype_breakdown = {}
    if sub_types:
        st_counts = defaultdict(lambda: {"total": 0, "pred_good": 0, "pred_bad": 0})
        for st, p in zip(sub_types, y_pred):
            st_counts[st]["total"] += 1
            if p == "Good":
                st_counts[st]["pred_good"] += 1
            else:
                st_counts[st]["pred_bad"] += 1

        for st, counts in st_counts.items():
            tot = counts["total"]
            bad_rate = counts["pred_bad"] / max(tot, 1)
            good_rate = counts["pred_good"] / max(tot, 1)
            subtype_breakdown[st] = {
                "total": tot,
                "pred_bad": counts["pred_bad"],
                "pred_good": counts["pred_good"],
                "bad_detection_rate": round(bad_rate, 4),
                "good_detection_rate": round(good_rate, 4),
            }

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "specificity": round(specificity, 4),
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "confusion_matrix": cm,
        "per_class": per_class,
        "subtype_breakdown": subtype_breakdown,
    }


# ===========================================================================
# Evaluation Engine
# ===========================================================================

class EvaluationEngine:
    def __init__(self, predictor: PosturePredictor):
        self.predictor = predictor

    def evaluate_split(
        self,
        dataset: MulticlassPostureDataset,
        split_name: str,
        verbose: bool = False,
        max_images: Optional[int] = None,
    ) -> Dict[str, Any]:
        y_true, y_pred, sub_types = [], [], []
        latencies = []
        no_detect_count = 0
        per_image_results = []

        total = len(dataset)
        if max_images:
            total = min(total, max_images)
        log.info(f"  [{split_name}] Evaluating {total} images...")
        t_split_start = time.perf_counter()

        for idx, (img_path, gt_label, sub_type) in enumerate(dataset.items(), 1):
            if max_images and idx > max_images:
                break

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                log.warning(f"  Could not load image: {img_path}")
                continue

            pred_label, lat_ms, details = self.predictor.predict_image(img_bgr)

            if details.get("no_detection"):
                no_detect_count += 1

            y_true.append(gt_label)
            y_pred.append(pred_label)
            sub_types.append(sub_type)
            latencies.append(lat_ms)

            per_image_results.append({
                "image": str(img_path.name),
                "gt": gt_label,
                "sub_type": sub_type,
                "pred": pred_label,
                "correct": gt_label == pred_label,
                **details,
            })

            if verbose or idx % 50 == 0 or idx == total:
                log.info(
                    f"    [{idx}/{total}] {img_path.name}: "
                    f"GT={gt_label} ({sub_type}), Pred={pred_label}, lat={lat_ms:.1f}ms"
                )

        split_wall_s = time.perf_counter() - t_split_start
        metrics = compute_classification_metrics(y_true, y_pred, sub_types)
        metrics["no_detect_count"] = no_detect_count
        metrics["no_detect_pct"] = round(100.0 * no_detect_count / max(len(y_true), 1), 1)
        metrics["n_images"] = len(y_true)
        metrics["split"] = split_name
        metrics["wall_time_s"] = round(split_wall_s, 2)

        if latencies:
            metrics["latency_mean_ms"] = round(float(np.mean(latencies)), 2)
            metrics["latency_std_ms"] = round(float(np.std(latencies)), 2)
            metrics["latency_p50_ms"] = round(float(np.percentile(latencies, 50)), 2)
            metrics["latency_p95_ms"] = round(float(np.percentile(latencies, 95)), 2)
            metrics["latency_p99_ms"] = round(float(np.percentile(latencies, 99)), 2)
            metrics["latency_min_ms"] = round(float(np.min(latencies)), 2)
            metrics["latency_max_ms"] = round(float(np.max(latencies)), 2)
            metrics["throughput_fps"] = round(1000.0 / max(metrics["latency_mean_ms"], 0.001), 1)

        metrics["per_image_results"] = per_image_results
        return metrics


# ===========================================================================
# Report Generator
# ===========================================================================

def generate_markdown_report(
    all_results: Dict[str, Dict[str, Any]],
    dataset_stats: Dict[str, Dict[str, int]],
    subtype_stats: Dict[str, Dict[str, int]],
    run_timestamp: str,
) -> str:
    lines = []
    lines.append("# Báo Cáo Đánh Giá Mô Hình HPE -- Phân Loại Tư Thế Ngồi (Sitting Posture Multiclass)")
    lines.append("")
    lines.append(f"> **Thời gian chạy:** {run_timestamp}")
    lines.append(f"> **Bộ dữ liệu:** Sitting Posture v2 Multiclass (`Sitting Posture.v2i.multiclass`)")
    lines.append(f"> **Nhiệm vụ:** Phân loại tư thế nhị phân (Good Posture vs Bad Posture)")
    lines.append(f"> **Tập dữ liệu đánh giá:** train, valid (eval), test")
    lines.append(f"> **Lớp mục tiêu Positive:** `Bad` (Bad Posture Detection)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Dataset overview
    lines.append("## 1. Tổng Quan Bộ Dữ Liệu (Dataset Overview)")
    lines.append("")
    lines.append("Bộ dữ liệu phân loại đa lớp Roboflow được quy về bài toán nhị phân chuẩn:")
    lines.append("- **Good Posture:** Gốc `goodposture` / file `good_posture-*`.")
    lines.append("- **Bad Posture:** Gộp `backwardbadposture` (tư thế ngả sau) và `forwardbadposture` (tư thế chúi trước / gù lưng) thành một lớp **Bad** duy nhất.")
    lines.append("")
    lines.append("### Phân bố nhãn nhị phân (Good vs Bad)")
    lines.append("")
    lines.append("| Tập dữ liệu (Split) | Tổng số ảnh | Good Posture | Bad Posture | Tỷ lệ Bad (%) |")
    lines.append("|---------------------|-------------|--------------|-------------|---------------|")
    for split in SPLITS:
        stats = dataset_stats.get(split, {})
        total = sum(stats.values())
        good = stats.get("Good", 0)
        bad = stats.get("Bad", 0)
        bad_pct = f"{100*bad/max(total,1):.1f}%"
        lines.append(f"| {split.capitalize()} | {total} | {good} | {bad} | {bad_pct} |")
    lines.append("")

    lines.append("### Phân bố chi tiết theo loại tư thế gốc")
    lines.append("")
    lines.append("| Tập (Split) | Good Posture | Forward Lean (Bad) | Backward Lean (Bad) | Tổng |")
    lines.append("|-------------|--------------|-------------------|--------------------|------|")
    for split in SPLITS:
        st = subtype_stats.get(split, {})
        good_cnt = st.get("good_posture", 0)
        fwd_cnt = st.get("forward_lean", 0)
        back_cnt = st.get("backward_lean", 0)
        tot = good_cnt + fwd_cnt + back_cnt
        lines.append(f"| {split.capitalize()} | {good_cnt} | {fwd_cnt} | {back_cnt} | {tot} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Models overview
    lines.append("## 2. Danh Sách 5 Mô Hình HPE Nhẹ Nhất (Plug-and-Play)")
    lines.append("")
    lines.append("| # | Mô hình | Keypoints | Kích thước trọng số | Backend | Kích thước Input |")
    lines.append("|---|---------|-----------|----------------------|---------|-------------------|")
    for i, mk in enumerate(MODELS_TO_EVALUATE, 1):
        info = MODEL_INFO.get(mk, {})
        lines.append(
            f"| {i} | **{info.get('display_name', mk)}** "
            f"| {info.get('keypoints','?')} kps "
            f"| {info.get('weight_kb','?')} KB "
            f"| {info.get('backend','?')} "
            f"| {info.get('input_size','?')} |"
        )
    lines.append("")
    lines.append("> *Ghi chú:* Các mô hình nặng hơn như YOLO-m (`yolo26m_pose`) và YOLO-x (`yolo26x_pose`) được loại trừ theo yêu cầu.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-model detailed results
    lines.append("## 3. Kết Quả Chi Tiết Từng Mô Hình")
    lines.append("")

    for mk in MODELS_TO_EVALUATE:
        model_results = all_results.get(mk, {})
        info = MODEL_INFO.get(mk, {})
        display = info.get("display_name", mk)
        idx = MODELS_TO_EVALUATE.index(mk) + 1
        lines.append(f"### 3.{idx} {display}")
        lines.append("")

        lines.append("#### Chỉ Số Phân Loại (Classification Metrics)")
        lines.append("")
        lines.append("| Split | N | Accuracy | Precision (Bad) | Recall (Bad) | F1-Score | Specificity (Good) | TP | FP | FN | TN | Không nhận diện (No-Det) |")
        lines.append("|-------|---|----------|-----------------|--------------|----------|--------------------|----|----|----|----|--------------------------|")
        for split in SPLITS:
            r = model_results.get(split, {})
            if not r:
                lines.append(f"| {split.capitalize()} | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |")
                continue
            lines.append(
                f"| {split.capitalize()} "
                f"| {r.get('n_images','?')} "
                f"| {format_pct(r.get('accuracy',0))} "
                f"| {format_pct(r.get('precision',0))} "
                f"| {format_pct(r.get('recall',0))} "
                f"| {format_pct(r.get('f1',0))} "
                f"| {format_pct(r.get('specificity',0))} "
                f"| {r.get('TP',0)} "
                f"| {r.get('FP',0)} "
                f"| {r.get('FN',0)} "
                f"| {r.get('TN',0)} "
                f"| {r.get('no_detect_count',0)} ({r.get('no_detect_pct',0)}%) |"
            )
        lines.append("")

        lines.append("#### Độ Trễ Suy Luận (Inference Latency & FPS trên CPU)")
        lines.append("")
        lines.append("| Split | Mean (ms) | Std (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Min (ms) | Max (ms) | FPS |")
        lines.append("|-------|-----------|----------|----------|----------|----------|----------|----------|-----|")
        for split in SPLITS:
            r = model_results.get(split, {})
            if not r:
                lines.append(f"| {split.capitalize()} | -- | -- | -- | -- | -- | -- | -- | -- |")
                continue
            lines.append(
                f"| {split.capitalize()} "
                f"| {r.get('latency_mean_ms','?')} "
                f"| {r.get('latency_std_ms','?')} "
                f"| {r.get('latency_p50_ms','?')} "
                f"| {r.get('latency_p95_ms','?')} "
                f"| {r.get('latency_p99_ms','?')} "
                f"| {r.get('latency_min_ms','?')} "
                f"| {r.get('latency_max_ms','?')} "
                f"| {r.get('throughput_fps','?')} |"
            )
        lines.append("")

        # Per-class breakdown and confusion matrix for valid & test
        for split in ["valid", "test"]:
            r = model_results.get(split, {})
            if not r or "per_class" not in r:
                continue
            lines.append(f"#### Chi Tiết Theo Lớp -- Tập {split.capitalize()}")
            lines.append("")
            lines.append("| Lớp (Class) | Precision | Recall | F1-Score | Support |")
            lines.append("|-------------|-----------|--------|----------|---------|")
            for cls in ["Good", "Bad"]:
                pc = r["per_class"].get(cls, {})
                lines.append(
                    f"| {cls} "
                    f"| {format_pct(pc.get('precision',0))} "
                    f"| {format_pct(pc.get('recall',0))} "
                    f"| {format_pct(pc.get('f1',0))} "
                    f"| {pc.get('support',0)} |"
                )
            lines.append("")

            # Subtype breakdown table
            sb = r.get("subtype_breakdown", {})
            if sb:
                lines.append(f"**Độ nhạy theo từng dạng tư thế thực tế ({split.capitalize()}):**")
                lines.append("")
                lines.append("| Dạng tư thế gốc | Tổng ảnh | Đoán là Bad (Tư thế xấu) | Đoán là Good (Tư thế tốt) | Tỷ lệ nhận diện đúng |")
                lines.append("|-----------------|----------|--------------------------|---------------------------|----------------------|")
                for st_key, st_name in [("forward_lean", "Forward Lean (Bad)"), ("backward_lean", "Backward Lean (Bad)"), ("good_posture", "Good Posture (Good)")]:
                    if st_key in sb:
                        d = sb[st_key]
                        if "Bad" in st_name:
                            correct_rate = format_pct(d["bad_detection_rate"])
                        else:
                            correct_rate = format_pct(d["good_detection_rate"])
                        lines.append(f"| {st_name} | {d['total']} | {d['pred_bad']} | {d['pred_good']} | **{correct_rate}** |")
                lines.append("")

            cm = r.get("confusion_matrix", {})
            if cm:
                lines.append(f"**Confusion Matrix ({split.capitalize()}):**")
                lines.append("```")
                lines.append("                       Predicted")
                lines.append("                     Good     Bad")
                lines.append(f"Actual   Good  [   {cm.get('Good',{}).get('Good',0):5d}    {cm.get('Good',{}).get('Bad',0):5d}   ]")
                lines.append(f"         Bad   [   {cm.get('Bad',{}).get('Good',0):5d}    {cm.get('Bad',{}).get('Bad',0):5d}   ]")
                lines.append("```")
                lines.append("")

    lines.append("---")
    lines.append("")

    # Cross-model comparison
    lines.append("## 4. So Sánh Tổng Thể Các Mô Hình (Cross-Model Comparison)")
    lines.append("")

    for split in SPLITS:
        idx = SPLITS.index(split) + 1
        lines.append(f"### 4.{idx} Tập {split.capitalize()} -- Toàn Bộ 5 Mô Hình")
        lines.append("")
        lines.append("| Mô hình | Accuracy | Precision (Bad) | Recall (Bad) | F1-Score | Specificity | Mean Latency (ms) | FPS (CPU) |")
        lines.append("|---------|----------|-----------------|--------------|----------|-------------|-------------------|-----------|")
        for mk in MODELS_TO_EVALUATE:
            r = all_results.get(mk, {}).get(split, {})
            info = MODEL_INFO.get(mk, {})
            if not r:
                lines.append(f"| {info.get('display_name',mk)} | -- | -- | -- | -- | -- | -- | -- |")
                continue
            lines.append(
                f"| {info.get('display_name',mk)} "
                f"| {format_pct(r.get('accuracy',0))} "
                f"| {format_pct(r.get('precision',0))} "
                f"| {format_pct(r.get('recall',0))} "
                f"| {format_pct(r.get('f1',0))} "
                f"| {format_pct(r.get('specificity',0))} "
                f"| {r.get('latency_mean_ms','?')} "
                f"| {r.get('throughput_fps','?')} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("")

    # Rankings
    lines.append("## 5. Bảng Xếp Hạng Đánh Giá (Summary Rankings)")
    lines.append("")

    for eval_split in ["valid", "test"]:
        lines.append(f"### 5.{1 if eval_split == 'valid' else 2} Xếp hạng trên tập {eval_split.capitalize()} (theo F1-Score)")
        lines.append("")
        ranking_data = []
        for mk in MODELS_TO_EVALUATE:
            r = all_results.get(mk, {}).get(eval_split, {})
            if not r:
                continue
            info = MODEL_INFO.get(mk, {})
            ranking_data.append({
                "model_key": mk,
                "display_name": info.get("display_name", mk),
                "f1": r.get("f1", 0),
                "accuracy": r.get("accuracy", 0),
                "recall": r.get("recall", 0),
                "precision": r.get("precision", 0),
                "latency_ms": r.get("latency_mean_ms", 9999),
                "throughput_fps": r.get("throughput_fps", 0),
            })
        ranking_data.sort(key=lambda x: (-x["f1"], -x["accuracy"]))

        lines.append("| Hạng | Mô hình | F1-Score | Accuracy | Recall (Bad) | Precision (Bad) | Độ trễ (ms) | FPS (CPU) |")
        lines.append("|------|---------|----------|----------|--------------|-----------------|-------------|-----------|")
        for rank, rd in enumerate(ranking_data, 1):
            lines.append(
                f"| {rank} | **{rd['display_name']}** "
                f"| **{format_pct(rd['f1'])}** "
                f"| {format_pct(rd['accuracy'])} "
                f"| {format_pct(rd['recall'])} "
                f"| {format_pct(rd['precision'])} "
                f"| {rd['latency_ms']} "
                f"| {rd['throughput_fps']} |"
            )
        lines.append("")

        if ranking_data:
            best = ranking_data[0]
            lines.append(f"> 🏆 **Mô hình tốt nhất trên tập {eval_split.capitalize()}:** **{best['display_name']}** (F1 = {format_pct(best['f1'])}, Accuracy = {format_pct(best['accuracy'])})")
            speed_rank = sorted(ranking_data, key=lambda x: x["latency_ms"])
            fastest = speed_rank[0]
            lines.append(f"> ⚡ **Mô hình nhanh nhất:** **{fastest['display_name']}** ({fastest['latency_ms']} ms/ảnh ~ {fastest['throughput_fps']} FPS)")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Methodology
    lines.append("## 6. Phương Pháp Đánh Giá & Nguyên Lý Kỹ Thuật (Methodology)")
    lines.append("")
    lines.append("### Pipeline Xử Lý Hoàn Chỉnh")
    lines.append("```")
    lines.append("Ảnh đầu vào (BGR) -> Mô hình HPE (MediaPipe / MoveNet / YOLO)")
    lines.append("                 -> Trích xuất UnifiedKeypoints chuẩn hóa")
    lines.append("                 -> PostureMetricsCalculator (Góc nghiêng CVA, góc thân, độ lệch vai)")
    lines.append("                 -> PostureClassifier (Chế độ Hybrid: Rule-based + ML)")
    lines.append("                 -> Nhãn dự đoán: Good | Bad")
    lines.append("                 -> Đối chiếu với Ground Truth -> Tính toán ma trận lỗi & chỉ số")
    lines.append("```")
    lines.append("")
    lines.append("### Quy tắc gộp lớp theo yêu cầu:")
    lines.append("- Nhãn **Good Posture** (`Good`): Các trường hợp tư thế ngồi đúng, thẳng lưng, công thái học tốt.")
    lines.append("- Nhãn **Bad Posture** (`Bad`): Gộp tất cả các dạng sai lệch tư thế bao gồm cả chúi đầu/nghiêng trước (`forwardbadposture`) và ngửa sau (`backwardbadposture`).")
    lines.append("- Safe-fail khi không phát hiện người (`no_detection`): Mặc định gán cảnh báo `Bad` để bảo đảm an toàn công thái học.")
    lines.append("")

    return "\n".join(lines)


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate 5 lightest HPE models on Sitting Posture multiclass dataset",
    )
    parser.add_argument(
        "--models", nargs="+", default=MODELS_TO_EVALUATE,
        choices=MODELS_TO_EVALUATE,
        help="HPE model keys to evaluate (default: all 5)",
    )
    parser.add_argument(
        "--splits", nargs="+", default=SPLITS,
        help="Dataset splits to evaluate (train, valid/eval, test)",
    )
    parser.add_argument(
        "--dataset-root", type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help="Path to Sitting Posture multiclass dataset root",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_RESULTS_DIR),
        help="Directory to save reports",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Inference device ('cpu' or 'cuda')",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed per-image results to console",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run on first 5 images per split only (for quick testing)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize split names: map 'eval' -> 'valid'
    normalized_splits = []
    for s in args.splits:
        norm = "valid" if s.lower() in ("eval", "val", "validation") else s.lower()
        if norm in SPLITS and norm not in normalized_splits:
            normalized_splits.append(norm)

    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.info("=" * 70)
    log.info("SpineVision EdgeAI -- Multiclass Sitting Posture Evaluation")
    log.info(f"  Run at : {run_timestamp}")
    log.info(f"  Models : {args.models}")
    log.info(f"  Splits : {normalized_splits}")
    log.info(f"  Dataset: {dataset_root}")
    log.info(f"  Output : {output_dir}")
    log.info("=" * 70)

    # Load datasets
    datasets: Dict[str, MulticlassPostureDataset] = {}
    dataset_stats: Dict[str, Dict[str, int]] = {}
    subtype_stats: Dict[str, Dict[str, int]] = {}

    for split in normalized_splits:
        split_dir = dataset_root / split
        if not split_dir.exists():
            log.warning(f"Split directory not found: {split_dir} -- skipping.")
            continue
        try:
            ds = MulticlassPostureDataset(split_dir)
            datasets[split] = ds
            dataset_stats[split] = ds.class_distribution()
            subtype_stats[split] = ds.subtype_distribution()
            log.info(
                f"  Loaded {split}: {len(ds)} images -- Binary: {dataset_stats[split]} | Subtypes: {subtype_stats[split]}"
            )
        except Exception as e:
            log.error(f"  Failed to load {split}: {e}")

    if not datasets:
        log.error("No datasets loaded. Exiting.")
        sys.exit(1)

    max_images = 5 if args.dry_run else None

    # Run evaluations
    all_results: Dict[str, Dict[str, Any]] = {}

    for model_key in args.models:
        log.info(f"\n{'='*60}")
        log.info(f"Evaluating: {MODEL_INFO.get(model_key,{}).get('display_name', model_key)}")
        log.info(f"{'='*60}")

        try:
            predictor = PosturePredictor(model_key=model_key, device=args.device)
        except Exception as e:
            log.error(f"Failed to load model '{model_key}': {e}")
            traceback.print_exc()
            all_results[model_key] = {}
            continue

        engine = EvaluationEngine(predictor)
        model_results = {}

        for split, dataset in datasets.items():
            log.info(f"\n  -- Split: {split.upper()} --")
            try:
                result = engine.evaluate_split(
                    dataset, split,
                    verbose=args.verbose,
                    max_images=max_images,
                )
                model_results[split] = result
                log.info(
                    f"  OK {split}: Acc={format_pct(result['accuracy'])} "
                    f"F1={format_pct(result['f1'])} "
                    f"Recall={format_pct(result['recall'])} "
                    f"Prec={format_pct(result['precision'])} "
                    f"Lat={result.get('latency_mean_ms','?')}ms"
                )
            except Exception as e:
                log.error(f"  Error evaluating {split}: {e}")
                traceback.print_exc()

        predictor.close()
        all_results[model_key] = model_results

    # Generate reports
    log.info(f"\n{'='*60}")
    log.info("Generating reports...")

    report_md = generate_markdown_report(all_results, dataset_stats, subtype_stats, run_timestamp)
    report_path = output_dir / "hpe_eval_multiclass_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    log.info(f"  Markdown report: {report_path}")

    def strip_per_image(results):
        stripped = {}
        for mk, splits_data in results.items():
            stripped[mk] = {}
            for split, m in splits_data.items():
                stripped[mk][split] = {k: v for k, v in m.items() if k != "per_image_results"}
        return stripped

    json_summary_path = output_dir / "hpe_eval_multiclass_summary.json"
    with json_summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_timestamp": run_timestamp,
                "dataset_stats": dataset_stats,
                "subtype_stats": subtype_stats,
                "results": strip_per_image(all_results),
            },
            f, indent=2, ensure_ascii=False,
        )
    log.info(f"  Summary JSON: {json_summary_path}")

    json_full_path = output_dir / "hpe_eval_multiclass_full.json"
    with json_full_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_timestamp": run_timestamp,
                "dataset_stats": dataset_stats,
                "subtype_stats": subtype_stats,
                "results": all_results,
            },
            f, indent=2, ensure_ascii=False, default=str,
        )
    log.info(f"  Full JSON: {json_full_path}")

    # Console summary
    log.info(f"\n{'='*70}")
    log.info("EVALUATION COMPLETE -- Summary (Valid & Test splits)")
    log.info(f"{'='*70}")
    for split in ["valid", "test"]:
        if split not in datasets:
            continue
        log.info(f"\n--- Split: {split.upper()} ---")
        log.info(f"{'Model':<30} {'Accuracy':>10} {'F1':>8} {'Recall':>8} {'Prec':>8} {'Lat(ms)':>10}")
        log.info(f"{'-'*76}")
        for mk in args.models:
            r = all_results.get(mk, {}).get(split, {})
            name = MODEL_INFO.get(mk, {}).get("display_name", mk)
            if not r:
                log.info(f"{name:<30}  N/A")
                continue
            log.info(
                f"{name:<30} "
                f"{format_pct(r.get('accuracy',0)):>10} "
                f"{format_pct(r.get('f1',0)):>8} "
                f"{format_pct(r.get('recall',0)):>8} "
                f"{format_pct(r.get('precision',0)):>8} "
                f"{r.get('latency_mean_ms','?'):>10}"
            )
    log.info(f"{'='*70}")
    log.info(f"\nAll reports written to: {output_dir}/")


if __name__ == "__main__":
    main()
