#!/usr/bin/env python3
"""
HPE Model Evaluation on Posture Classification Dataset
=======================================================
SpineVision EdgeAI — Plug-and-Play HPE Architecture

Evaluates the 5 lightest HPE models on the sitting posture dataset
(train / valid / test splits) using the project's existing posture
classification pipeline.

Models evaluated (lightest 5, excluding yolo26m and yolo26x):
  1. MediaPipe Pose LITE   (33 3D landmarks, ~5.8 MB)
  2. MediaPipe Pose FULL   (33 3D landmarks, ~9.4 MB)
  3. MoveNet Lightning     (17 COCO keypoints, 192x192, ~4.6 MB)
  4. MoveNet Thunder       (17 COCO keypoints, 256x256, ~12.0 MB)
  5. YOLO26n-Pose (Nano)   (17 COCO keypoints, ~6.3 MB)

Output:
  evaluation/results/hpe_eval_report.md    -- Full markdown report
  evaluation/results/hpe_eval_summary.json -- Aggregated metrics
  evaluation/results/hpe_eval_full.json    -- Per-image raw data

Usage:
  python evaluation/evaluate_hpe_classification.py [--models ...] [--splits ...]
  python evaluation/evaluate_hpe_classification.py --dry-run
"""

import os
import sys
import json
import time
import argparse
import logging
import datetime
import traceback
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

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
DATASET_ROOT = Path(
    "/media/linhcbs/DATA/NCKH/NCKH HÈ/DATA/"
    "sitting posture.v4-sitting_posture_4keypoint.coco"
)

RESULTS_DIR = SCRIPT_DIR / "results"

MODELS_TO_EVALUATE = [
    "mediapipe_pose_lite",
    "mediapipe_pose_full",
    "movenet_lightning",
    "movenet_thunder",
    "yolo26n_pose",
]

SPLITS = ["train", "valid", "test"]

LABEL_MAP = {"Good": "Good", "Bad": "Bad"}

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
log = logging.getLogger("hpe_eval")


# ===========================================================================
# Helper functions
# ===========================================================================

def posture_status_to_label(status: PostureStatus) -> str:
    if status == PostureStatus.GOOD:
        return "Good"
    return "Bad"


def format_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


# ===========================================================================
# COCO Dataset Loader
# ===========================================================================

class COCOPostureDataset:
    """Loads COCO-format posture annotations from a single split directory."""

    def __init__(self, split_dir: Path):
        ann_path = split_dir / "_annotations.coco.json"
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotations not found: {ann_path}")

        with ann_path.open() as f:
            data = json.load(f)

        self.images: Dict[int, Dict] = {img["id"]: img for img in data["images"]}
        self.categories: Dict[int, str] = {
            cat["id"]: cat["name"]
            for cat in data["categories"]
            if cat["id"] != 0
        }
        self.annotations: List[Dict] = data["annotations"]
        self._img_to_anns: Dict[int, List[Dict]] = defaultdict(list)
        for ann in self.annotations:
            self._img_to_anns[ann["image_id"]].append(ann)
        self.split_dir = split_dir

    def __len__(self) -> int:
        return len(self.images)

    def items(self):
        for img_id, img_info in self.images.items():
            img_path = self.split_dir / img_info["file_name"]
            anns = self._img_to_anns.get(img_id, [])
            if not anns:
                continue
            ann = anns[0]
            cat_id = ann["category_id"]
            gt_label = LABEL_MAP.get(self.categories.get(cat_id, ""), None)
            if gt_label is None:
                continue
            kps_flat = ann.get("keypoints", [])
            gt_kps = []
            for i in range(0, len(kps_flat), 3):
                gt_kps.append((float(kps_flat[i]), float(kps_flat[i + 1]), int(kps_flat[i + 2])))
            bbox = ann.get("bbox", [0, 0, 0, 0])
            yield img_path, gt_label, gt_kps, bbox

    def class_distribution(self) -> Dict[str, int]:
        counts: Counter = Counter()
        for ann in self.annotations:
            cat_id = ann["category_id"]
            name = self.categories.get(cat_id, "unknown")
            counts[name] += 1
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
        calib = CalibrationData()  # uncalibrated — no personal baseline
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

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "specificity": round(specificity, 4),
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "confusion_matrix": cm,
        "per_class": per_class,
    }


# ===========================================================================
# Evaluation Engine
# ===========================================================================

class EvaluationEngine:
    def __init__(self, predictor: PosturePredictor):
        self.predictor = predictor

    def evaluate_split(
        self,
        dataset: COCOPostureDataset,
        split_name: str,
        verbose: bool = False,
        max_images: Optional[int] = None,
    ) -> Dict[str, Any]:
        y_true, y_pred = [], []
        latencies = []
        no_detect_count = 0
        per_image_results = []

        total = len(dataset)
        if max_images:
            total = min(total, max_images)
        log.info(f"  [{split_name}] Evaluating {total} images...")
        t_split_start = time.perf_counter()

        for idx, (img_path, gt_label, gt_kps, bbox) in enumerate(dataset.items(), 1):
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
            latencies.append(lat_ms)

            per_image_results.append({
                "image": str(img_path.name),
                "gt": gt_label,
                "pred": pred_label,
                "correct": gt_label == pred_label,
                **details,
            })

            if verbose or idx % 50 == 0:
                log.info(
                    f"    [{idx}/{total}] {img_path.name}: "
                    f"GT={gt_label}, Pred={pred_label}, lat={lat_ms:.1f}ms"
                )

        split_wall_s = time.perf_counter() - t_split_start
        metrics = compute_classification_metrics(y_true, y_pred)
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
    run_timestamp: str,
) -> str:
    lines = []
    lines.append("# HPE Model Evaluation Report -- Posture Classification")
    lines.append("")
    lines.append(f"> **Generated:** {run_timestamp}")
    lines.append(f"> **Dataset:** Sitting Posture v4 (4-keypoint COCO format)")
    lines.append(f"> **Task:** Binary Posture Classification (Good / Bad)")
    lines.append(f"> **Splits evaluated:** train, valid, test")
    lines.append(f"> **Positive class:** `Bad` (bad posture detection target)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Dataset overview
    lines.append("## 1. Dataset Overview")
    lines.append("")
    lines.append("| Split | Total Images | Good | Bad | Bad% |")
    lines.append("|-------|-------------|------|-----|------|")
    for split in SPLITS:
        stats = dataset_stats.get(split, {})
        total = sum(stats.values())
        good = stats.get("Good", 0)
        bad = stats.get("Bad", 0)
        bad_pct = f"{100*bad/max(total,1):.1f}%"
        lines.append(f"| {split.capitalize()} | {total} | {good} | {bad} | {bad_pct} |")
    lines.append("")
    lines.append("> **Keypoints:** `bottom`, `shoulder`, `head`, `back` (4 custom anatomical points)")
    lines.append("> **Labels:** category_id=1 -> `Bad`, category_id=2 -> `Good`")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Models overview
    lines.append("## 2. Models Under Evaluation")
    lines.append("")
    lines.append("| # | Model | Keypoints | Weight Size | Backend | Input Size |")
    lines.append("|---|-------|-----------|-------------|---------|------------|")
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
    lines.append("> YOLO26m-Pose and YOLO26x-Pose are excluded per evaluation criteria.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-model detailed results
    lines.append("## 3. Detailed Results by Model")
    lines.append("")

    for mk in MODELS_TO_EVALUATE:
        model_results = all_results.get(mk, {})
        info = MODEL_INFO.get(mk, {})
        display = info.get("display_name", mk)
        idx = MODELS_TO_EVALUATE.index(mk) + 1
        lines.append(f"### 3.{idx} {display}")
        lines.append("")

        lines.append("#### Classification Metrics")
        lines.append("")
        lines.append("| Split | N | Accuracy | Precision | Recall | F1 | Specificity | TP | FP | FN | TN | No-Detection |")
        lines.append("|-------|---|----------|-----------|--------|----|-------------|----|----|----|----|----|")
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

        lines.append("#### Inference Latency (ms)")
        lines.append("")
        lines.append("| Split | Mean | Std | P50 | P95 | P99 | Min | Max | FPS |")
        lines.append("|-------|------|-----|-----|-----|-----|-----|-----|-----|")
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

        for split in SPLITS:
            r = model_results.get(split, {})
            if not r or "per_class" not in r:
                continue
            lines.append(f"#### Per-Class Metrics -- {split.capitalize()}")
            lines.append("")
            lines.append("| Class | Precision | Recall | F1 | Support |")
            lines.append("|-------|-----------|--------|----|---------|")
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

            cm = r.get("confusion_matrix", {})
            if cm:
                lines.append(f"#### Confusion Matrix -- {split.capitalize()}")
                lines.append("")
                lines.append("```")
                lines.append("                 Predicted")
                lines.append("                 Good    Bad")
                lines.append(f"Actual  Good  [ {cm.get('Good',{}).get('Good',0):5d}  {cm.get('Good',{}).get('Bad',0):5d} ]")
                lines.append(f"        Bad   [ {cm.get('Bad',{}).get('Good',0):5d}  {cm.get('Bad',{}).get('Bad',0):5d} ]")
                lines.append("```")
                lines.append("")

    lines.append("---")
    lines.append("")

    # Cross-model comparison
    lines.append("## 4. Cross-Model Comparison")
    lines.append("")

    for split in SPLITS:
        idx = SPLITS.index(split) + 1
        lines.append(f"### 4.{idx} {split.capitalize()} Split -- All Models")
        lines.append("")
        lines.append("| Model | Accuracy | Precision | Recall | F1 | Specificity | Latency (ms) | FPS |")
        lines.append("|-------|----------|-----------|--------|----|-------------|--------------|-----|")
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
    lines.append("## 5. Summary Rankings (Validation Split)")
    lines.append("")
    lines.append("Models ranked by F1-score on the `valid` split:")
    lines.append("")

    ranking_data = []
    for mk in MODELS_TO_EVALUATE:
        r = all_results.get(mk, {}).get("valid", {})
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

    lines.append("| Rank | Model | F1 | Accuracy | Recall | Precision | Latency (ms) | FPS |")
    lines.append("|------|-------|----|----------|--------|-----------|--------------|-----|")
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

    speed_rank = sorted(ranking_data, key=lambda x: x["latency_ms"])
    if speed_rank:
        fastest = speed_rank[0]
        lines.append(f"> Fastest model: {fastest['display_name']} @ {fastest['latency_ms']} ms/image")
    if ranking_data:
        best = ranking_data[0]
        lines.append(f"> Best F1 on validation: {best['display_name']} -- F1={format_pct(best['f1'])}, Accuracy={format_pct(best['accuracy'])}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Methodology
    lines.append("## 6. Methodology & Notes")
    lines.append("")
    lines.append("### Pipeline")
    lines.append("```")
    lines.append("Image (BGR) -> HPE Inference -> UnifiedKeypoints")
    lines.append("           -> PostureMetricsCalculator (CVA, trunk angle, shoulder tilt, ...)")
    lines.append("           -> LightweightPostureML.extract_features (8D feature vector)")
    lines.append("           -> Hybrid Classifier (rule-based + softmax ML)")
    lines.append("           -> Predicted Label: Good | Bad")
    lines.append("           -> Compare with GT COCO label -> Classification Metrics")
    lines.append("```")
    lines.append("")
    lines.append("### Label Mapping")
    lines.append("- PostureStatus.GOOD -> **Good**")
    lines.append("- All other statuses (SLOUCHED, FORWARD_HEAD, LEANING_*, TOO_CLOSE, ...) -> **Bad**")
    lines.append("")
    lines.append("### Positive Class")
    lines.append("- Positive class = Bad (bad posture detection target)")
    lines.append("- Precision: of all predicted Bad, how many truly are Bad")
    lines.append("- Recall: of all truly Bad images, how many were correctly flagged")
    lines.append("")
    lines.append("### No-Detection Cases")
    lines.append("- When HPE fails to detect any person, the prediction defaults to Bad (safe-fail)")
    lines.append("")
    lines.append("### Classifier Mode")
    lines.append("- Hybrid mode: rule-based checks applied first, then lightweight ML softmax")
    lines.append("- Uncalibrated (no personal baseline) for fair dataset-wide evaluation")
    lines.append("")
    lines.append("### Latency")
    lines.append("- Latency = HPE inference only (excludes image I/O and metric computation)")
    lines.append("- Measured on CPU (no CUDA)")
    lines.append("")

    return "\n".join(lines)


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate lightweight HPE models on posture classification dataset",
    )
    parser.add_argument(
        "--models", nargs="+", default=MODELS_TO_EVALUATE,
        choices=MODELS_TO_EVALUATE,
        help="HPE model keys to evaluate (default: all 5)",
    )
    parser.add_argument(
        "--splits", nargs="+", default=SPLITS, choices=SPLITS,
        help="Dataset splits to evaluate (default: train valid test)",
    )
    parser.add_argument(
        "--dataset-root", type=str,
        default=str(DATASET_ROOT),
        help="Path to COCO dataset root directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(RESULTS_DIR),
        help="Directory to save reports",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Inference device for YOLO ('cpu' or 'cuda')",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-image results to console",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run on first 5 images per split only (quick test)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.info("=" * 70)
    log.info("SpineVision EdgeAI -- HPE Posture Classification Evaluation")
    log.info(f"  Run at : {run_timestamp}")
    log.info(f"  Models : {args.models}")
    log.info(f"  Splits : {args.splits}")
    log.info(f"  Dataset: {dataset_root}")
    log.info(f"  Output : {output_dir}")
    log.info("=" * 70)

    # Load datasets
    datasets: Dict[str, COCOPostureDataset] = {}
    dataset_stats: Dict[str, Dict[str, int]] = {}
    for split in args.splits:
        split_dir = dataset_root / split
        if not split_dir.exists():
            log.warning(f"Split directory not found: {split_dir} -- skipping.")
            continue
        try:
            ds = COCOPostureDataset(split_dir)
            datasets[split] = ds
            dataset_stats[split] = ds.class_distribution()
            log.info(f"  Loaded {split}: {len(ds)} images -- {dataset_stats[split]}")
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

    report_md = generate_markdown_report(all_results, dataset_stats, run_timestamp)
    report_path = output_dir / "hpe_eval_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    log.info(f"  Markdown report: {report_path}")

    def strip_per_image(results):
        stripped = {}
        for mk, splits_data in results.items():
            stripped[mk] = {}
            for split, m in splits_data.items():
                stripped[mk][split] = {k: v for k, v in m.items() if k != "per_image_results"}
        return stripped

    json_summary_path = output_dir / "hpe_eval_summary.json"
    with json_summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"run_timestamp": run_timestamp, "dataset_stats": dataset_stats,
             "results": strip_per_image(all_results)},
            f, indent=2, ensure_ascii=False,
        )
    log.info(f"  Summary JSON: {json_summary_path}")

    json_full_path = output_dir / "hpe_eval_full.json"
    with json_full_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"run_timestamp": run_timestamp, "dataset_stats": dataset_stats,
             "results": all_results},
            f, indent=2, ensure_ascii=False, default=str,
        )
    log.info(f"  Full JSON: {json_full_path}")

    # Console summary
    log.info(f"\n{'='*70}")
    log.info("EVALUATION COMPLETE -- Summary (valid split)")
    log.info(f"{'='*70}")
    log.info(f"{'Model':<30} {'Accuracy':>10} {'F1':>8} {'Recall':>8} {'Lat(ms)':>10}")
    log.info(f"{'-'*70}")
    for mk in args.models:
        r = all_results.get(mk, {}).get("valid", {})
        name = MODEL_INFO.get(mk, {}).get("display_name", mk)
        if not r:
            log.info(f"{name:<30}  N/A")
            continue
        log.info(
            f"{name:<30} "
            f"{format_pct(r.get('accuracy',0)):>10} "
            f"{format_pct(r.get('f1',0)):>8} "
            f"{format_pct(r.get('recall',0)):>8} "
            f"{r.get('latency_mean_ms','?'):>10}"
        )
    log.info(f"{'='*70}")
    log.info(f"\nReports written to: {output_dir}/")


if __name__ == "__main__":
    main()
