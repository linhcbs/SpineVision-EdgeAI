#!/usr/bin/env python3
"""
Real-Time Posture Analysis Webcam Demo.
SpineVision EdgeAI — Plug-and-Play HPE Architecture

Features:
  - Plug-and-Play Pose Estimation (Supports ALL 7 HPE Models):
      1. MediaPipe Pose LITE   (33 3D landmarks, ultra-fast)
      2. MediaPipe Pose FULL   (33 3D landmarks, high accuracy)
      3. MoveNet Lightning     (17 COCO keypoints, 192x192, quantized)
      4. MoveNet Thunder       (17 COCO keypoints, 256x256, quantized)
      5. YOLO26n-pose (Nano)   (17 COCO keypoints, multi-person edge)
      6. YOLO26m-pose (Medium) (17 COCO keypoints, balanced multi-person)
      7. YOLO26x-pose (XLarge) (17 COCO keypoints, maximum accuracy)
  - Biometric Ergonomic Angle Calculation:
      * Craniovertebral Angle (CVA) — Forward Head / Tech Neck
      * Shoulder Tilt Angle — Lateral Spinal Asymmetry (Left / Right)
      * Trunk / Spine Slump Angle — Slouching / Kyphosis
      * Eye-to-Screen Distance — Myopia Risk Estimation
  - Hybrid Posture Classifier (Rule-based + Lightweight ML):
      * Posture States: Upright, Leaning Left, Leaning Right, Slouched, Tech Neck, Screen Near
      * Instantaneous & Prolonged Risk Diagnosis for Kyphosis and Myopia
  - Temporal Smoothing: One-Euro Filter (eliminates micro-jitter)
  - Auto-Calibration: Personalized "Good Posture" Baseline (3 seconds)
  - Multi-Screen & Workspace Surface Detection (YOLO11n + Geometric heuristics)
  - Comprehensive Visual HUD (Heads-Up Display) with Risk Progress Bars

Controls:
  [C] : Start 3-second Auto-Calibration (sit upright & look at screen)
  [R] : Reset Calibration (return to absolute clinical thresholds)
  [E] : Reset Prolonged Risk Exposure (simulate taking a break)
  [M] : Toggle Classifier Mode (Hybrid <-> Rule-Based <-> ML)
  [T] : Switch/Cycle HPE Model on-the-fly (MediaPipe / MoveNet / YOLO)
  [S] : Toggle One-Euro Landmark Smoothing ON / OFF
  [W] : Toggle Workspace (Desk & Screen) Overlays ON / OFF
  [Q] / [ESC] : Quit
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

# Ensure posture_analyzer and benchmarks modules are discoverable
CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from posture_analyzer import (
    PostureAnalysisEngine,
    PostureState,
    PostureStatus,
    AlertLevel,
    create_hpe_detector,
    list_supported_hpe_models,
    SUPPORTED_HPE_MODELS,
    canonical_model_key,
)
from posture_analyzer.config import get_posture_config
from benchmarks.models.drawing_utils import open_camera

try:
    import psutil
    _PROCESS = psutil.Process()
except ImportError:
    _PROCESS = None


MODEL_CYCLE_LIST = [
    "mediapipe_pose_full",
    "mediapipe_pose_lite",
    "movenet_lightning",
    "movenet_thunder",
    "yolo26n_pose",
    "yolo26m_pose",
    "yolo26x_pose",
]


def parse_args():
    cfg = get_posture_config()
    cam_cfg = cfg.get("camera", {})
    class_cfg = cfg.get("classification", {})

    parser = argparse.ArgumentParser(
        description="SpineVision Real-Time Posture Analysis Demo (Plug-and-Play HPE Architecture)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", "--hpe-model",
        type=str,
        default="mediapipe_pose_full",
        help=(
            "Select HPE backbone model. Supported: "
            + ", ".join(list(SUPPORTED_HPE_MODELS.keys()))
            + " (or aliases: mp_lite, lightning, thunder, yolo_n, yolo_m, yolo_x)"
        ),
    )
    parser.add_argument("--model-path", type=str, default=None, help="Custom path to model weights file override")
    parser.add_argument("--camera", type=int, default=cam_cfg.get("camera_index", 0), help="Camera index")
    parser.add_argument("--width", type=int, default=cam_cfg.get("width", 1280), help="Camera capture width")
    parser.add_argument("--height", type=int, default=cam_cfg.get("height", 720), help="Camera capture height")
    parser.add_argument("--no-smoothing", action="store_true", help="Disable One-Euro filter smoothing")
    parser.add_argument("--no-workspace", action="store_true", help="Disable workspace overlays")
    parser.add_argument("--sensitivity", type=float, default=class_cfg.get("default_sensitivity", 1.0), help="Classifier sensitivity")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device for YOLO ('cpu' or 'cuda')")
    parser.add_argument("--list-models", action="store_true", help="Print all 7 supported HPE models and exit")
    return parser.parse_args()


def print_models_table():
    """Print clean terminal table of all 7 supported HPE models."""
    print("=" * 80)
    print(" SPINEVISION EDGEAI — SUPPORTED HUMAN POSE ESTIMATION (HPE) MODELS")
    print("=" * 80)
    print(f"{'No.':<4} {'Model Key':<22} {'Keypoints':<11} {'Depth':<8} {'Description'}")
    print("-" * 80)
    for i, m in enumerate(list_supported_hpe_models(), 1):
        depth_str = "3D (z)" if m["has_depth"] else "2D"
        kps_str = f"{m['keypoints']} pts"
        print(f"{i:<4} {m['key']:<22} {kps_str:<11} {depth_str:<8} {m['description']}")
    print("=" * 80)


def main():
    cfg = get_posture_config()
    args = parse_args()

    if args.list_models:
        print_models_table()
        sys.exit(0)

    print("=" * 75)
    print(" SpineVision EdgeAI — Plug-and-Play Ergonomic Posture Analysis")
    print("=" * 75)
    print("Controls:")
    print("  [C] : Start Auto-Calibration (sit upright for ~3s)")
    print("  [R] : Reset Calibration (return to absolute thresholds)")
    print("  [E] : Reset Prolonged Risk Exposure (simulate taking a break)")
    print("  [M] : Toggle Classifier Mode (Hybrid / Rule-Based / ML)")
    print("  [T] : Switch/Cycle HPE Model (MediaPipe <-> MoveNet <-> YOLO)")
    print("  [S] : Toggle One-Euro Smoothing")
    print("  [W] : Toggle Workspace Surface & Screen Overlays")
    print("  [Q] : Quit")
    print("=" * 75)

    # 1. Initialize HPE Model via Universal Adapter Factory
    current_model_key = canonical_model_key(args.model)
    print(f"[HPE Init] Loading model: {current_model_key}...")
    try:
        detector = create_hpe_detector(
            model_name_or_key=current_model_key,
            weights_path=args.model_path,
            device=args.device,
        )
        print(f"[HPE Init] Successfully initialized: {detector.model_name}")
    except Exception as e:
        print(f"[Error] Failed to initialize HPE model '{args.model}': {e}")
        print("Falling back to MediaPipe Pose Full...")
        detector = create_hpe_detector("mediapipe_pose_full")

    # 2. Initialize Posture Analysis Engine
    engine = PostureAnalysisEngine(
        use_smoothing=not args.no_smoothing,
        one_euro_min_cutoff=1.0,
        one_euro_beta=0.005,
        use_relative_calibration=True,
        classifier_sensitivity=args.sensitivity,
        target_calibration_samples=90,  # ~3 seconds at 30 FPS
        calibration_distance_cm=60.0,
        enable_workspace_detection=not args.no_workspace,
    )

    # 3. Open Camera
    fallback_indices = cfg.get("camera", {}).get("fallback_indices", [1, 2, 3, 4])
    cap, cam_idx = open_camera(args.camera, width=args.width, height=args.height, fallback_indices=fallback_indices)
    if cap is None:
        print("[Error] Failed to open webcam. Exiting.")
        detector.close()
        sys.exit(1)

    window_name = "SpineVision EdgeAI - Posture Analysis"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    fps = 0.0
    frame_count = 0
    t_start = time.perf_counter()
    draw_workspace = not args.no_workspace
    last_timestamp_ms = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[Warning] Empty frame received. Retrying...")
                time.sleep(0.01)
                continue

            # Mirror view for natural interaction
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Ensure strictly monotonic timestamps for MediaPipe Video mode
            now_ms = int(time.time() * 1000)
            if now_ms <= last_timestamp_ms:
                now_ms = last_timestamp_ms + 1
            last_timestamp_ms = now_ms
            current_timestamp_ms = now_ms

            # 4. Universal HPE Pose Inference & Conversion
            raw_result, latency_ms = detector.infer(frame, timestamp_ms=current_timestamp_ms)
            kps = detector.to_unified(raw_result, frame_width=w, frame_height=h)

            # 5. Posture Analysis Pipeline
            if kps is not None:
                # Draw model skeleton
                detector.draw(frame, raw_result)

                # Process through posture analysis engine
                state = engine.process_keypoints(
                    kps,
                    timestamp_s=time.time(),
                    frame=frame,
                )

                # Render Ergonomic HUD & angle arcs
                engine.render_hud(
                    frame=frame,
                    state=state,
                    kps=kps,
                    draw_workspace=draw_workspace,
                )
            else:
                # No pose detected in current frame
                cv2.putText(
                    frame,
                    f"No person detected ({detector.model_name})",
                    (w // 2 - 180, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (80, 80, 240),
                    2,
                    cv2.LINE_AA,
                )

            # Calculate FPS
            frame_count += 1
            if frame_count % 15 == 0:
                now = time.perf_counter()
                fps = 15.0 / (now - t_start)
                t_start = now

            # Top-left info overlay with active HPE model & RAM consumption
            ram_mb = _PROCESS.memory_info().rss / (1024.0 * 1024.0) if _PROCESS is not None else 0.0
            cv2.putText(
                frame,
                f"HPE: {detector.model_name}",
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (220, 220, 100),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms | RAM: {ram_mb:.0f}MB",
                (15, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            # Show active status of features
            smoothing_str = "ON (One-Euro)" if engine.use_smoothing else "OFF"
            calib_str = "CALIBRATED" if engine.is_calibrated else "UNCALIBRATED"
            mode_str = engine.classifier.mode.upper()
            cv2.putText(
                frame,
                f"Filter: {smoothing_str} | Base: {calib_str} | Mode: {mode_str}",
                (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame)

            # Key controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('c') or key == ord('C'):
                print("[Action] Starting auto-calibration...")
                engine.start_calibration()
            elif key == ord('r') or key == ord('R'):
                print("[Action] Resetting calibration...")
                engine.reset_calibration()
            elif key == ord('e') or key == ord('E'):
                print("[Action] Resetting prolonged risk exposure counters (break taken)...")
                engine.classifier.reset_exposure()
            elif key == ord('m') or key == ord('M'):
                modes = ["hybrid", "rule_based", "ml"]
                cur_idx = modes.index(engine.classifier.mode) if engine.classifier.mode in modes else 0
                engine.classifier.mode = modes[(cur_idx + 1) % len(modes)]
                print(f"[Action] Posture Classifier mode switched to: {engine.classifier.mode.upper()}")
            elif key == ord('t') or key == ord('T'):
                # Cycle through the 7 HPE models on the fly!
                cur_idx = MODEL_CYCLE_LIST.index(current_model_key) if current_model_key in MODEL_CYCLE_LIST else 0
                next_key = MODEL_CYCLE_LIST[(cur_idx + 1) % len(MODEL_CYCLE_LIST)]
                print(f"\n[Action] Switching HPE model from {current_model_key} -> {next_key}...")
                try:
                    detector.close()
                    detector = create_hpe_detector(next_key, device=args.device)
                    current_model_key = next_key
                    print(f"[Action] Switched successfully to: {detector.model_name}")
                except Exception as ex:
                    print(f"[Error] Could not switch to {next_key}: {ex}")
            elif key == ord('s') or key == ord('S'):
                engine.use_smoothing = not engine.use_smoothing
                print(f"[Action] One-Euro Smoothing set to: {engine.use_smoothing}")
            elif key == ord('w') or key == ord('W'):
                draw_workspace = not draw_workspace
                print(f"[Action] Workspace overlays set to: {draw_workspace}")

    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user.")
    finally:
        if cap is not None:
            cap.release()
        detector.close()
        cv2.destroyAllWindows()
        print("[Info] Camera and detector released. Finished.")


if __name__ == "__main__":
    main()
