#!/usr/bin/env python3
"""
Real-Time Posture Analysis Webcam Demo.
SpineVision EdgeAI

Features:
  - Real-time Pose Estimation (MediaPipe Pose Full / Lite)
  - Biometric Ergonomic Angle Calculation:
      * Craniovertebral Angle (CVA) — Forward Head / Text Neck
      * Shoulder Tilt Angle — Lateral Spinal Asymmetry
      * Trunk / Spine Slump Angle — Slouching / Kyphosis
      * Eye-to-Screen Distance — Myopia Risk Estimation (Pinhole Model)
  - Temporal Smoothing: One-Euro Filter (eliminates micro-jitter)
  - Auto-Calibration: Personalized "Good Posture" Baseline (3 seconds)
  - Workspace Surface & Screen Detection
  - Comprehensive Visual HUD (Heads-Up Display)

Controls:
  [C] : Start 3-second Auto-Calibration (sit upright & look at screen)
  [R] : Reset Calibration (return to absolute clinical thresholds)
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
)
from posture_analyzer.config import get_posture_config
from benchmarks.models.mediapipe_pose_full import MediaPipePoseDetector, MODEL_PATH as DEFAULT_MP_PATH
from benchmarks.models.drawing_utils import open_camera

try:
    import psutil
    _PROCESS = psutil.Process()
except ImportError:
    _PROCESS = None


def parse_args():
    cfg = get_posture_config()
    cam_cfg = cfg.get("camera", {})
    class_cfg = cfg.get("classification", {})

    parser = argparse.ArgumentParser(description="SpineVision Real-Time Posture Analysis Demo")
    parser.add_argument("--camera", type=int, default=cam_cfg.get("camera_index", 0), help="Camera index")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MP_PATH, help="Path to pose_landmarker task file")
    parser.add_argument("--width", type=int, default=cam_cfg.get("width", 1280), help="Camera capture width")
    parser.add_argument("--height", type=int, default=cam_cfg.get("height", 720), help="Camera capture height")
    parser.add_argument("--no-smoothing", action="store_true", help="Disable One-Euro filter smoothing")
    parser.add_argument("--no-workspace", action="store_true", help="Disable workspace overlays")
    parser.add_argument("--sensitivity", type=float, default=class_cfg.get("default_sensitivity", 1.0), help="Classifier sensitivity")
    return parser.parse_args()



def main():
    cfg = get_posture_config()
    args = parse_args()

    print("=" * 70)
    print(" SpineVision EdgeAI — Real-Time Ergonomic Posture Analysis")
    print("=" * 70)
    print("Controls:")
    print("  [C] : Start Auto-Calibration (sit upright for ~3s)")
    print("  [R] : Reset Calibration")
    print("  [S] : Toggle One-Euro Smoothing")
    print("  [W] : Toggle Workspace Surface & Screen Overlays")
    print("  [Q] : Quit")
    print("=" * 70)

    # 1. Initialize Pose Detector
    if not os.path.exists(args.model_path):
        print(f"[Error] MediaPipe task weight file not found: {args.model_path}")
        print("Please check the path or run benchmark setup.")
        sys.exit(1)

    print(f"Loading MediaPipe Pose Landmarker from: {args.model_path}")
    detector = MediaPipePoseDetector(model_path=args.model_path, num_poses=1)

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

            # 4. Pose Inference
            result, latency_ms = detector.infer(frame, timestamp_ms=current_timestamp_ms)

            # 5. Posture Analysis
            if result and result.pose_landmarks and len(result.pose_landmarks) > 0:
                # Primary subject (first detected person)
                landmarks = result.pose_landmarks[0]

                # Draw base skeleton with stable visibility threshold (0.45 ignores occluded phantom limbs)
                detector.draw(frame, result, vis_thresh=0.45, radius_min=3, radius_max=6)

                # Process through posture analysis engine
                state, kps = engine.process_mediapipe(
                    landmarks,
                    frame_width=w,
                    frame_height=h,
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
                # No pose detected: draw a subtle notice
                cv2.putText(
                    frame,
                    "No person detected in frame",
                    (w // 2 - 160, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
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

            # Top-left info overlay with RAM consumption
            ram_mb = _PROCESS.memory_info().rss / (1024.0 * 1024.0) if _PROCESS is not None else 0.0
            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms | RAM: {ram_mb:.0f}MB",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )


            # Show active status of features
            smoothing_str = "ON (One-Euro)" if engine.use_smoothing else "OFF"
            calib_str = "CALIBRATED" if engine.is_calibrated else "UNCALIBRATED"
            cv2.putText(
                frame,
                f"Filter: {smoothing_str} | Base: {calib_str}",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
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
