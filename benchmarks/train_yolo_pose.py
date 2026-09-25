"""
Training / Fine-tuning Script for YOLO Pose Estimation Models
Reads training parameters from pose_models_config.json or accepts command line overrides.
Supports YOLO26n-pose, YOLO26m-pose, YOLO26x-pose (Nano, Medium, XLarge).
"""

import os
import sys
import json
import argparse
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "configs", "pose_models_config.json")


def load_model_train_config(model_key):
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            if "models" in cfg and model_key in cfg["models"]:
                return cfg["models"][model_key]
    return {}


def train_pose_model(model_key="yolo26n_pose",
                     dataset="coco8-pose.yaml",
                     epochs=None,
                     batch_size=None,
                     imgsz=None,
                     lr0=None,
                     device="cpu",
                     project=None):
    model_cfg = load_model_train_config(model_key)
    if not model_cfg:
        print(f"Warning: Model key '{model_key}' not found in {CONFIG_PATH}. Using standard settings.")

    weights_name = model_cfg.get("weights_file", "yolo11n-pose.pt")
    search_paths = [
        os.path.join(SCRIPT_DIR, "weights", weights_name),
        os.path.join(SCRIPT_DIR, "models", weights_name),
        os.path.join(os.path.dirname(SCRIPT_DIR), weights_name),
        weights_name
    ]
    weights_path = weights_name
    for p in search_paths:
        if os.path.exists(p):
            weights_path = os.path.abspath(p)
            break

    t_cfg = model_cfg.get("train_config", {})
    epochs     = epochs or t_cfg.get("epochs", 50)
    batch_size = batch_size or t_cfg.get("batch_size", 16)
    imgsz      = imgsz or model_cfg.get("input_size", 640)
    lr0        = lr0 or t_cfg.get("lr0", 0.01)
    dataset    = dataset or t_cfg.get("dataset_yaml", "coco8-pose.yaml")
    project    = project or t_cfg.get("save_dir", f"runs/pose/{model_key}")

    print("=" * 70)
    print(f"  Training Configuration for: {model_key}")
    print("=" * 70)
    print(f"  Base Weights : {weights_path}")
    print(f"  Dataset      : {dataset}")
    print(f"  Epochs       : {epochs}")
    print(f"  Batch Size   : {batch_size}")
    print(f"  Image Size   : {imgsz}")
    print(f"  Learning Rate: {lr0}")
    print(f"  Device       : {device}")
    print(f"  Save Output  : {project}")
    print("=" * 70)

    model = YOLO(weights_path)
    results = model.train(
        data=dataset,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        lr0=lr0,
        project=project,
        device=device
    )
    print(f"\n[Training Complete] Results saved to: {project}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train/Fine-tune YOLO-Pose Models")
    parser.add_argument("--model", type=str, default="yolo26n_pose",
                        choices=["yolo26n_pose", "yolo26m_pose", "yolo26x_pose"],
                        help="Model key to train")
    parser.add_argument("--dataset", type=str, default="coco8-pose.yaml",
                        help="Dataset yaml file (e.g. coco8-pose.yaml, custom_pose.yaml)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Image resolution")
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, 0, 1)")
    parser.add_argument("--save-dir", type=str, default=None, help="Output directory for weights and logs")
    args = parser.parse_args()

    train_pose_model(
        model_key=args.model,
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr,
        device=args.device,
        project=args.save_dir
    )


if __name__ == "__main__":
    main()
