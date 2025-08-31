"""
Example usage with long flags:
uv run train.py --model-path models/yolo11l-pose.pt --dataset-path datasets/data.yaml --epochs 100 --imgsz 640 --output-path output

Example usage with short flags:
uv run train.py -mp models/yolo11l-pose.pt -dp datasets/data.yaml -e 3 -i 640 -op output

uv run train.py -mp output/good-result-train4/weights/best.pt -dp datasets/data.yaml -e 300 -i 640 -op output
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from constants import OUTPUT


def train(
    model_name: str,
    data_path: Path,
    epochs: int,
    imgsz: int,
    output_path: Path,
    batch_size: int,
):
    """
    Train a YOLO pose estimation model.

    Args:
        model_name (str): The name of the model to use (e.g., 'yolov8n-pose.pt').
        data_path (Path): Path to the dataset YAML file.
        epochs (int): Number of training epochs.
        imgsz (int): Image size for training.
        output_path (Path): Directory to save training results.
        batch_size (int): The batch size for training.

    """
    # Ensure the output directory exists before training
    output_path.mkdir(parents=True, exist_ok=True)

    # Load a model
    model = YOLO(model_name)

    # The 'name' parameter will create a subdirectory in the 'project' directory.
    # For example, 'output/train'
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        project=str(output_path),
        device=0,
        name="train",
        # CRITICAL SETTINGS FOR HIGH ACCURACY
        optimizer="AdamW",  # Better than Adam for pose
        lr0=0.0001,  # Lower learning rate
        lrf=0.01,  # Learning rate final factor
        momentum=0.9,  # SGD momentum
        weight_decay=0.0005,  # L2 regularization
        warmup_epochs=5,  # Gradual warmup
        warmup_momentum=0.8,  # Warmup momentum
        # POSE-SPECIFIC PARAMETERS
        pose=15.0,  # Pose loss gain (critical!)
        kobj=1.0,  # Keypoint objectness
        cls=0.5,  # Classification loss gain
        box=7.5,  # Box regression loss gain
        # ADVANCED TRAINING SETTINGS
        cos_lr=True,  # Cosine learning rate scheduler
        patience=25,  # Early stopping patience
        save_period=10,  # Save checkpoint every N epochs
        # DATA AUGMENTATION (CRUCIAL FOR POSE)
        hsv_h=0.015,  # HSV hue augmentation
        hsv_s=0.7,  # HSV saturation
        hsv_v=0.4,  # HSV value
        degrees=10.0,  # Rotation degrees
        translate=0.1,  # Translation fraction
        scale=0.5,  # Scaling factor range
        shear=2.0,  # Shear degrees
        perspective=0.0001,  # Perspective transform
        flipud=0.0,  # Vertical flip probability
        fliplr=0.5,  # Horizontal flip probability
        mosaic=1.0,  # Mosaic augmentation probability
        mixup=0.1,  # Mixup augmentation probability
        copy_paste=0.1,  # Copy-paste augmentation
        # VALIDATION & MONITORING
        val=True,  # Validate during training
        plots=True,  # Save training plots
        save_json=True,  # Save results in JSON format
    )
    print(f"Training complete. Results saved to {results.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLO pose estimation model.")
    parser.add_argument(
        "-mp",
        "--model-path",
        type=str,
        default="models/yolov11l-pose.pt",
        help="Model to use for training (e.g., 'yolov8n-pose.pt').",
    )
    parser.add_argument(
        "-dp",
        "--dataset-path",
        type=Path,
        default=Path("datasets/data.yaml"),
        help="Path to the dataset YAML file.",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument(
        "-bz", "--batch-size", type=int, default=8, help="Batch size for training."
    )
    parser.add_argument(
        "-i", "--imgsz", type=int, default=640, help="Image size for training."
    )
    parser.add_argument(
        "-op",
        "--output-path",
        type=Path,
        default=OUTPUT,
        help="Directory to save training results.",
    )

    args = parser.parse_args()

    train(
        model_name=args.model_path,
        data_path=args.dataset_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        output_path=args.output_path,
    )
