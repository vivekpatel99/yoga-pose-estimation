"""
Example usage:
uv run testing_model.py --model-path output/train/weights/best.pt --dataset-path datasets/data_test.yaml --output-path prediction_output
uv run testing_model.py --model-path models/yolo11l-pose.pt --dataset-path datasets/data_test.yaml --output-path prediction_output
uv run testing_model.py --model-path models/yolo11l-pose.pt --dataset-path datasets/test/new_images --output-path prediction_output
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def test(
    trained_model_path: str,
    data_path: Path,
    output_path: Path,
    imgsz: int,
    file_extensions=("*.jpg", "*.jpeg", "*.png", "*.webp"),
):
    """
    Validate a YOLO pose estimation model on a test dataset.

    Args:
        trained_model_path (str): The path to the trained model weights (e.g., 'output/train/weights/best.pt').
        data_path (Path): Path to the dataset YAML file.
        output_path (Path): Directory to save validation results.
        imgsz (int): Image size for validation.
    """
    # Ensure the output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Load a trained model
    model = YOLO(trained_model_path)

    # Iterate over all supported image files in the test directory
    for ext in file_extensions:
        for img_path in data_path.glob(ext):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue

            # The 'name' parameter will create a subdirectory in the 'project' directory.
            # For example, 'prediction_output/val'
            model.predict(
                img,
                conf=0.5,
                plots=True,  # Save validation plots
                save=True,  # Save results in JSON format
                project=output_path,
                name="predictions",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate a YOLO pose estimation model."
    )
    parser.add_argument(
        "-mp",
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model weights (e.g., 'output/train/weights/best.pt').",
    )
    parser.add_argument(
        "-dp",
        "--dataset-path",
        type=Path,
        default=Path("datasets/data.yaml"),
        help="Path to the dataset YAML file.",
    )
    parser.add_argument(
        "-i", "--imgsz", type=int, default=640, help="Image size for validation."
    )
    parser.add_argument(
        "-op",
        "--output-path",
        type=Path,
        default=Path("prediction_output"),
        help="Directory to save validation results.",
    )

    args = parser.parse_args()

    test(
        trained_model_path=args.model_path,
        data_path=args.dataset_path,
        output_path=args.output_path,
        imgsz=args.imgsz,
    )
