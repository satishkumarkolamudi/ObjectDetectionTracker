"""Small helper to export an Ultralytics YOLO model to ONNX or other formats.

Usage examples (Windows cmd):
    python tools\export_model.py --weights yolov8l.pt --format onnx --output yolov8l.onnx
    python tools\export_model.py --weights yolov8l.pt --format onnx --opset 12

This script calls the model.export(...) helper provided by ultralytics.YOLO.
"""
import argparse
import os

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser("Export ultralytics YOLO model to ONNX/other formats")
    p.add_argument("--weights", required=True, help="Path to YOLO weights (e.g., yolov8l.pt)")
    p.add_argument("--format", default="onnx", choices=["onnx", "torch", "tensorrt", "openvino", "saved_model"], help="Target export format")
    p.add_argument("--output", default=None, help="Output file path (optional). If omitted, default location is used by the model.export helper")
    p.add_argument("--opset", type=int, default=None, help="ONNX opset version (optional)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.weights):
        print(f"Weights not found: {args.weights}")
        return

    model = YOLO(args.weights)

    export_kwargs = {"format": args.format}
    if args.opset is not None:
        export_kwargs["opset"] = args.opset
    if args.output is not None:
        export_kwargs["task"] = "detect"
        export_kwargs["file"] = args.output

    print(f"Exporting {args.weights} to format={args.format} ...")
    try:
        model.export(**export_kwargs)
        print("Export completed (check output path or default ultralytics export location).")
    except Exception as e:
        print(f"Export failed: {e}")


if __name__ == '__main__':
    main()

