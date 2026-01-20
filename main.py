import argparse
import time
import torch
from odt import Detector, VideoProcessor


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run vehicle tracking on a video using YOLOv8 + ByteTrack")
    parser.add_argument("--weights", default="yolov8l.pt", help="Path to YOLO weights file")
    parser.add_argument("--video", default="challenge-mle2.mp4", help="Path to input video file or camera index")
    parser.add_argument("--fx", type=float, default=1.5, help="Resize factor in X")
    parser.add_argument("--fy", type=float, default=1.5, help="Resize factor in Y")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IOU threshold")
    parser.add_argument("--classes", nargs="*", type=int, default=[0, 2, 3, 5, 7], help="Class IDs to keep (default: 0 for Pedestrian, 2 for cars, 3 for motorcycle, 5 for bus, 7 for truck)")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Tracker config path")
    parser.add_argument("--target-fps", type=float, default=None, help="Force display to this FPS (default: use source video's FPS)")
    parser.add_argument("--disable-skip", action="store_true", help="Disable frame skipping when processing is slower than realtime")
    parser.add_argument("--output", default=None, help="Path to output annotated video file (optional). Example: out.mp4")
    parser.add_argument("--no-display", action="store_true", help="Run in headless mode without showing cv2.imshow; useful for saving output only")
    parser.add_argument("--benchmark", action="store_true", help="Run a quick benchmark of the detector and exit")
    parser.add_argument("--benchmark-frames", type=int, default=20, help="Number of frames to run for the benchmark (default: 20)")
    parser.add_argument("--output-csv", default=None, help="Path to write CSV of detections: columns frame_id, object_id, class_id, x1, y1, x2, y2")
    return parser.parse_args(argv)


def run_benchmark(detector: Detector, frames: int = 20, size=(640, 360)):
    """Run a simple synthetic benchmark: run detector.track_frame on random frames and print average time."""
    import numpy as np

    print(f"Running benchmark with {frames} frames (frame size {size[0]}x{size[1]})")
    # create synthetic random frames in uint8 BGR
    times = []
    for i in range(frames):
        img = (np.random.rand(size[1], size[0], 3) * 255).astype('uint8')
        t0 = time.perf_counter()
        try:
            _ = detector.track_frame(img, persist=False)
        except Exception as e:
            # continue, but report
            print(f"Warning: detector failed on synthetic frame {i}: {e}")
        dt = time.perf_counter() - t0
        times.append(dt)
    avg = sum(times) / max(1, len(times))
    print(f"Benchmark completed: avg inference+track time per frame = {avg*1000:.1f} ms ({1.0/avg:.2f} FPS)")


def main(argv=None):
    args = parse_args(argv)

    # Print device info if available
    if torch.cuda.is_available():
        try:
            print("CUDA device:", torch.cuda.get_device_name(0))
        except Exception:
            print("CUDA is available but could not get device name")
    else:
        print("CUDA not available, using CPU")

    detector = Detector(weights_path=args.weights)

    # If benchmark requested, run and exit
    if args.benchmark:
        run_benchmark(detector, frames=args.benchmark_frames)
        return

    # If video is an integer (camera index) allow passing as int
    source = args.video
    try:
        # transform numeric strings to int for cv2.VideoCapture
        if str(source).isdigit():
            source = int(source)
    except Exception:
        pass

    vp = VideoProcessor(source=source, resize_fx=args.fx, resize_fy=args.fy, window_name="Vehicle Detection")

    vp.process(
        detector=detector,
        classes=args.classes,
        conf=args.conf,
        iou=args.iou,
        tracker=args.tracker,
        display=(not args.no_display),
        target_fps=args.target_fps,
        skip_frames=(not args.disable_skip),
        output_path=args.output,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
