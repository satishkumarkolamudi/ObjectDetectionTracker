ODT — Object Detection & Tracking
=================================

This repository contains a small Python project that runs object detection and multi-object tracking on a video using YOLOv8 and ByteTrack. It includes a lightweight CLI wrapper (`main.py`), a detector wrapper (`odt/detector.py`) and a video processing loop (`odt/video_processor.py`).

This README explains how to set up the environment, how to run the project, how to adjust common settings, and the advantages of the chosen model and tracker compared to alternatives.

Quick start (1, 2)
-------------------
A concise two-step quick start so you can get running immediately:

1. Setup — create a virtual environment and install dependencies (Windows cmd.exe):

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run / Usage — process a video (example):

```bat
python main.py --video challenge-mle2.mp4
```

Press `q` in the display window to quit early.

Quick repository layout
-----------------------
- `main.py` — CLI entrypoint. Parses arguments, creates `Detector` and `VideoProcessor`, and runs processing.
- `odt/detector.py` — Thin wrapper around `ultralytics.YOLO` to run detection & tracking for a single frame.
- `odt/video_processor.py` — Reads video frames, calls the detector, and displays annotated frames. Converts plot results to OpenCV-friendly frames.
- `requirements.txt` — Python package dependencies used by the project.
- `yolov8l.pt` — (Not tracked: large model weight file) YOLOv8 large weights used by default. Replace with any YOLOv8 weights you want.
- `bytetrack.yaml` — Tracker config used by the YOLO wrapper to run ByteTrack.

Goals of the project
--------------------
- Demonstrate vehicle detection and tracking using a modern YOLOv8 model and ByteTrack for robust ID assignment.
- Provide an easy-to-run CLI for experimenting with thresholds, resizing, and tracker configuration.
- Be a small, extensible starting point for computer-vision projects that need detection + tracking.

Requirements / Supported platforms
----------------------------------
- Python 3.8+ (tested on 3.10/3.11)
- Windows, macOS, Linux should all work. Displaying a GUI window requires a display (cv2.imshow) — headless servers need a different output flow (writing video/images to disk).
- A CPU-only configuration works; a CUDA-capable GPU will greatly increase throughput if you install a GPU-capable PyTorch build.

Setup (Windows examples)
------------------------
1. Clone / open the repository and switch to its root.
2. Create and activate a virtual environment (cmd.exe):

```bat
python -m venv .venv
.venv\Scripts\activate
```

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install Python dependencies:

```powershell
pip install -r requirements.txt
```

Important note about PyTorch and CUDA
-------------------------------------
- `torch` is a dependency and must match your CUDA toolchain if you want GPU acceleration. The simplest reliable option is to choose the recommended install command from PyTorch's website: https://pytorch.org/get-started/locally

Examples (common commands)
--------------------------
Run the default example (uses `yolov8l.pt` and `challenge-mle2.mp4`):

```bat
python main.py --video challenge-mle2.mp4
```

Resize on the fly (increase or decrease speed/accuracy trade-off):

```bat
python main.py --video challenge-mle2.mp4 --fx 1.5 --fy 1.5
```

Use a different weights file:

```bat
python main.py --weights my_weights.pt --video challenge-mle2.mp4
```

If your `--video` argument is a camera index (0, 1, ...), pass it as an integer:

```bat
python main.py --video 0
```


Downloads / sources
-------------------
The example files used by this project can be downloaded from the upstream sources used when the repository was created:

- YOLOv8 large weights (example default `yolov8l.pt`):
  https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt
- ByteTrack tracker config (example default `bytetrack.yaml`):
  https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/bytetrack.yaml

These links point to the upstream releases/configs from Ultralytics. Download the weight file and the YAML tracker config into the repository root (or pass their paths via `--weights` / `--tracker`).

CLI reference (flags and defaults)
----------------------------------
The project provides a small CLI wrapper in `main.py`. Below are the supported flags, their default values, and a short explanation of each:

- `--weights` (default: `yolov8l.pt`)
  Path to the YOLO weights file to load. Replace with custom weights if desired (e.g., `yolov8n.pt`, `yolov8s.pt`, or your own trained file).

- `--video` (default: `challenge-mle2.mp4`)
  Path to the input video file to process, or a camera index (e.g. `0`) to use a webcam. If the argument is a numeric string it will be converted to an integer.

- `--fx` (default: `1.5`)
  Resize factor applied to frames in X (width). Increasing this scales frames up, decreasing scales down. Use this to trade off speed vs. accuracy.

- `--fy` (default: `1.5`)
  Resize factor applied to frames in Y (height). Typically set the same as `--fx` to keep aspect ratio.

- `--conf` (default: `0.3`)
  Confidence threshold for detections (values range 0.0 - 1.0). Detections with confidence below this value are filtered out before tracking.

- `--iou` (default: `0.5`)
  IoU (intersection-over-union) threshold used during non-maximum suppression and for some tracker behavior. Typical values are 0.4 - 0.6.

- `--classes` (default: `0 2 3 5 7`)
  Space-separated list of class IDs to keep. The example default filters for common road/vehicle/pedestrian classes. The defaults in this project are: `0` (person/pedestrian), `2` (car), `3` (motorbike), `5` (bus), `7` (truck). Use this to ignore irrelevant classes.

- `--tracker` (default: `bytetrack.yaml`)
  Path to the tracker YAML config used by Ultralytics' tracker wrapper (ByteTrack in the example). You can supply a different tracker config or path.

- `--output`
  - Type: string (path) or omitted (None)
  - Default: `None` (no file written)
  - Usage: `--output out.mp4` to write annotated frames to `out.mp4`.

- `--output-csv`
  - Type: string (path) or omitted (None)
  - Default: `None` (no CSV written)
  - Usage: `--output-csv detections.csv` to write per-frame detection/tracking rows to `detections.csv`.
  - CSV columns: `frame_id, object_id, class_id, x1, y1, x2, y2` (object_id will be -1 when the tracker did not provide an ID). Coordinates are in pixel units (top-left origin) and use the annotated frame coordinates after any resizing specified by `--fx/--fy`.

Notes on CSV data:
- `frame_id`: zero-based frame index (counts frames processed by the worker). If you require timestamps instead, consider adding a timestamp column (can be added on request).
- `object_id`: tracker-assigned ID if available; `-1` if not provided.
- `class_id`: integer class ID matching the model's dataset (e.g., `0` for person in COCO).
- `x1,y1,x2,y2`: bounding box in XYXY format (top-left and bottom-right corners).

Example: write annotated video and a CSV with detections:

```bat
python main.py --video challenge-mle2.mp4 --output out_annotations.mp4 --output-csv detections.csv
```

Example: headless recording only (no display window):

```bat
python main.py --video challenge-mle2.mp4 --no-display --output out_annotations.mp4 --output-csv detections.csv
```

- `--no-display`
  - Type: flag (boolean). Use presence to set True.
  - Default: `False` (display enabled)
  - Usage: include `--no-display` to hide cv2.imshow windows and run headless (commonly used with `--output`).

- `--benchmark`
  - Type: flag (boolean)
  - Default: `False`
  - Usage: include `--benchmark` to run a short synthetic benchmark of the detector and exit.

- `--benchmark-frames`
  - Type: int
  - Default: `20`
  - Usage: `--benchmark-frames 50` to run 50 synthetic frames for the benchmark.

Tools
-----
- `tools/export_model.py`: helper to export Ultralytics YOLO weights to ONNX or other formats supported by Ultralytics' `model.export()`.

Example:

```bat
python tools\export_model.py --weights yolov8l.pt --format onnx --output yolov8l.onnx
```

Usage examples (recap)
----------------------
Run the default example (uses `yolov8l.pt` and `challenge-mle2.mp4`):

```bat
python main.py --video challenge-mle2.mp4
```

Resize on the fly (increase or decrease speed/accuracy trade-off):

```bat
python main.py --video challenge-mle2.mp4 --fx 1.5 --fy 1.5
```

Use a different weights file:

```bat
python main.py --weights my_weights.pt --video challenge-mle2.mp4
```

If your `--video` argument is a camera index (0, 1, ...), pass it as an integer:

```bat
python main.py --video 0
```

Force a specific display FPS (e.g., 25 FPS) — useful to throttle display or match a target output framerate:

```bat
python main.py --video challenge-mle2.mp4 --target-fps 25
```

Force 15 FPS and disable frame-skipping (process every frame; may slow playback):

```bat
python main.py --video challenge-mle2.mp4 --target-fps 15 --disable-skip
```

Disable skipping only (process every frame at the source/default fps):

```bat
python main.py --video challenge-mle2.mp4 --disable-skip
```

Save annotated output to a file (headless or with display):

```bat
python main.py --video challenge-mle2.mp4 --output out_annotations.mp4
```

Run in headless mode (no display) and save output:

```bat
python main.py --video challenge-mle2.mp4 --no-display --output out_annotations.mp4
```

Run a quick synthetic benchmark of the detector (no video required):

```bat
python main.py --weights yolov8l.pt --benchmark --benchmark-frames 50
```

Export the YOLO model to ONNX using the helper script:

```bat
python tools\export_model.py --weights yolov8l.pt --format onnx --output yolov8l.onnx
```

Re-enable verbose YOLO/ByteTrack logs (for debugging)
-----------------------------------------------------
The code suppresses verbose per-frame speed/inference messages by default. If you want to see the original logs, set the environment variable `YOLO_VERBOSE=1` before running:

Windows (cmd.exe):

```bat
set YOLO_VERBOSE=1
python main.py --video challenge-mle2.mp4
```

PowerShell:

```powershell
$env:YOLO_VERBOSE = "1"
python main.py --video challenge-mle2.mp4
```

Headless / save results to disk
-------------------------------
- If `cv2.imshow` doesn't work (headless server or remote), you can write annotated frames to an output video file instead.
- The current `VideoProcessor` displays frames by default. Two options to get saved output:
  1. Modify `odt/video_processor.py` to open a `cv2.VideoWriter` and write `annotated` for each frame. (Prefered for headless servers.)
  2. Set `display=False` when calling `vp.process(...)` in `main.py` and instrument the code to write frames.

Example minimal change to write an output video (conceptual):
- Open `cv2.VideoWriter` with the same frame size and fps before the loop, call `writer.write(annotated)` inside the loop, release writer at the end.

Performance tips
----------------
- Use a GPU-enabled PyTorch build for real-time or near-real-time performance. See PyTorch install page for the correct CUDA wheel.
- Use smaller YOLO models for higher FPS at the cost of accuracy. Weigh the tradeoffs:
  - yolov8n (nano): fastest, lowest accuracy
  - yolov8s (small): fast, reasonable accuracy
  - yolov8m (medium): balanced
  - yolov8l (large): higher accuracy, slower (used in this repo)
  - yolov8x (xlarge): best accuracy, slowest and largest memory footprint
- Resize frames (use `--fx`/`--fy`) to a smaller size for improved inference time; too-small frames harm detection quality.
- Consider converting the model to an optimized runtime (TensorRT on NVIDIA GPUs, ONNX + TensorRT, OpenVINO for Intel) for further throughput improvements.

Why YOLOv8 (large) + ByteTrack?
-------------------------------
Short answer: modern accuracy with a lightweight, reliable tracker.

- YOLOv8 (ultralytics) is a current, well-maintained object detection architecture that provides strong detection accuracy and a convenient Python API. The `yolov8l.pt` weights provide a good balance of detection quality for vehicle detection tasks without the runtime and memory cost of `yolov8x`.
- ByteTrack is a recent multi-object tracker that focuses on robust ID assignment and good performance even with crowded scenes and missed detections. It is simpler and often more stable than some older trackers.

Comparisons to other options

- vs YOLOv8n/s/m (smaller YOLO variants):
  - Pros: `yolov8l` has higher accuracy for small/partially occluded objects and better overall mAP on common benchmarks.
  - Cons: larger model, higher memory and latency.
- vs YOLOv7/YOLOR/other older YOLO variants:
  - YOLOv8 benefits from newer architecture refinements and an actively maintained Ultralytics API, plus more convenient tools for deployment.
- vs two-stage detectors (Faster R-CNN, Mask R-CNN):
  - YOLOv8 is typically far faster and simpler to deploy for real-time applications; two-stage detectors can be more accurate in some cases but are heavier and slower.
- Tracking: ByteTrack vs DeepSORT / StrongSORT:
  - ByteTrack has simpler design, fewer moving parts, and good ID persistence under occlusion.
  - DeepSORT relies on appearance descriptors and can be more sensitive to descriptor quality; StrongSORT improves DeepSORT but adds complexity and dependencies.
  - Choose based on your needs: if you need appearance-based re-identification across cameras, consider StrongSORT; for single-camera robust ID maintenance ByteTrack is a great default.

Design details
--------------
This project aims to be informative and practical. The following design notes document the choices made and how common edge-cases are handled.

- Models used for detection
  - Default: YOLOv8 (large) — `yolov8l.pt` (Ultralytics). This is the repository default and balances accuracy/latency for vehicle detection tasks.
  - Alternatives: any Ultralytics-compatible YOLOv8 weights can be used (e.g., `yolov8n`, `yolov8s`, `yolov8m`, `yolov8x`) by passing `--weights` on the CLI.
  - Notes: the wrapper uses the Ultralytics YOLO API (e.g., `ultralytics.YOLO`). Model I/O and results rely on the library's `predict`/`track` return types.

- Tracking methodology
  - Tracker: ByteTrack via the Ultralytics tracking wrapper with a YAML config (default `bytetrack.yaml`). The project uses `ultralytics.YOLO.track(...)` and supplies the tracker config path to configure ByteTrack behavior.
  - How it works (high-level): detections are associated across frames using motion and IoU-based matching and score/threshold heuristics. ByteTrack keeps short-lived unmatched tracks to bridge brief occlusions and assigns persistent numeric IDs to matched tracks.
  - IDs and missing IDs: when the tracker cannot confidently assign an ID it may emit a placeholder (the project writes `-1` in the CSV when no tracker ID is available).

- Performance optimization steps (what the project exposes and recommended extra steps)
  - Built-in options:
    - Resize frames with `--fx`/`--fy` to reduce the input size (fastest way to trade accuracy for speed).
    - Skip frames when processing is slower than real-time (default behavior) or disable skipping with `--disable-skip` to process every frame.
    - Choose a smaller YOLOv8 variant via `--weights` for higher FPS (e.g., `yolov8n.pt` or `yolov8s.pt`).
  - Recommended runtime optimizations:
    - Use a GPU and install a CUDA-enabled PyTorch wheel that matches your drivers.
    - Run the model in half-precision (FP16) on supported GPUs: move the model to CUDA and enable `half()` where applicable — Ultralytics supports `device` and `half` options through its API or conversion to optimized runtime.
    - Convert the model to an optimized runtime (ONNX -> TensorRT, TorchScript, OpenVINO) for significant speedups in production. The repository contains `tools/export_model.py` as a helper.
    - If doing batch processing or large-scale offline runs, run inference in parallel workers or use frame batching where the model API supports it.
  - I/O and display:
    - Disable `cv2.imshow` (use `--no-display`) and write annotated frames to disk with `--output` to avoid GUI overhead.
    - Use an appropriate `cv2.VideoWriter` codec and a lower resolution output to reduce disk I/O pressure.

- Handling occlusion, entry/exit of objects, and frame drops
  - Occlusion and short-term misses:
    - ByteTrack is resilient to short occlusions by keeping recently-lost tracks in memory for a configurable buffer time (controlled in the tracker YAML such as `track_buffer` / `max_time_lost` parameters). This allows a returning object to regain its previous ID.
    - Very long occlusions or complete disappearance will result in track deletion and a new ID when the object reappears.
  - Entry / exit of objects:
    - New detections not matched to existing tracks receive new IDs. When tracks are lost for longer than the configured retention they are removed (considered exit).
  - Frame drops and skipped frames:
    - The pipeline will attempt to match detections across the frames that were actually processed. If the processing cannot keep up with real-time, the default behavior is to skip source frames (faster playback) to avoid growing latency. Use `--disable-skip` to force processing every frame (may increase lag).
    - For offline processing or when frame continuity is required, run with `--disable-skip` and ensure the input source is read from disk (no frame-loss at capture time) or use buffering to prevent dropped frames.

- Any assumptions made
  - Model & dataset IDs: class IDs used in `--classes` follow COCO label indexing (e.g., 0=person, 2=car, etc.). If you use a custom model/dataset, update class IDs accordingly.
  - Coordinate space: CSV coordinates and annotated frames use the resized frame pixel space (after applying `--fx`/`--fy`). If you need original-video coordinates, you must scale boxes back to the original frame size.
  - Tracker config availability: `bytetrack.yaml` (or an alternative tracker config) must be present and compatible with your Ultralytics version.
  - Ultralytics API stability: the code relies on the Ultralytics `YOLO` API (`track`, `predict`, `results.plot()`); breaking API changes in new library releases may require small wrapper updates.
  - Display environment: `cv2.imshow` requires a display; for headless servers run with `--no-display` and `--output`.


License and attribution
-----------------------
- Model weights (e.g., `yolov8l.pt`) are generally distributed by Ultralytics under their license; check Ultralytics' terms before redistribution.
- This project is a learning/demo project — choose and assign an appropriate open-source license if you want to share it (e.g., MIT, Apache-2.0).

Contact / next steps
--------------------
If you'd like, I can:
- Add an `--output` flag and implement `cv2.VideoWriter` support so you can save the annotated video automatically.
- Add an option to run a benchmark (average inference time) across a few frames.
- Add example scripts to convert the model to ONNX or TensorRT for speed.

If you want me to implement any of these additions, tell me which one to do next and I'll modify the code and validate it.
