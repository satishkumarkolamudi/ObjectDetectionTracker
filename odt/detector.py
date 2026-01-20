import torch
from ultralytics import YOLO
import contextlib
import os
import logging

# Reduce ultralytics' logger verbosity by default
try:
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
except Exception:
    pass

class Detector:
    """Wrapper around YOLO model for detection and tracking.

    Responsibilities:
    - Load model from a weights path.
    - Provide a `track_frame` method which runs tracking on a single frame and returns the results object.
    """

    def __init__(self, weights_path: str = "yolov8l.pt", device: str = None):
        self.weights_path = weights_path
        # let ultralytics/torch choose device if None
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # instantiate model
        self.model = YOLO(self.weights_path)

        # try to move the model to the requested device; not all YOLO wrappers require this but it's safe to attempt
        try:
            # many PyTorch-based models support .to(device)
            self.model.to(self.device)
        except Exception:
            # if move fails, continue; the model may manage device internally
            pass

    def track_frame(self, frame, persist: bool = True, conf: float = 0.3, iou: float = 0.5, classes=None, tracker: str = "bytetrack.yaml"):
        """Run tracking on a single frame.

        Returns YOLO results list (usually length 1 for single image input).
        """
        # Allow overriding suppression via environment variable YOLO_VERBOSE=1
        verbose = os.getenv("YOLO_VERBOSE", "0") in ("1", "true", "True")

        if verbose:
            # don't redirect output, run normally
            results = self.model.track(
                frame,
                persist=persist,
                conf=conf,
                iou=iou,
                classes=classes,
                tracker=tracker,
            )
            return results

        # Suppress noisy console output from the model.track call (speed/inference logs)
        devnull = open(os.devnull, "w")
        try:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                results = self.model.track(
                    frame,
                    persist=persist,
                    conf=conf,
                    iou=iou,
                    classes=classes,
                    tracker=tracker,
                )
        finally:
            devnull.close()

        return results
