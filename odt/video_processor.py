import cv2
from typing import Optional
import numpy as np
from PIL import Image
import time
import threading
import queue

class VideoProcessor:
    """Handles reading frames from a video source, applying a detector, and displaying/saving results."""

    def __init__(self, source: str = 0, resize_fx: Optional[float] = 1.0, resize_fy: Optional[float] = 1.0, window_name: str = "Video"):
        self.source = source
        self.resize_fx = resize_fx
        self.resize_fy = resize_fy
        self.window_name = window_name

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {self.source}")

    def process(self, detector, classes=None, conf=0.3, iou=0.5, tracker="bytetrack.yaml", display: bool = True, target_fps: Optional[float] = None, skip_frames: bool = True, output_path: Optional[str] = None):
        """Read frames, run detector.track in a worker thread, and display annotated frames in the main thread.

        New arguments:
        - target_fps: if set, force display to this FPS; otherwise use source video's FPS (fallback 30).
        - skip_frames: when True, capture/worker will drop old frames when queues are full to try to keep up with realtime.
        """
        # Queues to communicate between threads
        raw_q = queue.Queue(maxsize=5)
        annotated_q = queue.Queue(maxsize=5)
        stop_event = threading.Event()

        # Determine source FPS (used as default for display pacing)
        source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        try:
            if source_fps is None or source_fps <= 0 or (isinstance(source_fps, float) and (source_fps != source_fps)):
                source_fps = 30.0
        except Exception:
            source_fps = 30.0

        fps = float(target_fps) if (target_fps is not None and target_fps > 0) else float(source_fps)
        frame_duration = 1.0 / fps
        try:
            print(f"Video source FPS: {source_fps:.2f} -> using display FPS: {fps:.2f} ({frame_duration*1000:.1f} ms/frame)")
        except Exception:
            pass

        def capture_loop():
            # Read frames and push to raw_q. Stop on EOF or stop_event.
            while not stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    # signal EOF
                    stop_event.set()
                    break

                if self.resize_fx != 1.0 or self.resize_fy != 1.0:
                    try:
                        frame = cv2.resize(frame, None, fx=self.resize_fx, fy=self.resize_fy, interpolation=cv2.INTER_CUBIC)
                    except Exception:
                        # If resize fails, continue with original frame
                        pass

                try:
                    raw_q.put(frame, timeout=0.01)
                except queue.Full:
                    if skip_frames:
                        # drop oldest frame to make room
                        try:
                            _ = raw_q.get_nowait()
                        except Exception:
                            pass
                        try:
                            raw_q.put_nowait(frame)
                        except Exception:
                            pass
                    else:
                        # block until space is available (slows capture)
                        try:
                            raw_q.put(frame, timeout=1.0)
                        except Exception:
                            # give up and continue
                            pass

        def worker_loop():
            # Consume raw frames, run detection, and push annotated frames
            while not stop_event.is_set() or not raw_q.empty():
                try:
                    frame = raw_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                try:
                    results = detector.track_frame(frame, persist=True, conf=conf, iou=iou, classes=classes, tracker=tracker)
                    annotated = results[0].plot()
                except Exception:
                    # If detector fails, skip this frame
                    annotated = None

                # Normalize annotated to a BGR numpy array if possible
                if annotated is None:
                    # nothing to show for this frame
                    continue

                if isinstance(annotated, Image.Image):
                    try:
                        annotated = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
                    except Exception:
                        continue
                elif isinstance(annotated, np.ndarray):
                    pass
                else:
                    try:
                        annotated = np.asarray(annotated)
                    except Exception:
                        continue

                try:
                    annotated_q.put(annotated, timeout=0.01)
                except queue.Full:
                    if skip_frames:
                        # drop oldest annotated to make room
                        try:
                            _ = annotated_q.get_nowait()
                        except Exception:
                            pass
                        try:
                            annotated_q.put_nowait(annotated)
                        except Exception:
                            pass
                    else:
                        try:
                            annotated_q.put(annotated, timeout=1.0)
                        except Exception:
                            pass

        # Start threads
        cap_thread = threading.Thread(target=capture_loop, name="capture-thread", daemon=True)
        worker_thread = threading.Thread(target=worker_loop, name="worker-thread", daemon=True)
        cap_thread.start()
        worker_thread.start()

        last_displayed = None
        writer = None
        # keep the output path in a local var (may be None)
        out_path = output_path
        try:
            # Main display loop - runs in the main thread so cv2.imshow is safe
            while not stop_event.is_set() or not annotated_q.empty() or not raw_q.empty():
                loop_start = time.perf_counter()

                # Prefer newest annotated frame to avoid showing stale frames; if none available, keep last
                annotated = None
                try:
                    # drain annotated queue to get the most recent frame
                    while True:
                        annotated = annotated_q.get_nowait()
                except queue.Empty:
                    pass

                if annotated is not None:
                    last_displayed = annotated

                # Lazily initialize the VideoWriter when we have the first annotated frame
                if writer is None and out_path is not None and last_displayed is not None:
                    try:
                        h, w = last_displayed.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                        if not writer.isOpened():
                            print(f"Warning: Could not open VideoWriter for {out_path}")
                            writer = None
                        else:
                            print(f"Writing annotated output to: {out_path} ({w}x{h} @ {fps:.2f} FPS)")
                    except Exception as e:
                        print(f"Warning: failed to create VideoWriter: {e}")
                        writer = None

                if last_displayed is None:
                    # nothing to display yet; wait a short time
                    if stop_event.is_set():
                        break
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        stop_event.set()
                        break
                    time.sleep(min(0.01, frame_duration))
                    continue

                if display:
                    try:
                        cv2.imshow(self.window_name, last_displayed)
                    except Exception:
                        # If imshow fails, continue and attempt graceful shutdown
                        pass

                # Write annotated frames to output if requested (non-blocking)
                if writer is not None:
                    try:
                        writer.write(last_displayed)
                    except Exception:
                        # ignore write errors; we don't want to crash the display loop
                        pass

                # maintain display pacing based on fps and processing time
                elapsed = time.perf_counter() - loop_start
                remaining = frame_duration - elapsed
                if remaining > 0:
                    key = cv2.waitKey(max(1, int(remaining * 1000))) & 0xFF
                    if key == ord("q"):
                        stop_event.set()
                        break
                else:
                    # we're behind; allow UI events and continue. Do not sleep here.
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        stop_event.set()
                        break

        finally:
            # signal threads to stop and clean up
            stop_event.set()
            try:
                cap_thread.join(timeout=1.0)
            except Exception:
                pass
            try:
                worker_thread.join(timeout=1.0)
            except Exception:
                pass
            self.cap.release()
            # release writer if it was opened
            try:
                if writer is not None:
                    writer.release()
            except Exception:
                pass
            if display:
                cv2.destroyAllWindows()
