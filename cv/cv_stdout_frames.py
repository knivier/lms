#!/usr/bin/env python3
"""
Run cv.py pipeline (camera + skeleton + text) and output each frame as base64 JPEG to stdout.
Used by the Tauri app so the Python output shows up natively in the UI (no HTTP stream).
Same pipeline as cv-view / cv_stream_server: create_view_core + produce_combined_frame.
"""
import argparse
import base64
import sys
import time
from pathlib import Path

# Max FPS sent to the app; lowers pipe backlog and perceived lag.
OUTPUT_FPS = 60

# Ensure cv package is importable (repo root)
_CV_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CV_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import importlib.util

from cv import CAMERA_ID

_cv_view_file = _CV_DIR / "cv_view.py" if (_CV_DIR / "cv_view.py").exists() else _CV_DIR / "cv-view.py"
_spec = importlib.util.spec_from_file_location("cv_view", _cv_view_file)
_cv_view = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cv_view)
create_view_core = _cv_view.create_view_core
produce_combined_frame = _cv_view.produce_combined_frame


def main():
    parser = argparse.ArgumentParser(description="Output cv.py frames as base64 JPEG to stdout (for Tauri native feed).")
    parser.add_argument("--camera", type=int, default=CAMERA_ID, help="Camera device id (default: from cv/config.yaml)")
    args = parser.parse_args()
    try:
        core = create_view_core(args.camera)
    except RuntimeError as e:
        print(f"CV init failed: {e}", file=sys.stderr)
        sys.exit(1)
    frame_interval = 1.0 / OUTPUT_FPS
    next_output_time = time.monotonic()
    try:
        while True:
            combined, cont = produce_combined_frame(core)
            if not cont:
                break
            now = time.monotonic()
            if now >= next_output_time:
                next_output_time = now + frame_interval
                _, jpeg = cv2.imencode(".jpg", combined)
                b64 = base64.standard_b64encode(jpeg.tobytes()).decode("ascii")
                try:
                    print(b64, flush=True)
                except BrokenPipeError:
                    break
            # Don't sleep: keep pulling frames so when we do output it's the freshest.
    finally:
        core.close()


if __name__ == "__main__":
    main()
