#!/usr/bin/env python3
"""
Replay a recorded video through the cv.py pipeline: pose overlay + JSONL logging.
Uses the same core (PoseCore) as cv-view.py but feeds from a file instead of live camera.
Run from repo root: python -m cv.cv-youtubefeeder  (default: cv/slowpushups.mp4)
Or: python -m cv.cv-youtubefeeder path/to/video.mp4
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from cv import (
    PoseCore,
    build_text_panel,
    draw_skeleton,
    flip_landmarks_x,
    install_ctrl_c,
    ELBOW_ALERT_RED_ALPHA,
)


def run_feeder(video_path, log_path=None, mirror=False, realtime=True):
    """Run pose overlay + logging on a recorded video."""
    try:
        core = PoseCore(video_path=video_path, log_path=log_path)
    except RuntimeError as exc:
        print(exc)
        return

    TEXT_PANEL_WIDTH = 420
    WIN_HEIGHT = core.height
    WIN_WIDTH = core.width + TEXT_PANEL_WIDTH
    cv2.namedWindow("Pose Overlay (video)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pose Overlay (video)", WIN_WIDTH, WIN_HEIGHT)

    exit_requested = False
    frame_delay_sec = 1.0 / core.fps if realtime else 0

    def _on_sigint(*_):
        nonlocal exit_requested
        exit_requested = True

    install_ctrl_c(_on_sigint)

    try:
        while not exit_requested:
            t0 = time.perf_counter()
            data = core.step()
            if data is None:
                break

            frame = data["frame"]
            if mirror:
                frame = cv2.flip(frame, 1)
                lm_draw = flip_landmarks_x(data["landmarks"])
            else:
                lm_draw = data["landmarks"]

            draw_skeleton(frame, lm_draw, data["connection_spec"])
            if data.get("alert_red"):
                red = np.full_like(frame, (0, 0, 255))
                frame = cv2.addWeighted(frame, 1.0 - ELBOW_ALERT_RED_ALPHA, red, ELBOW_ALERT_RED_ALPHA, 0)

            cam_display = cv2.resize(frame, (core.width, WIN_HEIGHT))
            text_panel = build_text_panel(data["text_lines"], width=TEXT_PANEL_WIDTH, height=WIN_HEIGHT)
            if text_panel.shape[0] != WIN_HEIGHT:
                text_panel = cv2.resize(text_panel, (TEXT_PANEL_WIDTH, WIN_HEIGHT))
            combined = np.hstack([cam_display, text_panel])
            cv2.imshow("Pose Overlay (video)", combined)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

            if realtime and frame_delay_sec > 0:
                elapsed = time.perf_counter() - t0
                sleep_sec = frame_delay_sec - elapsed
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
    except KeyboardInterrupt:
        pass
    finally:
        core.close()
        cv2.destroyAllWindows()
        print(f"Log written to: {core.log_path}")


if __name__ == "__main__":
    _script_dir = Path(__file__).resolve().parent
    _default_video = _script_dir / "crouching.mp4"

    parser = argparse.ArgumentParser(
        description="Replay a recorded video with pose overlay and JSONL logging (uses cv.py core)."
    )
    parser.add_argument(
        "video",
        type=str,
        nargs="?",
        default=str(_default_video),
        help=f"Path to video file (default: {_default_video.name} in cv/)",
    )
    parser.add_argument("--log", type=str, default=None, help="Output JSONL/JSONL.gz path (default: cv/pose_log.jsonl[.gz])")
    parser.add_argument("--mirror", action="store_true", help="Mirror the video (like a live camera view)")
    parser.add_argument("--no-realtime", action="store_true", help="Process as fast as possible (no playback throttling)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = _script_dir / video_path
    video_path = str(video_path.resolve())

    run_feeder(
        video_path=video_path,
        log_path=args.log,
        mirror=args.mirror,
        realtime=not args.no_realtime,
    )
