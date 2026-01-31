#!/usr/bin/env python3
"""
2D GUI viewer: camera + skeleton overlay + text panel.
Builds entirely on cv.py.
"""
import argparse
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


def run_view(camera_id=0, width=1920, height=1200):
    """All pipeline options (model, detect_every_n, detect_scale, etc.) come from cv.py config."""
    try:
        core = PoseCore(camera_id=camera_id, width=width, height=height)
    except RuntimeError as exc:
        print(exc)
        return

    TEXT_PANEL_WIDTH = 420
    WIN_HEIGHT = height
    WIN_WIDTH = width + TEXT_PANEL_WIDTH
    cv2.namedWindow("Skeleton Overlay", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Skeleton Overlay", WIN_WIDTH, WIN_HEIGHT)

    exit_requested = False

    def _on_sigint(*_):
        nonlocal exit_requested
        exit_requested = True

    install_ctrl_c(_on_sigint)

    try:
        while not exit_requested:
            data = core.step()
            if data is None:
                break

            frame = data["frame"]
            frame = cv2.flip(frame, 1)  # mirror view so it looks natural to the user
            lm_mirrored = flip_landmarks_x(data["landmarks"])  # align skeleton with mirrored frame
            draw_skeleton(frame, lm_mirrored, data["connection_spec"])
            if data.get("alert_red"):
                red = np.full_like(frame, (0, 0, 255))
                frame = cv2.addWeighted(frame, 1.0 - ELBOW_ALERT_RED_ALPHA, red, ELBOW_ALERT_RED_ALPHA, 0)

            cam_display = cv2.resize(frame, (width, WIN_HEIGHT))
            text_panel = build_text_panel(data["text_lines"], width=TEXT_PANEL_WIDTH, height=WIN_HEIGHT)
            if text_panel.shape[0] != WIN_HEIGHT:
                text_panel = cv2.resize(text_panel, (TEXT_PANEL_WIDTH, WIN_HEIGHT))
            combined = np.hstack([cam_display, text_panel])
            cv2.imshow("Skeleton Overlay", combined)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    except KeyboardInterrupt:
        pass
    finally:
        core.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pose GUI (all options from cv.py / config.yaml).")
    parser.add_argument("--camera", type=int, default=0, help="Camera device id")
    parser.add_argument("--width", type=int, default=1920, help="Capture width")
    parser.add_argument("--height", type=int, default=1200, help="Capture height")
    args = parser.parse_args()
    run_view(camera_id=args.camera, width=args.width, height=args.height)
