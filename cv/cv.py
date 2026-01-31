#!/usr/bin/env python3
"""
Core CV pipeline: pose detection, smoothing, angles, text panel content, logging.
Used by cv_view.py (2D skeleton + text panel) and graphts.py (3D viewer).
"""

# Qt/OpenCV GUI: reduce platform/font noise
import os
import signal
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import json
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

if "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "xcb"
if "QT_QPA_FONTDIR" not in os.environ:
    for _path in ("/usr/share/fonts", "/usr/local/share/fonts"):
        if os.path.isdir(_path):
            os.environ["QT_QPA_FONTDIR"] = _path
            break
_ql = os.environ.get("QT_LOGGING_RULES", "")
if "qpa" not in _ql.lower():
    os.environ["QT_LOGGING_RULES"] = (_ql + ";qt.qpa.*=false").strip(";")

# ------------------ Config ------------------
DETECT_EVERY_N = 2          # Run pose detection every N frames (2 = ~2x FPS)
USE_GPU = True              # Try GPU delegate (may fallback to CPU)
LOG_BATCH_SIZE = 30         # Write log entries in batches
LOG_SAVE_MS = True          # Timestamps in milliseconds
SAVE_WORLD_COORDS = False   # Store world coords in log for 3D replay
SMOOTHING_ALPHA = 0.4       # Temporal smoothing (0=full smooth, 1=no smooth)

# Target angle ranges (deg): (min, max). Limb drawn green if in range, red otherwise.
TARGET_ANGLES = {
    "right_elbow": (80, 180),
    "left_elbow": (80, 180),
    "right_knee": (80, 180),
    "left_knee": (80, 180),
}

# MediaPipe Tasks API aliases
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
PoseLandmark = mp.tasks.vision.PoseLandmark
PoseLandmarksConnections = mp.tasks.vision.PoseLandmarksConnections
drawing_utils = mp.tasks.vision.drawing_utils


# ------------------ Helpers ------------------
def calculate_angle(a, b, c, use_3d=False):
    """Calculate angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    if use_3d and len(a) >= 3 and len(b) >= 3 and len(c) >= 3:
        vec1 = a - b
        vec2 = c - b
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 > 0 and norm2 > 0:
            cos_angle = np.clip(dot_product / (norm1 * norm2), -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180.0 / np.pi
        else:
            angle = 0.0
    else:
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
    return angle


def landmark_to_xy(landmark, width, height):
    return [landmark.x * width, landmark.y * height]


def landmark_to_xyz(landmark):
    return [landmark.x, landmark.y, landmark.z]


def smooth_landmarks(prev_list, new_list, alpha):
    """Exponential smoothing over lists of NormalizedLandmark."""
    if prev_list is None or len(prev_list) != len(new_list):
        return new_list
    from mediapipe.tasks.python.components.containers import landmark as landmark_module
    out = []
    for p, n in zip(prev_list, new_list):
        x = alpha * n.x + (1 - alpha) * p.x
        y = alpha * n.y + (1 - alpha) * p.y
        z = (getattr(n, "z", 0) or 0) * alpha + (getattr(p, "z", 0) or 0) * (1 - alpha)
        out.append(
            landmark_module.NormalizedLandmark(
                x=x, y=y, z=z,
                visibility=getattr(n, "visibility"),
                presence=getattr(n, "presence"),
            )
        )
    return out


def build_text_panel(lines, width=400, height=1200, bg_color=(40, 40, 40),
                     text_color=(220, 220, 220), line_height=32):
    """Build an image with text lines. lines can be str or (str, bgr_color)."""
    pad = 80
    dynamic_h = max(height, len(lines) * line_height + pad)
    panel = np.full((dynamic_h, width, 3), bg_color, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    x_margin = 20
    y_start = 40
    for i, item in enumerate(lines):
        line = item[0] if isinstance(item, (tuple, list)) else item
        color = item[1] if isinstance(item, (tuple, list)) and len(item) > 1 else text_color
        y = y_start + i * line_height
        cv2.putText(panel, line, (x_margin, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return panel


def get_pose_model_path(model_type="full"):
    """Return path to pose landmarker model, downloading if needed."""
    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"pose_landmarker_{model_type}.task"
    if not model_path.exists():
        url = f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{model_type}/float16/latest/pose_landmarker_{model_type}.task"
        print(f"Downloading pose landmarker {model_type} model (this may take a minute)...")
        urllib.request.urlretrieve(url, model_path)
        print("Done.")
    return str(model_path)


def create_pose_landmarker(model_type="heavy", use_gpu=USE_GPU):
    model_path = get_pose_model_path(model_type)
    base_opts = BaseOptions(model_asset_path=model_path)
    if use_gpu:
        base_opts.delegate = BaseOptions.Delegate.GPU
    options = PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    try:
        return PoseLandmarker.create_from_options(options)
    except Exception:
        if use_gpu:
            base_opts.delegate = BaseOptions.Delegate.CPU
            options = PoseLandmarkerOptions(
                base_options=base_opts,
                running_mode=VisionRunningMode.VIDEO,
                min_pose_detection_confidence=0.6,
                min_pose_presence_confidence=0.6,
                min_tracking_confidence=0.6,
            )
            return PoseLandmarker.create_from_options(options)
        raise


class PoseCore:
    """Core CV pipeline: capture, detect, smooth, compute angles, and format text lines."""
    def __init__(
        self,
        camera_id=0,
        width=1920,
        height=1200,
        model_type="heavy",
        detect_every_n=DETECT_EVERY_N,
        smoothing_alpha=SMOOTHING_ALPHA,
        log_path=None,
        save_world_coords=SAVE_WORLD_COORDS,
        log_batch_size=LOG_BATCH_SIZE,
        use_gpu=USE_GPU,
    ):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.detect_every_n = detect_every_n
        self.smoothing_alpha = smoothing_alpha
        self.save_world_coords = save_world_coords
        self.log_batch_size = log_batch_size
        self.log_buffer = []
        self.log_path = Path(log_path) if log_path else Path(__file__).resolve().parent / "pose_log.jsonl"
        self.pose = create_pose_landmarker(model_type=model_type, use_gpu=use_gpu)
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            self.pose.close()
            raise RuntimeError(f"Could not open camera {camera_id}.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame_interval_ms = 1000 / self.fps
        self.frame_count = 0
        self.last_results = None
        self.last_smoothed_lm = None

    def close(self):
        if self.cap:
            self.cap.release()
        if self.pose:
            self.pose.close()
        self.flush_log_buffer()

    def flush_log_buffer(self):
        if not self.log_buffer:
            return
        with open(self.log_path, "a", encoding="utf-8") as f:
            for entry in self.log_buffer:
                f.write(json.dumps(entry) + "\n")
        self.log_buffer.clear()

    def _build_text_lines(self, lm, lm_world, w, h):
        """Build text lines and angle values from landmarks."""
        # Visibility thresholds
        VISIBILITY_THRESHOLD = 0.5
        WRIST_VISIBILITY_MIN = 0.7
        LEG_VISIBILITY_MIN = 0.65

        def wrist_below_elbow(lm_norm, side):
            if side == "left":
                return lm_norm[PoseLandmark.LEFT_WRIST].y > lm_norm[PoseLandmark.LEFT_ELBOW].y
            return lm_norm[PoseLandmark.RIGHT_WRIST].y > lm_norm[PoseLandmark.RIGHT_ELBOW].y

        def ankle_below_knee(lm_norm, side):
            if side == "left":
                return lm_norm[PoseLandmark.LEFT_ANKLE].y > lm_norm[PoseLandmark.LEFT_KNEE].y
            return lm_norm[PoseLandmark.RIGHT_ANKLE].y > lm_norm[PoseLandmark.RIGHT_KNEE].y

        def leg_visibility_ok(lm_norm, side):
            if side == "left":
                v_hip = getattr(lm_norm[PoseLandmark.LEFT_HIP], "visibility", None) or 0
                v_knee = getattr(lm_norm[PoseLandmark.LEFT_KNEE], "visibility", None) or 0
                v_ankle = getattr(lm_norm[PoseLandmark.LEFT_ANKLE], "visibility", None) or 0
            else:
                v_hip = getattr(lm_norm[PoseLandmark.RIGHT_HIP], "visibility", None) or 0
                v_knee = getattr(lm_norm[PoseLandmark.RIGHT_KNEE], "visibility", None) or 0
                v_ankle = getattr(lm_norm[PoseLandmark.RIGHT_ANKLE], "visibility", None) or 0
            return (
                v_hip >= LEG_VISIBILITY_MIN
                and v_knee >= LEG_VISIBILITY_MIN
                and v_ankle >= LEG_VISIBILITY_MIN
                and ankle_below_knee(lm_norm, side)
            )

        # Angles using world coords for better accuracy
        angle_r_elbow = angle_l_elbow = angle_r_knee = angle_l_knee = None
        if (lm_world[PoseLandmark.RIGHT_SHOULDER].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.RIGHT_ELBOW].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.RIGHT_WRIST].visibility > VISIBILITY_THRESHOLD):
            shoulder_r = landmark_to_xyz(lm_world[PoseLandmark.RIGHT_SHOULDER])
            elbow_r = landmark_to_xyz(lm_world[PoseLandmark.RIGHT_ELBOW])
            wrist_r = landmark_to_xyz(lm_world[PoseLandmark.RIGHT_WRIST])
            angle_r_elbow = calculate_angle(shoulder_r, elbow_r, wrist_r, use_3d=True)

        if (lm_world[PoseLandmark.LEFT_SHOULDER].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.LEFT_ELBOW].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.LEFT_WRIST].visibility > VISIBILITY_THRESHOLD):
            shoulder_l = landmark_to_xyz(lm_world[PoseLandmark.LEFT_SHOULDER])
            elbow_l = landmark_to_xyz(lm_world[PoseLandmark.LEFT_ELBOW])
            wrist_l = landmark_to_xyz(lm_world[PoseLandmark.LEFT_WRIST])
            angle_l_elbow = calculate_angle(shoulder_l, elbow_l, wrist_l, use_3d=True)

        if (lm_world[PoseLandmark.RIGHT_HIP].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.RIGHT_KNEE].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.RIGHT_ANKLE].visibility > VISIBILITY_THRESHOLD):
            hip_r = landmark_to_xyz(lm_world[PoseLandmark.RIGHT_HIP])
            knee_r = landmark_to_xyz(lm_world[PoseLandmark.RIGHT_KNEE])
            ankle_r = landmark_to_xyz(lm_world[PoseLandmark.RIGHT_ANKLE])
            angle_r_knee = calculate_angle(hip_r, knee_r, ankle_r, use_3d=True)

        if (lm_world[PoseLandmark.LEFT_HIP].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.LEFT_KNEE].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.LEFT_ANKLE].visibility > VISIBILITY_THRESHOLD):
            hip_l = landmark_to_xyz(lm_world[PoseLandmark.LEFT_HIP])
            knee_l = landmark_to_xyz(lm_world[PoseLandmark.LEFT_KNEE])
            ankle_l = landmark_to_xyz(lm_world[PoseLandmark.LEFT_ANKLE])
            angle_l_knee = calculate_angle(hip_l, knee_l, ankle_l, use_3d=True)

        # Color limbs by target range: green = in range, red = out
        limb_connections = {
            (12, 14): "right_elbow", (14, 16): "right_elbow",
            (11, 13): "left_elbow", (13, 15): "left_elbow",
            (24, 26): "right_knee", (26, 28): "right_knee",
            (23, 25): "left_knee", (25, 27): "left_knee",
        }
        angle_by_joint = {
            "right_elbow": angle_r_elbow, "left_elbow": angle_l_elbow,
            "right_knee": angle_r_knee, "left_knee": angle_l_knee
        }
        default_spec = drawing_utils.DrawingSpec(color=(224, 224, 224))
        connection_spec = {}
        for conn in PoseLandmarksConnections.POSE_LANDMARKS:
            key = (conn.start, conn.end)
            if key in limb_connections:
                joint = limb_connections[key]
                angle = angle_by_joint.get(joint)
                lo, hi = TARGET_ANGLES.get(joint, (0, 180))
                if angle is not None and lo <= angle <= hi:
                    connection_spec[key] = drawing_utils.DrawingSpec(color=drawing_utils.GREEN_COLOR)
                else:
                    connection_spec[key] = drawing_utils.DrawingSpec(color=drawing_utils.RED_COLOR)
            else:
                connection_spec[key] = default_spec

        # Text panel lines
        def fmt_angle(angle):
            return f"{angle:.1f} deg" if angle is not None else "N/A"

        left_wrist_ok = (lm[PoseLandmark.LEFT_WRIST].visibility or 0) >= WRIST_VISIBILITY_MIN and wrist_below_elbow(lm, "left")
        right_wrist_ok = (lm[PoseLandmark.RIGHT_WRIST].visibility or 0) >= WRIST_VISIBILITY_MIN and wrist_below_elbow(lm, "right")
        wrist_l_px = landmark_to_xy(lm[PoseLandmark.LEFT_WRIST], w, h) if left_wrist_ok else (0, 0)
        wrist_r_px = landmark_to_xy(lm[PoseLandmark.RIGHT_WRIST], w, h) if right_wrist_ok else (0, 0)
        left_hand_str = f"elbow {fmt_angle(angle_l_elbow)}, wrist ({int(wrist_l_px[0])},{int(wrist_l_px[1])})" if left_wrist_ok else f"elbow {fmt_angle(angle_l_elbow)}, wrist (low conf)"
        right_hand_str = f"elbow {fmt_angle(angle_r_elbow)}, wrist ({int(wrist_r_px[0])},{int(wrist_r_px[1])})" if right_wrist_ok else f"elbow {fmt_angle(angle_r_elbow)}, wrist (low conf)"

        left_leg_ok = leg_visibility_ok(lm, "left")
        right_leg_ok = leg_visibility_ok(lm, "right")
        ankle_l_px = landmark_to_xy(lm[PoseLandmark.LEFT_ANKLE], w, h) if left_leg_ok else (0, 0)
        ankle_r_px = landmark_to_xy(lm[PoseLandmark.RIGHT_ANKLE], w, h) if right_leg_ok else (0, 0)
        left_leg_str = f"knee {fmt_angle(angle_l_knee)}, ankle ({int(ankle_l_px[0])},{int(ankle_l_px[1])})" if left_leg_ok else "not in frame"
        right_leg_str = f"knee {fmt_angle(angle_r_knee)}, ankle ({int(ankle_r_px[0])},{int(ankle_r_px[1])})" if right_leg_ok else "not in frame"

        def in_range(angle, joint):
            lo, hi = TARGET_ANGLES.get(joint, (0, 180))
            return angle is not None and lo <= angle <= hi
        green_bgr = (0, 200, 0)
        red_bgr = (0, 0, 255)
        def line_with_status(angle, joint, label, val_str):
            lo, hi = TARGET_ANGLES.get(joint, (0, 180))
            status = "OK" if in_range(angle, joint) else "OUT"
            color = green_bgr if in_range(angle, joint) else red_bgr
            return (f"{label}: {val_str} ({lo}-{hi}) {status}", color)

        text_lines = [
            "--- Pose Output ---",
            "",
            line_with_status(angle_l_elbow, "left_elbow", "Left hand ", left_hand_str),
            line_with_status(angle_r_elbow, "right_elbow", "Right hand", right_hand_str),
            "",
            line_with_status(angle_l_knee, "left_knee", "Left leg  ", left_leg_str),
            line_with_status(angle_r_knee, "right_knee", "Right leg ", right_leg_str),
        ]
        angles = {
            "right_elbow": angle_r_elbow,
            "left_elbow": angle_l_elbow,
            "right_knee": angle_r_knee,
            "left_knee": angle_l_knee,
        }
        return text_lines, connection_spec, angles

    def step(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        h, w, _ = frame.shape

        if self.frame_count % self.detect_every_n == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(self.frame_count * self.frame_interval_ms)
            self.last_results = self.pose.detect_for_video(mp_image, timestamp_ms)

        results = self.last_results
        text_lines = [
            "--- Pose Output ---", "",
            "Left hand:  --",
            "Right hand: --", "",
            "Left leg:   --",
            "Right leg:  --",
        ]
        connection_spec = None
        lm = lm_world = None
        angles = None

        if results and results.pose_landmarks and results.pose_world_landmarks:
            lm_raw = results.pose_landmarks[0]
            lm = smooth_landmarks(self.last_smoothed_lm, lm_raw, self.smoothing_alpha)
            self.last_smoothed_lm = lm
            lm_world = results.pose_world_landmarks[0]
            text_lines, connection_spec, angles = self._build_text_lines(lm, lm_world, w, h)

            # Log batch
            ts_ms = int(self.frame_count * self.frame_interval_ms)
            entry = {
                "timestamp_ms": ts_ms,
                "frame_index": self.frame_count,
                "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            }
            if angles:
                for k, v in angles.items():
                    entry[k] = round(v, 1) if v is not None else None
            if self.save_world_coords and lm_world:
                entry["world_landmarks"] = [[getattr(p, "x"), getattr(p, "y"), getattr(p, "z")] for p in lm_world]
            if angles and any(v is not None for v in angles.values()):
                self.log_buffer.append(entry)
                if len(self.log_buffer) >= self.log_batch_size:
                    self.flush_log_buffer()

        self.frame_count += 1
        return {
            "frame": frame,
            "text_lines": text_lines,
            "connection_spec": connection_spec,
            "landmarks": lm,
            "world_landmarks": lm_world,
        }


def draw_skeleton(frame, lm, connection_spec=None):
    if lm is None:
        return
    drawing_utils.draw_landmarks(
        frame,
        lm,
        PoseLandmarksConnections.POSE_LANDMARKS,
        connection_drawing_spec=connection_spec,
    )


def install_ctrl_c(handler):
    signal.signal(signal.SIGINT, handler)

