#!/usr/bin/env python3
"""
Core CV pipeline: pose detection, smoothing, angles, text panel content, logging.
Used by cv_view.py (2D skeleton + text panel) and graphts.py (3D viewer).
"""

# Qt/OpenCV GUI: reduce platform/font noise (set before importing cv2)
import os
import signal
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import json
import urllib.request

if "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "xcb"
if "QT_QPA_FONTDIR" not in os.environ:
    for _path in (
        "/usr/share/fonts/truetype",
        "/usr/share/fonts/TTF",
        "/usr/share/fonts",
        "/usr/local/share/fonts",
    ):
        if os.path.isdir(_path):
            os.environ["QT_QPA_FONTDIR"] = _path
            break
_ql = os.environ.get("QT_LOGGING_RULES", "")
for _tag in ("qpa", "gui", "font"):
    if _tag not in _ql.lower():
        os.environ["QT_LOGGING_RULES"] = (
            (_ql.rstrip(";") + ";qt.qpa.*=false;qt.gui.*=false").strip(";")
        )
        break

# Optional: suppress known MediaPipe/absl and Qt font warnings (set CV_QUIET_WARNINGS=1)
if os.environ.get("CV_QUIET_WARNINGS", "").lower() in ("1", "true", "yes"):
    _stderr_orig = sys.stderr
    _skip_phrases = (
        "inference_feedback_manager",
        "QFont::fromString",
        "QFontDatabase:",
        "Note that Qt no longer ships fonts",
        "landmark_projection_calculator",
        "WARNING: All log messages before absl",
    )
    class _FilteredStderr:
        def write(self, s):
            if s and not any(p in s for p in _skip_phrases):
                _stderr_orig.write(s)
        def flush(self):
            _stderr_orig.flush()
    sys.stderr = _FilteredStderr()

import cv2
import mediapipe as mp
import numpy as np

# ------------------ Config (loaded from cv/config.yaml) ------------------
_CV_DIR = Path(__file__).resolve().parent
_CONFIG_CANDIDATES = (_CV_DIR / "config.yaml", _CV_DIR / "config.yml", _CV_DIR / "config.json")

_DEFAULT_CONFIG = {
    "model_type": "heavy",
    "detect_every_n": 2,
    "use_gpu": True,
    "detect_scale": 1.0,
    "preview_width": 1920,
    "preview_height": 1080,
    "cv_width": 1280,
    "cv_height": 720,
    "log_batch_size": 30,
    "log_update_interval_ms": 100,
    "log_use_gzip": True,
    "log_min_visibility": 0.05,
    "log_save_ms": True,
    "save_world_coords": False,
    "smoothing_alpha": 0.4,
    "pose_hold_frames": 6,
    "angle_hold_frames": 6,
    "arm_angle_offset": 0,
    "elbow_angle_mode": "max",
    "elbow_auto_depth_m": 0.15,
    "elbow_alert_angle": 90,
    "elbow_alert_tolerance": 5,
    "elbow_alert_red_alpha": 0.35,
    "target_angles": {
        "right_elbow": [80, 180],
        "left_elbow": [80, 180],
        "right_knee": [80, 180],
        "left_knee": [80, 180],
    },
    "pose_landmarker": {
        "min_pose_detection_confidence": 0.6,
        "min_pose_presence_confidence": 0.6,
        "min_tracking_confidence": 0.6,
    },
}


def load_config(path=None):
    """Load config from cv/config.yaml (or .yml / config.json). YAML natively supports # comments. Missing keys or file → use defaults."""
    import yaml
    cfg = json.loads(json.dumps(_DEFAULT_CONFIG))  # deep copy
    candidates = [Path(path) if not isinstance(path, Path) else path] if path is not None else list(_CONFIG_CANDIDATES)
    for p in candidates:
        if p is None or not p.exists():
            continue
        try:
            with open(p, encoding="utf-8") as f:
                raw = f.read()
            suf = p.suffix.lower()
            if suf in (".yaml", ".yml"):
                data = yaml.safe_load(raw)
            else:
                data = json.loads(raw)
            if not isinstance(data, dict):
                continue
            for k, v in data.items():
                if k == "target_angles" and isinstance(v, dict):
                    existing = cfg.get("target_angles") or {}
                    for j, r in v.items():
                        existing[j] = tuple(r) if isinstance(r, list) else r
                    cfg["target_angles"] = existing
                elif k == "pose_landmarker" and isinstance(v, dict):
                    cfg.setdefault("pose_landmarker", {}).update(v)
                else:
                    cfg[k] = v
            break
        except (json.JSONDecodeError, yaml.YAMLError, OSError):
            continue
    # Ensure target_angles values are tuples
    if "target_angles" in cfg and isinstance(cfg["target_angles"], dict):
        cfg["target_angles"] = {
            j: (tuple(r) if isinstance(r, list) else r)
            for j, r in cfg["target_angles"].items()
        }
    return cfg


_CONFIG = load_config()
MODEL_TYPE = _CONFIG.get("model_type", "heavy")
DETECT_EVERY_N = _CONFIG["detect_every_n"]
USE_GPU = _CONFIG["use_gpu"]
DETECT_SCALE = _CONFIG["detect_scale"]
PREVIEW_WIDTH = int(_CONFIG.get("preview_width", 1920))
PREVIEW_HEIGHT = int(_CONFIG.get("preview_height", 1080))
CV_WIDTH = int(_CONFIG.get("cv_width", 1280))
CV_HEIGHT = int(_CONFIG.get("cv_height", 720))
LOG_BATCH_SIZE = _CONFIG["log_batch_size"]
LOG_UPDATE_INTERVAL_MS = int(_CONFIG.get("log_update_interval_ms", 100))
LOG_USE_GZIP = bool(_CONFIG.get("log_use_gzip", True))
LOG_MIN_VISIBILITY = float(_CONFIG.get("log_min_visibility", 0.05))
LOG_SAVE_MS = _CONFIG["log_save_ms"]
SAVE_WORLD_COORDS = _CONFIG["save_world_coords"]
SMOOTHING_ALPHA = _CONFIG["smoothing_alpha"]
POSE_HOLD_FRAMES = int(_CONFIG.get("pose_hold_frames", 6))
ANGLE_HOLD_FRAMES = int(_CONFIG.get("angle_hold_frames", 6))
TARGET_ANGLES = _CONFIG["target_angles"]
ARM_ANGLE_OFFSET = _CONFIG.get("arm_angle_offset", 0)
ELBOW_ANGLE_MODE = _CONFIG.get("elbow_angle_mode", "max")
ELBOW_AUTO_DEPTH_M = float(_CONFIG.get("elbow_auto_depth_m", 0.15))
ELBOW_ALERT_ANGLE = float(_CONFIG.get("elbow_alert_angle", 90))
ELBOW_ALERT_TOL = float(_CONFIG.get("elbow_alert_tolerance", 5))
ELBOW_ALERT_RED_ALPHA = float(_CONFIG.get("elbow_alert_red_alpha", 0.35))
_POSE_LANDMARKER_CFG = _CONFIG.get("pose_landmarker", _DEFAULT_CONFIG["pose_landmarker"])

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
    """Angle at vertex b (a–b–c) in degrees [0, 180], using atan2(norm(cross), dot).
    For 3D, use true 3D vectors; for 2D, project onto image plane (x, y only).
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    
    if not use_3d:
        # 2D: only use x, y
        a = a[:2]
        b = b[:2]
        c = c[:2]
    
    vec1 = a - b
    vec2 = c - b
    
    if use_3d:
        # 3D angle using cross product magnitude and dot product
        cross = np.cross(vec1, vec2)
        cross_mag = np.linalg.norm(cross)
        dot = np.dot(vec1, vec2)
        return float(np.degrees(np.arctan2(cross_mag, dot)))
    else:
        # 2D angle (image plane)
        cross = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        dot = np.dot(vec1, vec2)
        return float(np.degrees(np.arctan2(abs(cross), dot)))


def angle_between_vectors(vec1, vec2):
    """Angle between two vectors in degrees [0, 180]."""
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None
    cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def signed_angle_2d(vec1, vec2):
    """Signed angle from vec1 to vec2 in degrees [-180, 180]."""
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None
    v1 = v1 / n1
    v2 = v2 / n2
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    dot = np.dot(v1, v2)
    return float(np.degrees(np.arctan2(cross, dot)))


def landmark_to_xy(landmark, width, height):
    return [landmark.x * width, landmark.y * height]


def landmark_to_norm_xy(landmark):
    """Normalized image-plane [x, y] for angle-in-view-plane (e.g. elbow)."""
    return [landmark.x, landmark.y]


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


def get_pose_model_path(model_type="lite"):
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
    pm = _POSE_LANDMARKER_CFG
    model_path = get_pose_model_path(model_type)
    base_opts = BaseOptions(model_asset_path=model_path)
    if use_gpu:
        base_opts.delegate = BaseOptions.Delegate.GPU
    options = PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=pm.get("min_pose_detection_confidence", 0.6),
        min_pose_presence_confidence=pm.get("min_pose_presence_confidence", 0.6),
        min_tracking_confidence=pm.get("min_tracking_confidence", 0.6),
    )
    try:
        return PoseLandmarker.create_from_options(options)
    except Exception:
        if use_gpu:
            base_opts.delegate = BaseOptions.Delegate.CPU
            options = PoseLandmarkerOptions(
                base_options=base_opts,
                running_mode=VisionRunningMode.VIDEO,
                min_pose_detection_confidence=pm.get("min_pose_detection_confidence", 0.6),
                min_pose_presence_confidence=pm.get("min_pose_presence_confidence", 0.6),
                min_tracking_confidence=pm.get("min_tracking_confidence", 0.6),
            )
            return PoseLandmarker.create_from_options(options)
        raise


class PoseCore:
    """Core CV pipeline: capture, detect, smooth, compute angles, and format text lines.
    Supports live camera (camera_id) or recorded video (video_path).
    """
    def __init__(
        self,
        camera_id=0,
        video_path=None,
        width=PREVIEW_WIDTH,
        height=PREVIEW_HEIGHT,
        model_type=MODEL_TYPE,
        detect_every_n=DETECT_EVERY_N,
        detect_scale=DETECT_SCALE,
        cv_width=CV_WIDTH,
        cv_height=CV_HEIGHT,
        smoothing_alpha=SMOOTHING_ALPHA,
        log_path=None,
        save_world_coords=SAVE_WORLD_COORDS,
        log_batch_size=LOG_BATCH_SIZE,
        use_gpu=USE_GPU,
    ):
        self.video_path = video_path
        self.use_video_time = video_path is not None
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.detect_every_n = detect_every_n
        self.detect_scale = max(0.25, min(1.0, float(detect_scale)))
        self.cv_width = int(cv_width)
        self.cv_height = int(cv_height)
        self.smoothing_alpha = smoothing_alpha
        self.save_world_coords = save_world_coords
        self.log_batch_size = log_batch_size
        self.log_update_interval_ms = LOG_UPDATE_INTERVAL_MS
        self.log_use_gzip = LOG_USE_GZIP
        self.log_min_visibility = LOG_MIN_VISIBILITY
        self.log_buffer = []
        default_log = Path(__file__).resolve().parent / ("pose_log.jsonl.gz" if LOG_USE_GZIP else "pose_log.jsonl")
        self.log_path = Path(log_path) if log_path else default_log
        self.last_log_time_ms = 0
        self.pose = create_pose_landmarker(model_type=model_type, use_gpu=use_gpu)
        if video_path is not None:
            self.cap = cv2.VideoCapture(str(video_path))
            if not self.cap.isOpened():
                self.pose.close()
                raise RuntimeError(f"Could not open video: {video_path}")
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
        else:
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
        self._last_good_results = None
        self._pose_hold_frames = POSE_HOLD_FRAMES
        self._pose_hold_count = 0
        self.last_smoothed_lm = None
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.session_start_ms = int(datetime.now().timestamp() * 1000)
        self._angle_hold_frames = ANGLE_HOLD_FRAMES
        self._angle_hold = {}

    def close(self):
        if self.cap:
            self.cap.release()
        if self.pose:
            self.pose.close()
        self.flush_log_buffer()

    def flush_log_buffer(self):
        if not self.log_buffer:
            return
        if self.log_use_gzip:
            import gzip
            with gzip.open(self.log_path, "at", encoding="utf-8") as f:
                for entry in self.log_buffer:
                    f.write(json.dumps(entry, separators=(',', ':')) + "\n")
        else:
            with open(self.log_path, "a", encoding="utf-8") as f:
                for entry in self.log_buffer:
                    f.write(json.dumps(entry, separators=(',', ':')) + "\n")
        self.log_buffer.clear()

    def _stabilize_angles(self, angles):
        """Hold last valid angle values for a few frames to reduce flicker."""
        stabilized = {}
        for key, value in angles.items():
            if value is None:
                cached = self._angle_hold.get(key)
                if cached is not None:
                    cached_val, age = cached
                    if age < self._angle_hold_frames:
                        stabilized[key] = cached_val
                        self._angle_hold[key] = (cached_val, age + 1)
                    else:
                        stabilized[key] = None
                        self._angle_hold.pop(key, None)
                else:
                    stabilized[key] = None
            else:
                stabilized[key] = value
                self._angle_hold[key] = (value, 0)
        return stabilized

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

        def select_elbow_angle(a3d, a2d, depth_delta=None, extension_angle=None, auto_use_3d=None):
            if a3d is None and a2d is None:
                return None
            mode = str(ELBOW_ANGLE_MODE).lower()
            if mode == "extension":
                return extension_angle if extension_angle is not None else (a3d if a3d is not None else a2d)
            if mode == "auto":
                if auto_use_3d is True:
                    return a3d if a3d is not None else a2d
                if auto_use_3d is False:
                    return a2d if a2d is not None else a3d
                if depth_delta is not None and abs(depth_delta) >= ELBOW_AUTO_DEPTH_M:
                    return a3d if a3d is not None else a2d
                return a2d if a2d is not None else a3d
            if mode == "3d":
                return a3d if a3d is not None else a2d
            if mode == "2d":
                return a2d if a2d is not None else a3d
            if mode == "avg":
                vals = [v for v in (a3d, a2d) if v is not None]
                return sum(vals) / len(vals) if vals else None
            if mode == "max":
                vals = [v for v in (a3d, a2d) if v is not None]
                return max(vals) if vals else None
            return a3d if a3d is not None else a2d

        # Elbow angles: compute 3D (world) and 2D (image-plane), then select by config.
        angle_r_elbow = angle_l_elbow = angle_r_knee = angle_l_knee = None
        if (lm_world[PoseLandmark.RIGHT_SHOULDER].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.RIGHT_ELBOW].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.RIGHT_WRIST].visibility > VISIBILITY_THRESHOLD):
            
            shoulder_r = landmark_to_xyz(lm_world[PoseLandmark.RIGHT_SHOULDER])
            elbow_r = landmark_to_xyz(lm_world[PoseLandmark.RIGHT_ELBOW])
            wrist_r = landmark_to_xyz(lm_world[PoseLandmark.RIGHT_WRIST])
            
            angle_r_elbow_3d = calculate_angle(shoulder_r, elbow_r, wrist_r, use_3d=True)
            depth_r = wrist_r[2] - shoulder_r[2]
            se_r = np.linalg.norm(np.array(shoulder_r) - np.array(elbow_r))
            ew_r = np.linalg.norm(np.array(elbow_r) - np.array(wrist_r))
            sw_r = np.linalg.norm(np.array(shoulder_r) - np.array(wrist_r))
            max_r = se_r + ew_r if se_r + ew_r > 1e-6 else None
            extension_r = 180.0 * (sw_r / max_r) if max_r else None
        else:
            angle_r_elbow_3d = None
            depth_r = None
            extension_r = None
        if (
            (lm[PoseLandmark.RIGHT_SHOULDER].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.RIGHT_ELBOW].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.RIGHT_WRIST].visibility or 0) > VISIBILITY_THRESHOLD
        ):
            shoulder_r_2d = landmark_to_norm_xy(lm[PoseLandmark.RIGHT_SHOULDER])
            elbow_r_2d = landmark_to_norm_xy(lm[PoseLandmark.RIGHT_ELBOW])
            wrist_r_2d = landmark_to_norm_xy(lm[PoseLandmark.RIGHT_WRIST])
            angle_r_elbow_2d = calculate_angle(shoulder_r_2d, elbow_r_2d, wrist_r_2d, use_3d=False)
        else:
            angle_r_elbow_2d = None
        if extension_r is None and angle_r_elbow_2d is not None:
            se_r = np.linalg.norm(np.array(shoulder_r_2d) - np.array(elbow_r_2d))
            ew_r = np.linalg.norm(np.array(elbow_r_2d) - np.array(wrist_r_2d))
            sw_r = np.linalg.norm(np.array(shoulder_r_2d) - np.array(wrist_r_2d))
            max_r = se_r + ew_r if se_r + ew_r > 1e-6 else None
            extension_r = 180.0 * (sw_r / max_r) if max_r else None
        if (lm_world[PoseLandmark.LEFT_SHOULDER].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.LEFT_ELBOW].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.LEFT_WRIST].visibility > VISIBILITY_THRESHOLD):
            shoulder_l = landmark_to_xyz(lm_world[PoseLandmark.LEFT_SHOULDER])
            elbow_l = landmark_to_xyz(lm_world[PoseLandmark.LEFT_ELBOW])
            wrist_l = landmark_to_xyz(lm_world[PoseLandmark.LEFT_WRIST])
            angle_l_elbow_3d = calculate_angle(shoulder_l, elbow_l, wrist_l, use_3d=True)
            depth_l = wrist_l[2] - shoulder_l[2]
            se_l = np.linalg.norm(np.array(shoulder_l) - np.array(elbow_l))
            ew_l = np.linalg.norm(np.array(elbow_l) - np.array(wrist_l))
            sw_l = np.linalg.norm(np.array(shoulder_l) - np.array(wrist_l))
            max_l = se_l + ew_l if se_l + ew_l > 1e-6 else None
            extension_l = 180.0 * (sw_l / max_l) if max_l else None
        else:
            angle_l_elbow_3d = None
            depth_l = None
            extension_l = None
        auto_use_3d = None
        if str(ELBOW_ANGLE_MODE).lower() == "auto":
            depth_candidates = [d for d in (depth_r, depth_l) if d is not None]
            if depth_candidates:
                auto_use_3d = max(abs(d) for d in depth_candidates) >= ELBOW_AUTO_DEPTH_M
        if (
            (lm[PoseLandmark.LEFT_SHOULDER].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.LEFT_ELBOW].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.LEFT_WRIST].visibility or 0) > VISIBILITY_THRESHOLD
        ):
            shoulder_l_2d = landmark_to_norm_xy(lm[PoseLandmark.LEFT_SHOULDER])
            elbow_l_2d = landmark_to_norm_xy(lm[PoseLandmark.LEFT_ELBOW])
            wrist_l_2d = landmark_to_norm_xy(lm[PoseLandmark.LEFT_WRIST])
            angle_l_elbow_2d = calculate_angle(shoulder_l_2d, elbow_l_2d, wrist_l_2d, use_3d=False)
        else:
            angle_l_elbow_2d = None
        if extension_l is None and angle_l_elbow_2d is not None:
            se_l = np.linalg.norm(np.array(shoulder_l_2d) - np.array(elbow_l_2d))
            ew_l = np.linalg.norm(np.array(elbow_l_2d) - np.array(wrist_l_2d))
            sw_l = np.linalg.norm(np.array(shoulder_l_2d) - np.array(wrist_l_2d))
            max_l = se_l + ew_l if se_l + ew_l > 1e-6 else None
            extension_l = 180.0 * (sw_l / max_l) if max_l else None
        angle_r_elbow = select_elbow_angle(
            angle_r_elbow_3d,
            angle_r_elbow_2d,
            depth_r,
            extension_r,
            auto_use_3d,
        )
        if angle_r_elbow is not None:
            angle_r_elbow = angle_r_elbow + ARM_ANGLE_OFFSET

        angle_l_elbow = select_elbow_angle(
            angle_l_elbow_3d,
            angle_l_elbow_2d,
            depth_l,
            extension_l,
            auto_use_3d,
        )
        if angle_l_elbow is not None:
            angle_l_elbow = angle_l_elbow + ARM_ANGLE_OFFSET

        angle_r_knee_3d = angle_r_knee_2d = None
        if (lm_world[PoseLandmark.RIGHT_HIP].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.RIGHT_KNEE].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.RIGHT_ANKLE].visibility > VISIBILITY_THRESHOLD):
            hip_r = landmark_to_xyz(lm_world[PoseLandmark.RIGHT_HIP])
            knee_r = landmark_to_xyz(lm_world[PoseLandmark.RIGHT_KNEE])
            ankle_r = landmark_to_xyz(lm_world[PoseLandmark.RIGHT_ANKLE])
            angle_r_knee_3d = calculate_angle(hip_r, knee_r, ankle_r, use_3d=True)
        
        if (
            (lm[PoseLandmark.RIGHT_HIP].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.RIGHT_KNEE].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.RIGHT_ANKLE].visibility or 0) > VISIBILITY_THRESHOLD
        ):
            hip_r_2d = landmark_to_norm_xy(lm[PoseLandmark.RIGHT_HIP])
            knee_r_2d = landmark_to_norm_xy(lm[PoseLandmark.RIGHT_KNEE])
            ankle_r_2d = landmark_to_norm_xy(lm[PoseLandmark.RIGHT_ANKLE])
            angle_r_knee_2d = calculate_angle(hip_r_2d, knee_r_2d, ankle_r_2d, use_3d=False)
        
        angle_r_knee = angle_r_knee_3d if angle_r_knee_3d is not None else angle_r_knee_2d

        angle_l_knee_3d = angle_l_knee_2d = None
        if (lm_world[PoseLandmark.LEFT_HIP].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.LEFT_KNEE].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.LEFT_ANKLE].visibility > VISIBILITY_THRESHOLD):
            hip_l = landmark_to_xyz(lm_world[PoseLandmark.LEFT_HIP])
            knee_l = landmark_to_xyz(lm_world[PoseLandmark.LEFT_KNEE])
            ankle_l = landmark_to_xyz(lm_world[PoseLandmark.LEFT_ANKLE])
            angle_l_knee_3d = calculate_angle(hip_l, knee_l, ankle_l, use_3d=True)
        
        if (
            (lm[PoseLandmark.LEFT_HIP].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.LEFT_KNEE].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.LEFT_ANKLE].visibility or 0) > VISIBILITY_THRESHOLD
        ):
            hip_l_2d = landmark_to_norm_xy(lm[PoseLandmark.LEFT_HIP])
            knee_l_2d = landmark_to_norm_xy(lm[PoseLandmark.LEFT_KNEE])
            ankle_l_2d = landmark_to_norm_xy(lm[PoseLandmark.LEFT_ANKLE])
            angle_l_knee_2d = calculate_angle(hip_l_2d, knee_l_2d, ankle_l_2d, use_3d=False)
        
        angle_l_knee = angle_l_knee_3d if angle_l_knee_3d is not None else angle_l_knee_2d

        # Shoulder angles: torso-shoulder-elbow (important for overhead press, lateral raises)
        angle_r_shoulder_3d = angle_r_shoulder_2d = None
        if (lm_world[PoseLandmark.RIGHT_HIP].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.RIGHT_SHOULDER].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.RIGHT_ELBOW].visibility > VISIBILITY_THRESHOLD):
            hip_r = landmark_to_xyz(lm_world[PoseLandmark.RIGHT_HIP])
            shoulder_r = landmark_to_xyz(lm_world[PoseLandmark.RIGHT_SHOULDER])
            elbow_r = landmark_to_xyz(lm_world[PoseLandmark.RIGHT_ELBOW])
            angle_r_shoulder_3d = calculate_angle(hip_r, shoulder_r, elbow_r, use_3d=True)
        
        if (
            (lm[PoseLandmark.RIGHT_HIP].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.RIGHT_SHOULDER].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.RIGHT_ELBOW].visibility or 0) > VISIBILITY_THRESHOLD
        ):
            hip_r_2d = landmark_to_norm_xy(lm[PoseLandmark.RIGHT_HIP])
            shoulder_r_2d = landmark_to_norm_xy(lm[PoseLandmark.RIGHT_SHOULDER])
            elbow_r_2d = landmark_to_norm_xy(lm[PoseLandmark.RIGHT_ELBOW])
            angle_r_shoulder_2d = calculate_angle(hip_r_2d, shoulder_r_2d, elbow_r_2d, use_3d=False)
        
        angle_r_shoulder = angle_r_shoulder_3d if angle_r_shoulder_3d is not None else angle_r_shoulder_2d

        angle_l_shoulder_3d = angle_l_shoulder_2d = None
        if (lm_world[PoseLandmark.LEFT_HIP].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.LEFT_SHOULDER].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.LEFT_ELBOW].visibility > VISIBILITY_THRESHOLD):
            hip_l = landmark_to_xyz(lm_world[PoseLandmark.LEFT_HIP])
            shoulder_l = landmark_to_xyz(lm_world[PoseLandmark.LEFT_SHOULDER])
            elbow_l = landmark_to_xyz(lm_world[PoseLandmark.LEFT_ELBOW])
            angle_l_shoulder_3d = calculate_angle(hip_l, shoulder_l, elbow_l, use_3d=True)
        
        if (
            (lm[PoseLandmark.LEFT_HIP].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.LEFT_SHOULDER].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.LEFT_ELBOW].visibility or 0) > VISIBILITY_THRESHOLD
        ):
            hip_l_2d = landmark_to_norm_xy(lm[PoseLandmark.LEFT_HIP])
            shoulder_l_2d = landmark_to_norm_xy(lm[PoseLandmark.LEFT_SHOULDER])
            elbow_l_2d = landmark_to_norm_xy(lm[PoseLandmark.LEFT_ELBOW])
            angle_l_shoulder_2d = calculate_angle(hip_l_2d, shoulder_l_2d, elbow_l_2d, use_3d=False)
        
        angle_l_shoulder = angle_l_shoulder_3d if angle_l_shoulder_3d is not None else angle_l_shoulder_2d

        # Hip angles: shoulder-hip-knee (important for squats, lunges)
        angle_r_hip_3d = angle_r_hip_2d = None
        if (lm_world[PoseLandmark.RIGHT_SHOULDER].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.RIGHT_HIP].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.RIGHT_KNEE].visibility > VISIBILITY_THRESHOLD):
            shoulder_r = landmark_to_xyz(lm_world[PoseLandmark.RIGHT_SHOULDER])
            hip_r = landmark_to_xyz(lm_world[PoseLandmark.RIGHT_HIP])
            knee_r = landmark_to_xyz(lm_world[PoseLandmark.RIGHT_KNEE])
            angle_r_hip_3d = calculate_angle(shoulder_r, hip_r, knee_r, use_3d=True)
        
        if (
            (lm[PoseLandmark.RIGHT_SHOULDER].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.RIGHT_HIP].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.RIGHT_KNEE].visibility or 0) > VISIBILITY_THRESHOLD
        ):
            shoulder_r_2d = landmark_to_norm_xy(lm[PoseLandmark.RIGHT_SHOULDER])
            hip_r_2d = landmark_to_norm_xy(lm[PoseLandmark.RIGHT_HIP])
            knee_r_2d = landmark_to_norm_xy(lm[PoseLandmark.RIGHT_KNEE])
            angle_r_hip_2d = calculate_angle(shoulder_r_2d, hip_r_2d, knee_r_2d, use_3d=False)
        
        angle_r_hip = angle_r_hip_3d if angle_r_hip_3d is not None else angle_r_hip_2d

        angle_l_hip_3d = angle_l_hip_2d = None
        if (lm_world[PoseLandmark.LEFT_SHOULDER].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.LEFT_HIP].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.LEFT_KNEE].visibility > VISIBILITY_THRESHOLD):
            shoulder_l = landmark_to_xyz(lm_world[PoseLandmark.LEFT_SHOULDER])
            hip_l = landmark_to_xyz(lm_world[PoseLandmark.LEFT_HIP])
            knee_l = landmark_to_xyz(lm_world[PoseLandmark.LEFT_KNEE])
            angle_l_hip_3d = calculate_angle(shoulder_l, hip_l, knee_l, use_3d=True)
        
        if (
            (lm[PoseLandmark.LEFT_SHOULDER].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.LEFT_HIP].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.LEFT_KNEE].visibility or 0) > VISIBILITY_THRESHOLD
        ):
            shoulder_l_2d = landmark_to_norm_xy(lm[PoseLandmark.LEFT_SHOULDER])
            hip_l_2d = landmark_to_norm_xy(lm[PoseLandmark.LEFT_HIP])
            knee_l_2d = landmark_to_norm_xy(lm[PoseLandmark.LEFT_KNEE])
            angle_l_hip_2d = calculate_angle(shoulder_l_2d, hip_l_2d, knee_l_2d, use_3d=False)
        
        angle_l_hip = angle_l_hip_3d if angle_l_hip_3d is not None else angle_l_hip_2d

        # Ankle angles: angle between leg (knee->ankle) and foot (ankle->foot_index)
        # 0° means foot is straight in line with the leg.
        angle_r_ankle_3d = angle_r_ankle_2d = None
        if (lm_world[PoseLandmark.RIGHT_KNEE].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.RIGHT_ANKLE].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.RIGHT_FOOT_INDEX].visibility > VISIBILITY_THRESHOLD):
            knee_r = landmark_to_xyz(lm_world[PoseLandmark.RIGHT_KNEE])
            ankle_r = landmark_to_xyz(lm_world[PoseLandmark.RIGHT_ANKLE])
            foot_r = landmark_to_xyz(lm_world[PoseLandmark.RIGHT_FOOT_INDEX])
            leg_vec_r = np.array(ankle_r) - np.array(knee_r)
            foot_vec_r = np.array(foot_r) - np.array(ankle_r)
            angle_r_ankle_3d = angle_between_vectors(leg_vec_r, foot_vec_r)
        
        if (
            (lm[PoseLandmark.RIGHT_KNEE].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.RIGHT_ANKLE].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.RIGHT_FOOT_INDEX].visibility or 0) > VISIBILITY_THRESHOLD
        ):
            knee_r_2d = landmark_to_norm_xy(lm[PoseLandmark.RIGHT_KNEE])
            ankle_r_2d = landmark_to_norm_xy(lm[PoseLandmark.RIGHT_ANKLE])
            foot_r_2d = landmark_to_norm_xy(lm[PoseLandmark.RIGHT_FOOT_INDEX])
            leg_vec_r_2d = np.array(ankle_r_2d) - np.array(knee_r_2d)
            foot_vec_r_2d = np.array(foot_r_2d) - np.array(ankle_r_2d)
            angle_r_ankle_2d = angle_between_vectors(leg_vec_r_2d, foot_vec_r_2d)
        
        angle_r_ankle = angle_r_ankle_3d if angle_r_ankle_3d is not None else angle_r_ankle_2d

        angle_l_ankle_3d = angle_l_ankle_2d = None
        if (lm_world[PoseLandmark.LEFT_KNEE].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.LEFT_ANKLE].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.LEFT_FOOT_INDEX].visibility > VISIBILITY_THRESHOLD):
            knee_l = landmark_to_xyz(lm_world[PoseLandmark.LEFT_KNEE])
            ankle_l = landmark_to_xyz(lm_world[PoseLandmark.LEFT_ANKLE])
            foot_l = landmark_to_xyz(lm_world[PoseLandmark.LEFT_FOOT_INDEX])
            leg_vec_l = np.array(ankle_l) - np.array(knee_l)
            foot_vec_l = np.array(foot_l) - np.array(ankle_l)
            angle_l_ankle_3d = angle_between_vectors(leg_vec_l, foot_vec_l)
        
        if (
            (lm[PoseLandmark.LEFT_KNEE].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.LEFT_ANKLE].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.LEFT_FOOT_INDEX].visibility or 0) > VISIBILITY_THRESHOLD
        ):
            knee_l_2d = landmark_to_norm_xy(lm[PoseLandmark.LEFT_KNEE])
            ankle_l_2d = landmark_to_norm_xy(lm[PoseLandmark.LEFT_ANKLE])
            foot_l_2d = landmark_to_norm_xy(lm[PoseLandmark.LEFT_FOOT_INDEX])
            leg_vec_l_2d = np.array(ankle_l_2d) - np.array(knee_l_2d)
            foot_vec_l_2d = np.array(foot_l_2d) - np.array(ankle_l_2d)
            angle_l_ankle_2d = angle_between_vectors(leg_vec_l_2d, foot_vec_l_2d)
        
        angle_l_ankle = angle_l_ankle_3d if angle_l_ankle_3d is not None else angle_l_ankle_2d

        # Ankle roll (inversion/eversion) and yaw (toe-in/out) in 2D image plane
        roll_r = roll_l = yaw_r = yaw_l = None
        if (
            (lm[PoseLandmark.RIGHT_KNEE].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.RIGHT_ANKLE].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.RIGHT_FOOT_INDEX].visibility or 0) > VISIBILITY_THRESHOLD
        ):
            knee_r_2d = landmark_to_norm_xy(lm[PoseLandmark.RIGHT_KNEE])
            ankle_r_2d = landmark_to_norm_xy(lm[PoseLandmark.RIGHT_ANKLE])
            foot_r_2d = landmark_to_norm_xy(lm[PoseLandmark.RIGHT_FOOT_INDEX])
            leg_vec_r_2d = np.array(ankle_r_2d) - np.array(knee_r_2d)
            foot_vec_r_2d = np.array(foot_r_2d) - np.array(ankle_r_2d)
            roll_r = signed_angle_2d(leg_vec_r_2d, foot_vec_r_2d)
        if (
            (lm[PoseLandmark.LEFT_KNEE].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.LEFT_ANKLE].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.LEFT_FOOT_INDEX].visibility or 0) > VISIBILITY_THRESHOLD
        ):
            knee_l_2d = landmark_to_norm_xy(lm[PoseLandmark.LEFT_KNEE])
            ankle_l_2d = landmark_to_norm_xy(lm[PoseLandmark.LEFT_ANKLE])
            foot_l_2d = landmark_to_norm_xy(lm[PoseLandmark.LEFT_FOOT_INDEX])
            leg_vec_l_2d = np.array(ankle_l_2d) - np.array(knee_l_2d)
            foot_vec_l_2d = np.array(foot_l_2d) - np.array(ankle_l_2d)
            roll_l = signed_angle_2d(leg_vec_l_2d, foot_vec_l_2d)
        if (
            (lm[PoseLandmark.RIGHT_HIP].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.RIGHT_KNEE].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.RIGHT_ANKLE].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.RIGHT_FOOT_INDEX].visibility or 0) > VISIBILITY_THRESHOLD
        ):
            hip_r_2d = landmark_to_norm_xy(lm[PoseLandmark.RIGHT_HIP])
            knee_r_2d = landmark_to_norm_xy(lm[PoseLandmark.RIGHT_KNEE])
            ankle_r_2d = landmark_to_norm_xy(lm[PoseLandmark.RIGHT_ANKLE])
            foot_r_2d = landmark_to_norm_xy(lm[PoseLandmark.RIGHT_FOOT_INDEX])
            thigh_vec_r_2d = np.array(knee_r_2d) - np.array(hip_r_2d)
            foot_vec_r_2d = np.array(foot_r_2d) - np.array(ankle_r_2d)
            yaw_r = signed_angle_2d(thigh_vec_r_2d, foot_vec_r_2d)
        if (
            (lm[PoseLandmark.LEFT_HIP].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.LEFT_KNEE].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.LEFT_ANKLE].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.LEFT_FOOT_INDEX].visibility or 0) > VISIBILITY_THRESHOLD
        ):
            hip_l_2d = landmark_to_norm_xy(lm[PoseLandmark.LEFT_HIP])
            knee_l_2d = landmark_to_norm_xy(lm[PoseLandmark.LEFT_KNEE])
            ankle_l_2d = landmark_to_norm_xy(lm[PoseLandmark.LEFT_ANKLE])
            foot_l_2d = landmark_to_norm_xy(lm[PoseLandmark.LEFT_FOOT_INDEX])
            thigh_vec_l_2d = np.array(knee_l_2d) - np.array(hip_l_2d)
            foot_vec_l_2d = np.array(foot_l_2d) - np.array(ankle_l_2d)
            yaw_l = signed_angle_2d(thigh_vec_l_2d, foot_vec_l_2d)

        # Torso/spine angle: hip-shoulder-vertical (important for posture, deadlifts, squats)
        # Calculate torso lean using average of left/right shoulders and hips
        angle_torso_3d = angle_torso_2d = None
        if (lm_world[PoseLandmark.LEFT_SHOULDER].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.RIGHT_SHOULDER].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.LEFT_HIP].visibility > VISIBILITY_THRESHOLD and
            lm_world[PoseLandmark.RIGHT_HIP].visibility > VISIBILITY_THRESHOLD):
            # Average shoulder and hip positions for center-line
            ls = landmark_to_xyz(lm_world[PoseLandmark.LEFT_SHOULDER])
            rs = landmark_to_xyz(lm_world[PoseLandmark.RIGHT_SHOULDER])
            lh = landmark_to_xyz(lm_world[PoseLandmark.LEFT_HIP])
            rh = landmark_to_xyz(lm_world[PoseLandmark.RIGHT_HIP])
            shoulder_center = [(ls[i] + rs[i]) / 2 for i in range(3)]
            hip_center = [(lh[i] + rh[i]) / 2 for i in range(3)]
            # Vertical reference point (straight up from hip)
            vertical_ref = [hip_center[0], hip_center[1] - 1.0, hip_center[2]]
            angle_torso_3d = calculate_angle(vertical_ref, hip_center, shoulder_center, use_3d=True)
        
        if (
            (lm[PoseLandmark.LEFT_SHOULDER].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.RIGHT_SHOULDER].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.LEFT_HIP].visibility or 0) > VISIBILITY_THRESHOLD
            and (lm[PoseLandmark.RIGHT_HIP].visibility or 0) > VISIBILITY_THRESHOLD
        ):
            ls_2d = landmark_to_norm_xy(lm[PoseLandmark.LEFT_SHOULDER])
            rs_2d = landmark_to_norm_xy(lm[PoseLandmark.RIGHT_SHOULDER])
            lh_2d = landmark_to_norm_xy(lm[PoseLandmark.LEFT_HIP])
            rh_2d = landmark_to_norm_xy(lm[PoseLandmark.RIGHT_HIP])
            shoulder_center_2d = [(ls_2d[i] + rs_2d[i]) / 2 for i in range(2)]
            hip_center_2d = [(lh_2d[i] + rh_2d[i]) / 2 for i in range(2)]
            vertical_ref_2d = [hip_center_2d[0], hip_center_2d[1] - 1.0]
            angle_torso_2d = calculate_angle(vertical_ref_2d, hip_center_2d, shoulder_center_2d, use_3d=False)
        
        angle_torso = angle_torso_3d if angle_torso_3d is not None else angle_torso_2d

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

        angles = {
            "right_elbow": angle_r_elbow,
            "left_elbow": angle_l_elbow,
            "right_knee": angle_r_knee,
            "left_knee": angle_l_knee,
            "right_shoulder": angle_r_shoulder,
            "left_shoulder": angle_l_shoulder,
            "right_hip": angle_r_hip,
            "left_hip": angle_l_hip,
            "right_ankle": angle_r_ankle,
            "left_ankle": angle_l_ankle,
            "right_ankle_roll": roll_r,
            "left_ankle_roll": roll_l,
            "right_ankle_yaw": yaw_r,
            "left_ankle_yaw": yaw_l,
            "torso": angle_torso,
        }
        angles = self._stabilize_angles(angles)

        angle_r_elbow = angles.get("right_elbow")
        angle_l_elbow = angles.get("left_elbow")
        angle_r_knee = angles.get("right_knee")
        angle_l_knee = angles.get("left_knee")
        angle_r_ankle = angles.get("right_ankle")
        angle_l_ankle = angles.get("left_ankle")
        roll_r = angles.get("right_ankle_roll")
        roll_l = angles.get("left_ankle_roll")
        yaw_r = angles.get("right_ankle_yaw")
        yaw_l = angles.get("left_ankle_yaw")

        left_hand_str = f"elbow {fmt_angle(angle_l_elbow)}, wrist ({int(wrist_l_px[0])},{int(wrist_l_px[1])})" if left_wrist_ok else f"elbow {fmt_angle(angle_l_elbow)}, wrist (low conf)"
        right_hand_str = f"elbow {fmt_angle(angle_r_elbow)}, wrist ({int(wrist_r_px[0])},{int(wrist_r_px[1])})" if right_wrist_ok else f"elbow {fmt_angle(angle_r_elbow)}, wrist (low conf)"
        left_leg_str = f"knee {fmt_angle(angle_l_knee)}, ankle ({int(ankle_l_px[0])},{int(ankle_l_px[1])})" if left_leg_ok else "not in frame"
        right_leg_str = f"knee {fmt_angle(angle_r_knee)}, ankle ({int(ankle_r_px[0])},{int(ankle_r_px[1])})" if right_leg_ok else "not in frame"

        text_lines = [
            "--- Pose Output ---",
            "",
            line_with_status(angle_l_elbow, "left_elbow", "Left hand ", left_hand_str),
            line_with_status(angle_r_elbow, "right_elbow", "Right hand", right_hand_str),
            "",
            line_with_status(angle_l_knee, "left_knee", "Left leg  ", left_leg_str),
            line_with_status(angle_r_knee, "right_knee", "Right leg ", right_leg_str),
            "",
            f"Left ankle angle : {fmt_angle(angle_l_ankle)}",
            f"Right ankle angle: {fmt_angle(angle_r_ankle)}",
            f"Left ankle roll  : {fmt_angle(roll_l)}",
            f"Right ankle roll : {fmt_angle(roll_r)}",
            f"Left ankle yaw   : {fmt_angle(yaw_l)}",
            f"Right ankle yaw  : {fmt_angle(yaw_r)}",
        ]
        return text_lines, connection_spec, angles

    def step(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        # Convert to grayscale BEFORE any CV processing (testing color impact)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self._clahe.apply(gray)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        h, w, _ = frame.shape

        if self.frame_count % self.detect_every_n == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.cv_width > 0 and self.cv_height > 0:
                if self.cv_width != w or self.cv_height != h:
                    frame_rgb = cv2.resize(frame_rgb, (self.cv_width, self.cv_height), interpolation=cv2.INTER_LINEAR)
            elif self.detect_scale < 1.0:
                dw, dh = int(w * self.detect_scale), int(h * self.detect_scale)
                if dw > 0 and dh > 0:
                    frame_rgb = cv2.resize(frame_rgb, (dw, dh), interpolation=cv2.INTER_LINEAR)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(self.frame_count * self.frame_interval_ms)
            new_results = self.pose.detect_for_video(mp_image, timestamp_ms)
            has_pose = bool(new_results and new_results.pose_landmarks and new_results.pose_world_landmarks)
            if has_pose:
                self.last_results = new_results
                self._last_good_results = new_results
                self._pose_hold_count = 0
            else:
                if self._last_good_results is not None and self._pose_hold_count < self._pose_hold_frames:
                    self.last_results = self._last_good_results
                    self._pose_hold_count += 1
                else:
                    self.last_results = new_results

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
        alert_red = False
        frame_json = None

        if results and results.pose_landmarks and results.pose_world_landmarks:
            lm_raw = results.pose_landmarks[0]
            lm = smooth_landmarks(self.last_smoothed_lm, lm_raw, self.smoothing_alpha)
            self.last_smoothed_lm = lm
            lm_world = results.pose_world_landmarks[0]
            text_lines, connection_spec, angles = self._build_text_lines(lm, lm_world, w, h)
            if angles:
                le = angles.get("left_elbow")
                re = angles.get("right_elbow")
                alert_red = any(
                    a is not None and abs(a - ELBOW_ALERT_ANGLE) <= ELBOW_ALERT_TOL
                    for a in (le, re)
                )
            else:
                alert_red = False

            # Build comprehensive JSON frame
            now_ms = int(datetime.now().timestamp() * 1000)
            frame_json = self._build_frame_json(now_ms, lm, lm_world, angles, w, h)
            
            # Time-based logging: write every log_update_interval_ms (use video time when reading from file)
            current_log_time_ms = int(self.frame_count * self.frame_interval_ms) if self.use_video_time else now_ms
            if current_log_time_ms - self.last_log_time_ms >= self.log_update_interval_ms:
                self.log_buffer.append(frame_json)
                self.last_log_time_ms = current_log_time_ms
            
            # Batch write when buffer is full
            if len(self.log_buffer) >= self.log_batch_size:
                self.flush_log_buffer()

        self.frame_count += 1
        return {
            "frame": frame,
            "text_lines": text_lines,
            "connection_spec": connection_spec,
            "landmarks": lm,
            "world_landmarks": lm_world,
            "alert_red": alert_red,
            "frame_json": frame_json,
        }

    def _build_frame_json(self, now_ms, lm, lm_world, angles, w, h):
        """Build comprehensive JSON for this frame: timestamps, landmarks (filtered by visibility), angles, exercise_metrics."""
        if self.use_video_time:
            timestamp_ms = int(self.frame_count * self.frame_interval_ms)
        else:
            timestamp_ms = now_ms - self.session_start_ms
        min_vis = self.log_min_visibility
        
        # Filter world landmarks: skip low-visibility to reduce size
        world_lms = []
        for idx, p in enumerate(lm_world):
            vis = getattr(p, "visibility", 0) or 0
            if vis >= min_vis:
                world_lms.append({
                    "idx": idx,
                    "x": round(p.x, 6),
                    "y": round(p.y, 6),
                    "z": round(p.z, 6),
                    "visibility": round(vis, 3),
                })
        
        # Filter image landmarks: skip low-visibility (2D mostly for overlay; can reconstruct from 3D)
        image_lms = []
        for idx, p in enumerate(lm):
            vis = getattr(p, "visibility", 0) or 0
            if vis >= min_vis:
                image_lms.append({
                    "idx": idx,
                    "x": round(p.x * w, 1),
                    "y": round(p.y * h, 1),
                    "visibility": round(vis, 3),
                })
        
        entry = {
            "timestamp_ms": timestamp_ms,
            "timestamp_utc": now_ms,
            "frame_index": self.frame_count,
            # "pose_world": {"landmarks": world_lms},
            # "pose_image": {"width": w, "height": h, "landmarks": image_lms},
            "angles": {k: round(v, 1) if v is not None else None for k, v in (angles or {}).items()},
            "exercise_metrics": {
                "rep_detected": False,
                "hand_above_head": False,
                "foot_off_ground": False,
            },
        }
        return entry


def flip_landmarks_x(lm):
    """Return a copy of normalized landmarks with x flipped (1 - x) for mirror view."""
    if lm is None:
        return None
    from mediapipe.tasks.python.components.containers import landmark as landmark_module
    return [
        landmark_module.NormalizedLandmark(
            x=1.0 - p.x,
            y=p.y,
            z=getattr(p, "z", 0) or 0,
            visibility=getattr(p, "visibility", None),
            presence=getattr(p, "presence", None),
        )
        for p in lm
    ]


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

