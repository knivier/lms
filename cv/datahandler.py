import json
import sys
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
import time
import torch
import torch.nn as nn

# Rep detection parameters per workout (joints and angle thresholds).
# Pushups: elbow at top (extended) ~150-180°, at bottom (bent) ~70-100°. We need top >= max, then bottom <= min, then top >= max again.
TOLERANCE_DEGREES = 8
TOLERANCE_TIME = 0.8
WORKOUT_TO_PARAMETERS = {
    "pushups": {"min_threshold": 100, "max_threshold": 150, "joints": ("left_elbow", "right_elbow"), "target_min_angle": 90, "target_max_angle": 160, "target_duration": 1.5},
    "pushup": {"min_threshold": 100, "max_threshold": 150, "joints": ("left_elbow", "right_elbow"), "target_min_angle": 90, "target_max_angle": 160, "target_duration": 1.5},  # alias
    "squat": {"min_threshold": 90, "max_threshold": 150, "joints": ("left_knee", "right_knee"), "target_min_angle": 70, "target_max_angle": 150, "target_duration": 2.0},
    "bicep_curl": {"min_threshold": 80, "max_threshold": 135, "joints": ("left_elbow", "right_elbow"), "target_min_angle": 40, "target_max_angle": 155, "target_duration": 1.8},
}
class SimpleRepDetector:
    WAITING_TOP = 0
    DESCENDING = 1
    BOTTOM_REACHED = 2
    ASCENDING = 3

    def __init__(self, min_threshold, max_threshold, joints):
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.joints = joints

        self.state = self.WAITING_TOP
        self.prev_angle = None
        self.current_rep = []

    def _get_angle(self, joint_angles):
        a = joint_angles.get(self.joints[0])
        b = joint_angles.get(self.joints[1])
        if a is None and b is None:
            return None
        if a is None:
            return b
        if b is None:
            return a
        return (a + b) / 2.0

    def feed(self, joint_angles, timestamp):
        angle = self._get_angle(joint_angles)
        if angle is None:
            return None

        if self.prev_angle is None:
            self.prev_angle = angle
            return None

        decreasing = angle < self.prev_angle
        increasing = angle > self.prev_angle

        # -------------------------
        # STATE MACHINE
        # -------------------------
        if self.state == self.WAITING_TOP:
            if angle >= self.max_threshold:
                self.state = self.DESCENDING
                self.current_rep = [{"angle": angle, "timestamp": timestamp}]
                _log_transition("DESCENDING", angle)

        elif self.state == self.DESCENDING:
            self.current_rep.append({"angle": angle, "timestamp": timestamp})
            if angle <= self.min_threshold:
                self.state = self.BOTTOM_REACHED
                _log_transition("BOTTOM_REACHED", angle)

        elif self.state == self.BOTTOM_REACHED:
            self.current_rep.append({"angle": angle, "timestamp": timestamp})
            # Leave bottom: angle has risen above min (more robust than requiring frame-to-frame increase)
            if angle > self.min_threshold:
                self.state = self.ASCENDING
                _log_transition("ASCENDING", angle)

        elif self.state == self.ASCENDING:
            self.current_rep.append({"angle": angle, "timestamp": timestamp})
            # Complete rep when we reach top again (use slightly lower than max so we don't require full lockout)
            completion_threshold = self.max_threshold - 5
            if angle >= completion_threshold:
                rep = self.current_rep
                self.current_rep = []
                self.state = self.DESCENDING  # allow next rep immediately
                self.prev_angle = angle
                _log_transition("REP_COMPLETE", angle)
                
                return rep  # full rep collected

        self.prev_angle = angle
        return None

rep_indexes = []

def _log_transition(state_name, angle):
    """Log state transition to stderr so we can see why reps might not complete."""
    print(f"[Datahandler] state -> {state_name} (angle={angle:.1f}°)", file=sys.stderr, flush=True)

def to_fixed_length_nd(points, target_len=50):
    points = np.asarray(points, dtype=float)
    n_dims = points.shape[1]

    x_old = np.linspace(0, 1, len(points))
    x_new = np.linspace(0, 1, target_len)

    out = np.zeros((target_len, n_dims))
    for d in range(n_dims):
        out[:, d] = np.interp(x_new, x_old, points[:, d])

    return out
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CROUCH_MODEL_PATH = _REPO_ROOT / "quantprocess" / "crouch_model.pth"

model = nn.Sequential(
    nn.Linear(50, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)
if _CROUCH_MODEL_PATH.exists():
    model.load_state_dict(torch.load(_CROUCH_MODEL_PATH, map_location="cpu"))
else:
    print(f"[Datahandler] quality model not found at {_CROUCH_MODEL_PATH}, rep_quality will be untrained", file=sys.stderr, flush=True)
model.eval()
def rep_summary(rep, workout_type="pushups"):
    global model
    angles = [p["angle"] for p in rep]
    times = [p["timestamp"] for p in rep]

    # Resample angle series to 50 points (1D); model expects (batch, 50)
    points_1d = to_fixed_length(angles[:-1], target_len=50)
    pt_min, pt_max = np.min(points_1d), np.max(points_1d)
    points_norm = (points_1d - pt_min) / (pt_max - pt_min) if pt_max > pt_min else np.zeros_like(points_1d)
    rep_quality = model(torch.tensor(points_norm, dtype=torch.float32).unsqueeze(0)).item()
    min_angle = min(angles)
    max_angle = max(angles)
    duration = (times[-1] - times[0]) / 1000.0  # seconds
    range_of_motion = max_angle - min_angle

    msg = ""
    if WORKOUT_TO_PARAMETERS.get(workout_type):
        params = WORKOUT_TO_PARAMETERS[workout_type]
        target_min = params.get("target_min_angle", 0)
        target_max = params.get("target_max_angle", 180)
        if abs(min_angle - target_min) < TOLERANCE_DEGREES:
            msg += "Good depth.\n"
        elif min_angle < target_min - TOLERANCE_DEGREES:
            msg += "TOO FAR.\n"
        else:
            msg += "Not low enough.\n"
        if abs(max_angle - target_max) < TOLERANCE_DEGREES:
            msg += "Good extension.\n"
        elif max_angle < target_max - TOLERANCE_DEGREES:
            msg += "Not fully extended.\n"
        else:
            msg += "Overextended.\n"
        target_duration = params.get("target_duration", 1.0)
        if abs(duration - target_duration) < TOLERANCE_TIME:
            msg += " Good tempo.\n"
        elif duration < target_duration - TOLERANCE_TIME:
            msg += " Too fast.\n"
        else:
            msg += " Too slow.\n"

    return {
        "min_angle": min_angle,
        "max_angle": max_angle,
        "duration": duration,
        "range_of_motion": range_of_motion,
        "num_frames": len(rep),
        "quality_score": rep_quality,
        "feedback": msg.strip(),
        "input": points_norm.tolist(),
    }

reps = []
def to_fixed_length(points, target_len=50):
    points = np.asarray(points, dtype=float)

    if len(points) == 0:
        return np.zeros(target_len)

    x_old = np.linspace(0, 1, len(points))
    x_new = np.linspace(0, 1, target_len)

    return np.interp(x_new, x_old, points)

def _workout_id_path():
    return Path(__file__).resolve().parent.parent / "workout_id.json"

def _reps_log_path():
    """Path for appending detected reps (JSONL) during live run."""
    return Path(__file__).resolve().parent / "reps_log.jsonl"

def _read_workout_state():
    """Read workout_id.json (JSONL: one line). Returns dict with workout_id, session."""
    p = _workout_id_path()
    if not p.exists():
        return {"workout_id": "pushups", "session": "off"}
    try:
        with open(p, "r") as f:
            line = f.readline()
            if not line.strip():
                return {"workout_id": "pushups", "session": "off"}
            return json.loads(line)
    except (json.JSONDecodeError, OSError):
        return {"workout_id": "pushups", "session": "off"}

def session_is_on():
    """True if workout_id.json has session == 'on'."""
    return _read_workout_state().get("session", "off").lower() == "on"

def workout_init():
    global detector
    data = _read_workout_state()
    workout_type = data.get("workout_id", "pushups") or "pushups"
    params = WORKOUT_TO_PARAMETERS.get(workout_type, WORKOUT_TO_PARAMETERS["pushups"])
    detector = SimpleRepDetector(
        min_threshold=params["min_threshold"],
        max_threshold=params["max_threshold"],
        joints=params["joints"],
    )
    print(f"[Datahandler] Session ON, workout={workout_type!r}, rep detection active", file=sys.stderr, flush=True)

detector = None
_last_workout_id = None
_debug_last_print_time = [0.0]  # list so we can mutate in closure

def run_workout(joint_angles, timestamp):
    global reps, detector, _last_workout_id, _debug_last_print_time
    data = _read_workout_state()
    wid = data.get("workout_id") or "pushups"
    if detector is None or _last_workout_id != wid:
        _last_workout_id = wid
        workout_init()
    rep = detector.feed(joint_angles, timestamp)
    # Throttled debug: print current angle and state ~once per second so we can verify values
    t = time.monotonic()
    if t - _debug_last_print_time[0] >= 1.0:
        _debug_last_print_time[0] = t
        a = detector._get_angle(joint_angles)
        state_name = ["WAITING_TOP", "DESCENDING", "BOTTOM_REACHED", "ASCENDING"][detector.state]
        joint_label = detector.joints[0].split("_")[-1]  # elbow, knee, etc.
        print(f"[Datahandler] {joint_label} avg={a:.1f}° state={state_name} (min={detector.min_threshold}, max={detector.max_threshold})", file=sys.stderr, flush=True)
    if rep is not None:
        reps.append(rep)
        summary = rep_summary(rep, workout_type=wid)
        print(f"[Datahandler] Rep detected: {summary}", file=sys.stderr, flush=True)
        # Log rep to disk (JSONL) so live runs persist reps
        try:
            entry = {
                "workout_id": wid,
                "timestamp_ms": timestamp,
                "summary": summary,
            }
            with open(_reps_log_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass
        return summary
    return None
readLines = 0
def store_reps():
    global reps
    with open("pose_log.json", "r") as f:
        readingLine = 0
        for line in f:
            readingLine += 1
            if readingLine > readLines:
                summary = run_workout(json.loads(line).get("angles", {}), json.loads(line).get("timestamp_utc", 0))
                if summary is not None:
                    print(f"Rep detected: {summary}", file=sys.stderr, flush=True)
                readLines += 1
            
    with open("reps_summary.json", "w") as f:
        json.dump([rep_summary(rep, workout_type=_last_workout_id or "pushups") for rep in reps], f, indent=4)

if __name__ == "__main__":
    while True:
        data = _read_workout_state()
        if data.get("session", "off").lower() != "on":
            time.sleep(1)
            continue
        store_reps()
        time.sleep(1)