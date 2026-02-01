import json
import sys
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
from time import sleep

# Rep detection parameters per workout (joints and angle thresholds).
WORKOUT_TO_PARAMETERS = {
    "pushups": {"min_threshold": 120, "max_threshold": 145, "joints": ("left_elbow", "right_elbow")},
    "squat": {"min_threshold": 80, "max_threshold": 170, "joints": ("left_knee", "right_knee")},
    "bicep_curl": {"min_threshold": 30, "max_threshold": 150, "joints": ("left_elbow", "right_elbow")},
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
        if a is None or b is None:
            return None
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

        elif self.state == self.DESCENDING:
            self.current_rep.append({"angle": angle, "timestamp": timestamp})
            if angle <= self.min_threshold:
                self.state = self.BOTTOM_REACHED

        elif self.state == self.BOTTOM_REACHED:
            self.current_rep.append({"angle": angle, "timestamp": timestamp})
            if increasing:
                self.state = self.ASCENDING

        elif self.state == self.ASCENDING:
            self.current_rep.append({"angle": angle, "timestamp": timestamp})
            if angle >= self.max_threshold:
                rep = self.current_rep
                self.current_rep = []
                self.state = self.DESCENDING  # allow next rep immediately
                self.prev_angle = angle
                return rep  # full rep collected

        self.prev_angle = angle
        return None

rep_indexes = []

def rep_summary(rep):
    angles = [p["angle"] for p in rep]
    times = [p["timestamp"] for p in rep]

    min_angle = min(angles)
    max_angle = max(angles)
    duration = (times[-1] - times[0]) / 1000.0  # seconds
    range_of_motion = max_angle - min_angle

    return {
        "min_angle": min_angle,
        "max_angle": max_angle,
        "duration": duration,
        "range_of_motion": range_of_motion,
        "num_frames": len(rep)
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

def run_workout(joint_angles, timestamp):
    global reps, detector, _last_workout_id
    data = _read_workout_state()
    wid = data.get("workout_id") or "pushups"
    if detector is None or _last_workout_id != wid:
        _last_workout_id = wid
        workout_init()
    rep = detector.feed(joint_angles, timestamp)
    if rep is not None:
        reps.append(rep)
        summary = rep_summary(rep)
        print(f"[Datahandler] Rep detected: {summary}", file=sys.stderr, flush=True)
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
        json.dump([rep_summary(rep) for rep in reps], f, indent=4)

if __name__ == "__main__":
    while True:
        data = _read_workout_state()
        if data.get("session", "off").lower() != "on":
            sleep(1)
            continue
        store_reps()
        sleep(1)