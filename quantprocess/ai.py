# Model testing
import json
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
from RepTracker import SimpleRepDetector

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
JSONL_FILE = REPO_ROOT / "training-data" / "pushups" / "proman.jsonl"
OUTPUT_FILE = BASE_DIR / "dataset.jsonl"  # where we append all reps


    
    
# ---------------------------
# Load data
# ---------------------------
time = []
signal = []

detector = SimpleRepDetector(
    min_threshold=120,
    max_threshold=145,
    joints=("left_elbow", "right_elbow")
)

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
with open(JSONL_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        line = line[:line.rfind("}") + 1]

        try:
            obj = json.loads(line)
        except:
            continue
        
        left_elbow = obj.get("angles", {}).get("left_elbow")
        right_elbow = obj.get("angles", {}).get("right_elbow")
        if left_elbow is None or right_elbow is None:
            continue

        value = (left_elbow + right_elbow) / 2.0

        # ✅ append FIRST
        signal.append(value)
        time.append(obj.get("timestamp_utc", len(time)))

        # print(f"Feeding angle {value} at time {obj.get('timestamp_utc')}")
        # feed AFTER
        rep = detector.feed(obj["angles"], obj["timestamp_utc"])
        if rep is not None:
            summary = rep_summary(rep)
            print("Rep detected:", summary)
            reps.append(rep)
            rep_indexes.append(len(signal)  + 1)  # index of the bottom point

signal = np.array(signal)
time = np.array(time)
time = (time - time[0]) / 1000.0

plt.plot(time, signal)
plt.plot(time[rep_indexes], signal[rep_indexes], "x")
plt.xlabel("Time (s)")
plt.ylabel("Avg Knee Angle")
plt.title("Detected Rep Bottoms")
plt.show()

# ---------------------------
# Detect all reps
# ---------------------------
# inverted = -signal
# peaks, _ = find_peaks(inverted, distance=5)
# peaks = peaks[signal[peaks] < 140]
# plt.plot(time, signal)
# plt.plot(time[rep_indexes], signal[rep_indexes], "x")
# plt.show()
# if len(peaks) < 2:
#     raise ValueError("Not enough peaks detected.")

# ---------------------------
# Extract each rep
# ---------------------------
fixed_length = 50
dataset = []

model = nn.Sequential(
    nn.Linear(50, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)

model.load_state_dict(torch.load(BASE_DIR / "crouch_model.pth"))
model.eval()
peaks = signal[np.array(rep_indexes)]
peaks = np.array(rep_indexes)
def to_fixed_length(points, target_len=50):
    points = np.asarray(points, dtype=float)

    if len(points) == 0:
        return np.zeros(target_len)

    x_old = np.linspace(0, 1, len(points))
    x_new = np.linspace(0, 1, target_len)

    return np.interp(x_new, x_old, points)


for i in range(len(peaks) - 1):
    start, end = peaks[i], peaks[i + 1]
    rep_segment = signal[start:end + 1]
    rep_time = time[start:end + 1]

    if len(rep_segment) < 5:  # skip very short segments
        continue

    # 🔁 Resample to fixed length using interpolation
    rep_resampled = to_fixed_length(rep_segment, fixed_length)

    # Normalize to [0, 1]
    min_val = np.min(rep_resampled)
    max_val = np.max(rep_resampled)
    if max_val - min_val == 0:
        continue  # avoid divide-by-zero reps

    rep_norm = (rep_resampled - min_val) / (max_val - min_val)

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(rep_time, rep_segment, label="Original Rep")
    plt.plot(
        np.linspace(rep_time[0], rep_time[-1], fixed_length),
        rep_resampled,
        '--',
        label="Interpolated (Fixed Length)"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Knee Angle")
    plt.title(f"Rep {i+1} Interpolated to Fixed Length")
    plt.legend()
    plt.show()

    # Model prediction
    with torch.no_grad():
        pred = model(
            torch.tensor(rep_norm, dtype=torch.float32).unsqueeze(0)
        ).item()

    print("Predicted Score:", pred)
    print(f"Rep {i+1}/{len(peaks)-1}:")
    print("Normalized vector preview:", rep_norm[:5], "...")

    # Ask user for label
    label = input("Enter label/output for this rep (e.g., 0 or 1): ")
    try:
        label = float(label)
    except:
        print("Invalid input, defaulting to 0")
        label = 0.0

    # Append to dataset
    dataset.append({
        "input": rep_norm.tolist(),
        "output": label
    })
# ---------------------------
# Save dataset
# ---------------------------
with open(OUTPUT_FILE, "a") as f:  # append mode
    for entry in dataset:
        f.write(json.dumps(entry) + "\n")

print(f"Saved {len(dataset)} reps to {OUTPUT_FILE}")
