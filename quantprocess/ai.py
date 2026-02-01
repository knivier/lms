# Model testing
import json
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
JSONL_FILE = REPO_ROOT / "training-data" / "crouch" / "pose_log.jsonl"
OUTPUT_FILE = BASE_DIR / "dataset.jsonl"  # where we append all reps

# ---------------------------
# Load data
# ---------------------------
time = []
signal = []

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

        left_knee = obj.get("angles", {}).get("left_knee")
        right_knee = obj.get("angles", {}).get("right_knee")
        if left_knee is None or right_knee is None:
            continue

        value = (left_knee + right_knee) / 2.0
        signal.append(value)
        time.append(obj.get("timestamp_utc", len(time)))

signal = np.array(signal)
time = np.array(time)
time = (time - time[0]) / 1000.0  # convert to seconds

# ---------------------------
# Detect all reps
# ---------------------------
inverted = -signal
peaks, _ = find_peaks(inverted, distance=5)
peaks = peaks[signal[peaks] < 140]
plt.plot(time, signal)
plt.plot(time[peaks], signal[peaks], "x")
plt.show()
if len(peaks) < 2:
    raise ValueError("Not enough peaks detected.")

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

for i in range(len(peaks)-1):
    start, end = peaks[i], peaks[i+1]
    rep_segment = signal[start:end+1]
    rep_time = time[start:end+1]

    if len(rep_segment) < 5:  # skip very short segments
        continue
    
    # spline fit
    spline = UnivariateSpline(rep_time, rep_segment, k=5, s=0)
    time_resampled = np.linspace(rep_time[0], rep_time[-1], fixed_length)
    spline_vals_resampled = spline(time_resampled)
    spline_vals_norm = (spline_vals_resampled - np.min(spline_vals_resampled)) / \
                       (np.max(spline_vals_resampled) - np.min(spline_vals_resampled))
    plt.figure(figsize=(8, 4))
    plt.plot(rep_time, rep_segment, label="Original Rep")
    plt.plot(time_resampled, spline_vals_resampled, '--', label="5th-degree Spline Fit")
    plt.xlabel("Time (s)")
    plt.ylabel("Knee Angle")
    plt.title("First Rep Spline Fit Using Actual Time")
    plt.legend()
    plt.show()
    # Ask user for label
    print("Predicted Score: " + str(model(torch.tensor(spline_vals_norm, dtype=torch.float32).unsqueeze(0)).item()))
    print(f"Rep {i+1}/{len(peaks)-1}:")
    print("Normalized spline vector preview:", spline_vals_norm[:5], "...")  # first 5 values
    label = input("Enter label/output for this rep (e.g., 0 or 1): ")
    try:
        label = float(label)
    except:
        print("Invalid input, defaulting to 0")
        label = 0

    # Append to dataset
    dataset.append({"input": spline_vals_norm.tolist(), "output": label})

# ---------------------------
# Save dataset
# ---------------------------
with open(OUTPUT_FILE, "a") as f:  # append mode
    for entry in dataset:
        f.write(json.dumps(entry) + "\n")

print(f"Saved {len(dataset)} reps to {OUTPUT_FILE}")
