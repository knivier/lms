import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline


JSONL_FILE = "./lms/cv/pose_log.jsonl"
ANGLE_KEY = "left_knee"

time = []
signal = []

with open(JSONL_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # truncate trailing junk if any
        line = line[:line.rfind("}") + 1]
        try:
            obj = json.loads(line)
        except:
            continue

        left_knee = obj.get("angles", {}).get("left_knee")
        if left_knee is None:
            continue  # skip nulls
        right_knee = obj.get("angles", {}).get("right_knee")
        if right_knee is None:
            continue  # skip nulls
        value = (left_knee + right_knee) / 2.0
        signal.append(value)

signal = np.array(signal[0:150])


inverted = -np.array(signal)

# detect peaks (dips)
peaks, _ = find_peaks(inverted, distance=10)  # distance can be adjusted
peaks = peaks[signal[peaks] < 140]
print("Number of reps:", len(peaks))


plt.figure(figsize=(10,4))
plt.plot(signal, label='Both Knees')
plt.plot(peaks, signal[peaks], 'rx', label='Dips/Reps < 140')
plt.legend()
plt.show()

# ============================
# FFT between first and last peak
# # ============================
# if len(peaks) >= 2:
#     start = peaks[0]
#     end = peaks[-1]
#     segment = signal[start:end+1]

#     # Compute FFT
#     N = len(segment)
#     fft_vals = np.fft.fft(segment)
#     fft_freq = np.fft.fftfreq(N, d=1)  # d=1 assumes 1 sample per unit time; adjust if you know FPS

#     # Only take positive frequencies
#     pos_mask = fft_freq > 0
#     fft_freq = fft_freq[pos_mask]
#     fft_vals = np.abs(fft_vals[pos_mask])

#     # Plot FFT
#     plt.figure(figsize=(10,4))
#     plt.plot(fft_freq, fft_vals)
#     plt.title("FFT of segment between first and last peak")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Magnitude")
#     plt.show()
# else:
#     print("Not enough peaks detected for FFT.")

if len(peaks) < 2:
    raise ValueError("Not enough peaks detected.")

# Segment between first two peaks
start, end = peaks[0], peaks[1]
segment = signal[start:end+1]
x = np.arange(len(segment))

# Fit 6th-degree spline
spline = UnivariateSpline(x, segment, k=5, s=0)  # s=0 forces exact fit
spline_vals = spline(x)

print()
# Plot
plt.figure(figsize=(8,4))
plt.plot(x, segment, label="Original Rep")
plt.plot(x, spline_vals, '--', label="6th-degree Spline Fit")
plt.xlabel("Frame")
plt.ylabel("Knee Angle")
plt.title("Spline Fit of First Rep")
plt.legend()
plt.show()
    
    