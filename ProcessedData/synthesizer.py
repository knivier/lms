import json
import random
from pathlib import Path

import numpy as np

# Generates SyntheticData.jsonl (JSONL) from existing reps.
# This simulates human variance by adding noise, time-warp, trend, and mirroring.

DATA_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = DATA_DIR / "SyntheticData.jsonl"

SOURCE_FILES = [
    "AICrouch.jsonl",
    "FacingForwardCrouch.jsonl",
    "HorridCrouch.jsonl",
    "NathanCrouching.jsonl",
]

GOOD_COPIES_PER_REP = 15
BAD_COPIES_PER_REP = 45
BAD_THRESHOLD = 0.6  # < 0.6 is bad, >= 0.6 is good

NOISE_STD = 0.025            # Gaussian noise added to each feature
TREND_STD = 0.02             # small linear slope added across the 50 dims
TIME_WARP_RANGE = (0.9, 1.1) # stretch/compress rep timing
MIRROR_PROB = 0.5            # reverse the vector (approx left/right mirror)
GOOD_OUTPUT_JITTER = 0.08    # soften good labels slightly
BAD_OUTPUT_JITTER = 0.08     # soften bad labels slightly


def time_warp(vec, scale):
    """Resample vector to same length after time scaling."""
    n = len(vec)
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, n) * scale
    x_new = np.clip(x_new, 0, 1)
    return np.interp(x_old, x_new, vec)


def synthesize_one(vec):
    v = np.array(vec, dtype=float)

    # time warp
    scale = random.uniform(*TIME_WARP_RANGE)
    v = time_warp(v, scale)

    # small trend change (torso slope / rep slope)
    trend = np.linspace(-1, 1, len(v)) * random.gauss(0.0, TREND_STD)
    v = v + trend

    # gaussian noise on features
    v = v + np.random.normal(0.0, NOISE_STD, size=v.shape)

    # optional mirror (approx left/right)
    if random.random() < MIRROR_PROB:
        v = v[::-1]

    # clamp to [0, 1] since original inputs are in range
    v = np.clip(v, 0.0, 1.0)
    return v.tolist()


def main():
    good_sources = []
    bad_sources = []
    for name in SOURCE_FILES:
        path = DATA_DIR / name
        if not path.exists():
            continue
        with open(path, "r") as f:
            for line in f:
                obj = json.loads(line)
                out = obj.get("output", 0.0)
                if not isinstance(out, (int, float)):
                    continue
                if out < BAD_THRESHOLD:
                    bad_sources.append(obj)
                else:
                    good_sources.append(obj)

    if not good_sources and not bad_sources:
        raise SystemExit("No reps found to synthesize from.")

    synthetic = []
    for obj in good_sources:
        vec = obj["input"]
        base_out = float(obj["output"])
        for _ in range(GOOD_COPIES_PER_REP):
            v = synthesize_one(vec)
            # soften the label slightly so model learns a range, not just 1.0
            out = max(0.0, min(1.0, base_out - abs(random.gauss(0.0, GOOD_OUTPUT_JITTER))))
            synthetic.append({"input": v, "output": out})

    for obj in bad_sources:
        vec = obj["input"]
        base_out = float(obj["output"])
        for _ in range(BAD_COPIES_PER_REP):
            v = synthesize_one(vec)
            # keep bad labels below the threshold and add small variance
            out = max(0.0, min(1.0, base_out + random.gauss(0.0, BAD_OUTPUT_JITTER)))
            if out >= BAD_THRESHOLD:
                out = BAD_THRESHOLD - 0.01
            synthetic.append({"input": v, "output": out})

    with open(OUTPUT_PATH, "w") as f:
        for item in synthetic:
            f.write(json.dumps(item) + "\n")

    print(
        f"Wrote {len(synthetic)} synthetic samples to {OUTPUT_PATH} "
        f"(good={len(good_sources)}*{GOOD_COPIES_PER_REP}, bad={len(bad_sources)}*{BAD_COPIES_PER_REP})"
    )


if __name__ == "__main__":
    main()
