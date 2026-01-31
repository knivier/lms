#!/usr/bin/env python3
"""
3D animated human model using the same MediaPipe pose pipeline as Compy.py.
Renders a capsule-based humanoid (thick limbs + head) that moves with your pose—
like a video camera view of an animated character.
Run: python pose_3d_viewer.py
"""

import numpy as np
from pathlib import Path


def _capsule_triangles(p1, p2, radius, n_theta=10):
    """Triangles for a capsule (cylinder) from p1 to p2. Returns list of (3,3) arrays."""
    p1, p2 = np.array(p1), np.array(p2)
    vec = p2 - p1
    length = np.linalg.norm(vec)
    if length < 1e-6:
        return []
    z_axis = vec / length
    # Build cylinder along Z from 0 to length, then rotate and translate
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    z = np.linspace(0, length, 4)
    verts_local = []
    for zi in z:
        for ti in theta:
            verts_local.append([radius * np.cos(ti), radius * np.sin(ti), zi])
    verts_local = np.array(verts_local)
    # Rotation from Z to z_axis
    z_unit = np.array([0, 0, 1])
    axis = np.cross(z_unit, z_axis)
    axis_norm = np.linalg.norm(axis)
    if axis_norm > 1e-6:
        axis /= axis_norm
        angle = np.arccos(np.clip(np.dot(z_unit, z_axis), -1, 1))
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    else:
        R = np.eye(3) if np.dot(z_unit, z_axis) > 0 else -np.eye(3)
    verts = (R @ verts_local.T).T + p1
    n_z, n_t = len(z), len(theta)
    triangles = []
    for i in range(n_z - 1):
        for j in range(n_t):
            i0 = i * n_t + j
            i1 = i * n_t + (j + 1) % n_t
            i2 = (i + 1) * n_t + (j + 1) % n_t
            i3 = (i + 1) * n_t + j
            triangles.append(verts[[i0, i1, i2]])
            triangles.append(verts[[i0, i2, i3]])
    return triangles


def _sphere_triangles(center, radius, n=8):
    """Triangles for a sphere at center. Returns list of (3,3) arrays."""
    center = np.array(center)
    phi = np.linspace(0, np.pi, n + 1)
    theta = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    verts = []
    for pi in phi:
        for ti in theta:
            x = radius * np.sin(pi) * np.cos(ti)
            y = radius * np.sin(pi) * np.sin(ti)
            z = radius * np.cos(pi)
            verts.append(center + [x, y, z])
    verts = np.array(verts)
    triangles = []
    n_phi, n_theta = len(phi), len(theta)
    for i in range(n_phi - 1):
        for j in range(n_theta):
            i0 = i * n_theta + j
            i1 = i * n_theta + (j + 1) % n_theta
            i2 = (i + 1) * n_theta + (j + 1) % n_theta
            i3 = (i + 1) * n_theta + j
            triangles.append(verts[[i0, i1, i2]])
            triangles.append(verts[[i0, i2, i3]])
    return triangles


# World coordinate transform (MediaPipe → viewable 3D)
# MediaPipe: x,y ~ meters, z negative in front of camera. We flip z, scale, center on hips.
WORLD_SCALE = 2.0          # scale so limbs are visible in plot
WORLD_FLIP_Z = True        # z = -z for standard camera convention
WORLD_CENTER_HIPS = True   # center model on hip midpoint
# Matplotlib: x=horizontal, y=vertical, z=depth. If pose looks twisted, try "yz" to swap y/z.
WORLD_AXIS_SWAP = None     # None or "yz"

# Temporal smoothing (alpha: 0=full smooth, 1=no smooth)
SMOOTH_ALPHA = 0.4

# Coherent human: only main body segments (no face/finger wires).
# MediaPipe: 0=nose, 11=L shoulder, 12=R shoulder, 13=L elbow, 14=R elbow, 15=L wrist, 16=R wrist,
#            23=L hip, 24=R hip, 25=L knee, 26=R knee, 27=L ankle, 28=R ankle
BODY_SEGMENTS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
]
HEAD_LANDMARK = 0
HEAD_RADIUS = 0.06 * WORLD_SCALE   # scale with figure
# Neck: shoulder_mid → mid-neck (below head), not all the way to nose (avoids tilt skew)
NECK_FRACTION = 0.65       # neck ends 65% toward head from shoulders


def _transform_world_coords(xs, ys, zs):
    """Flip z, scale, center on hips; optionally swap axes for Matplotlib."""
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    zs = np.array(zs, dtype=float)
    if WORLD_FLIP_Z:
        zs = -zs
    xs, ys, zs = xs * WORLD_SCALE, ys * WORLD_SCALE, zs * WORLD_SCALE
    if WORLD_CENTER_HIPS and len(xs) > 24:
        hip_mid_x = (xs[23] + xs[24]) / 2
        hip_mid_y = (ys[23] + ys[24]) / 2
        hip_mid_z = (zs[23] + zs[24]) / 2
        xs, ys, zs = xs - hip_mid_x, ys - hip_mid_y, zs - hip_mid_z
    if WORLD_AXIS_SWAP == "yz":
        xs, ys, zs = xs, zs, ys
    return xs.tolist(), ys.tolist(), zs.tolist()


def _smooth_landmarks(prev, curr, alpha=SMOOTH_ALPHA):
    """Exponential smoothing over landmark arrays. curr/prev are (N,3) or (xs,ys,zs)."""
    if prev is None or len(prev[0]) != len(curr[0]):
        return curr
    out = (
        [alpha * c + (1 - alpha) * p for c, p in zip(curr[0], prev[0])],
        [alpha * c + (1 - alpha) * p for c, p in zip(curr[1], prev[1])],
        [alpha * c + (1 - alpha) * p for c, p in zip(curr[2], prev[2])],
    )
    return out


def _capsule_radius(p1, p2, base=0.02, frac=0.05):
    """Radius proportional to segment length for consistent limb thickness."""
    dist = np.linalg.norm(np.array(p2) - np.array(p1))
    return max(base, frac * dist)


def _build_human_mesh(xs, ys, zs):
    """Build coherent human: torso, arms, legs (capsules, radii ∝ length) + head + neck."""
    pts = np.column_stack([xs, ys, zs])
    triangles = []
    for (i, j) in BODY_SEGMENTS:
        r = _capsule_radius(pts[i], pts[j], base=0.02, frac=0.05)
        triangles.extend(_capsule_triangles(pts[i], pts[j], r, n_theta=12))
    head_center = pts[HEAD_LANDMARK]
    shoulder_mid = (pts[11] + pts[12]) * 0.5
    # Neck: shoulder_mid → point partway to head (avoids nose tilt skew)
    neck_top = shoulder_mid + NECK_FRACTION * (head_center - shoulder_mid)
    r_neck = _capsule_radius(shoulder_mid, neck_top, base=0.02, frac=0.05)
    triangles.extend(_capsule_triangles(shoulder_mid, neck_top, r_neck, n_theta=10))
    triangles.extend(_sphere_triangles(head_center, HEAD_RADIUS, n=10))
    return triangles


def _get_pose_model_path(model_type="full"):
    """Same as Compy.get_pose_model_path: return path to pose landmarker model."""
    import urllib.request
    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"pose_landmarker_{model_type}.task"
    if not model_path.exists():
        url = f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{model_type}/float16/latest/pose_landmarker_{model_type}.task"
        print(f"Downloading pose landmarker {model_type} model...")
        urllib.request.urlretrieve(url, model_path)
        print("Done.")
    return str(model_path)


# Pose skeleton connections (same as MediaPipe PoseLandmarksConnections.POSE_LANDMARKS)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


def world_landmarks_to_arrays(pose_world_landmarks, transform=True):
    """Convert pose_world_landmarks to (xs, ys, zs). If transform=True, apply flip/scale/center."""
    if not pose_world_landmarks:
        return None
    xs = [p.x for p in pose_world_landmarks]
    ys = [p.y for p in pose_world_landmarks]
    zs = [p.z for p in pose_world_landmarks]
    if transform:
        xs, ys, zs = _transform_world_coords(xs, ys, zs)
    return xs, ys, zs


def run_viewer(camera_id=0, model_type="full", detect_every_n=2):
    import cv2
    import mediapipe as mp
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D, art3d

    # MediaPipe setup (same as Compy)
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    model_path = _get_pose_model_path(model_type)
    base_opts = BaseOptions(model_asset_path=model_path)
    base_opts.delegate = BaseOptions.Delegate.CPU  # 3D viewer: keep CPU for compatibility
    options = PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose = PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Could not open camera.")
        pose.close()
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval_ms = 1000 / fps

    # 3D figure: animated human (capsule limbs + head)
    plt.ion()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Animated Human (pose from camera)")
    # Scaled/centered coords: use ±1 for view volume
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_facecolor((0.15, 0.15, 0.2))

    human_mesh = art3d.Poly3DCollection([], facecolor=(0.9, 0.75, 0.7), edgecolor="none", shade=False)
    ax.add_collection3d(human_mesh)

    frame_count = 0
    last_results = None
    last_smoothed = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % detect_every_n == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                timestamp_ms = int(frame_count * frame_interval_ms)
                last_results = pose.detect_for_video(mp_image, timestamp_ms)
            results = last_results

            if results and results.pose_world_landmarks:
                wlm = results.pose_world_landmarks[0]
                xs, ys, zs = world_landmarks_to_arrays(wlm, transform=True)
                curr = (xs, ys, zs)
                curr = _smooth_landmarks(last_smoothed, curr)
                last_smoothed = curr
                xs, ys, zs = curr
                triangles = _build_human_mesh(xs, ys, zs)
                human_mesh.set_verts(triangles)
                human_mesh.set_facecolors([(0.9, 0.75, 0.7)] * len(triangles))
            else:
                human_mesh.set_verts([])
                human_mesh.set_facecolors([])

            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

            frame_count += 1
            if plt.get_fignums() == []:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        pose.close()
        plt.ioff()
        plt.close()


def run_replay(jsonl_path, speed=1.0):
    """Replay 3D animated human from pose_log.jsonl (Compy with SAVE_WORLD_COORDS=True)."""
    import json
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D, art3d

    path = Path(jsonl_path)
    if not path.exists():
        print(f"File not found: {path}")
        return
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not entries:
        print("No valid pose entries. Run Compy with SAVE_WORLD_COORDS=True and record.")
        return
    if "world_landmarks" not in entries[0]:
        print("Log has no world_landmarks. Run Compy with SAVE_WORLD_COORDS=True.")
        return

    plt.ion()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Animated Human Replay")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_facecolor((0.15, 0.15, 0.2))
    human_mesh = art3d.Poly3DCollection([], facecolor=(0.9, 0.75, 0.7), edgecolor="none", shade=False)
    ax.add_collection3d(human_mesh)

    last_smoothed = None
    try:
        for entry in entries:
            wlm = entry.get("world_landmarks")
            if not wlm or len(wlm) < 33:
                continue
            xs = [p[0] for p in wlm]
            ys = [p[1] for p in wlm]
            zs = [p[2] for p in wlm]
            xs, ys, zs = _transform_world_coords(xs, ys, zs)
            curr = (xs, ys, zs)
            curr = _smooth_landmarks(last_smoothed, curr)
            last_smoothed = curr
            xs, ys, zs = curr
            triangles = _build_human_mesh(xs, ys, zs)
            human_mesh.set_verts(triangles)
            human_mesh.set_facecolors([(0.9, 0.75, 0.7)] * len(triangles))
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.03 / speed)
    except KeyboardInterrupt:
        pass
    plt.ioff()
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="3D pose viewer using same pipeline as Compy")
    parser.add_argument("--camera", type=int, default=0, help="Camera device id")
    parser.add_argument("--model", choices=("lite", "full", "heavy"), default="full", help="Pose model")
    parser.add_argument("--skip", type=int, default=2, help="Run pose every N frames")
    parser.add_argument("--replay", type=str, metavar="JSONL", help="Replay from pose_log.jsonl (Compy with SAVE_WORLD_COORDS=True)")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed (only with --replay)")
    args = parser.parse_args()
    if args.replay:
        run_replay(args.replay, speed=args.speed)
    else:
        run_viewer(camera_id=args.camera, model_type=args.model, detect_every_n=args.skip)
