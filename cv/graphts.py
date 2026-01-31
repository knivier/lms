#!/usr/bin/env python3
"""
3D animated human model using cv.py core (PoseCore for detection).
Equivalent to pose_3d_viewer.py but built on cv.py.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d

from cv import PoseCore


# World coordinate transform (MediaPipe → viewable 3D)
WORLD_SCALE = 2.0
WORLD_FLIP_Z = True
WORLD_CENTER_HIPS = True
WORLD_AXIS_SWAP = None  # None or "yz"
SMOOTH_ALPHA = 0.4

BODY_SEGMENTS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
]
HEAD_LANDMARK = 0
HEAD_RADIUS = 0.06 * WORLD_SCALE
NECK_FRACTION = 0.65


def _transform_world_coords(xs, ys, zs):
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


def _smooth_world(prev, curr, alpha=SMOOTH_ALPHA):
    if prev is None or len(prev[0]) != len(curr[0]):
        return curr
    out = (
        [alpha * c + (1 - alpha) * p for c, p in zip(curr[0], prev[0])],
        [alpha * c + (1 - alpha) * p for c, p in zip(curr[1], prev[1])],
        [alpha * c + (1 - alpha) * p for c, p in zip(curr[2], prev[2])],
    )
    return out


def _capsule_triangles(p1, p2, radius, n_theta=10):
    p1, p2 = np.array(p1), np.array(p2)
    vec = p2 - p1
    length = np.linalg.norm(vec)
    if length < 1e-6:
        return []
    z_axis = vec / length
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    z = np.linspace(0, length, 4)
    verts_local = []
    for zi in z:
        for ti in theta:
            verts_local.append([radius * np.cos(ti), radius * np.sin(ti), zi])
    verts_local = np.array(verts_local)
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


def _capsule_radius(p1, p2, base=0.02, frac=0.05):
    dist = np.linalg.norm(np.array(p2) - np.array(p1))
    return max(base, frac * dist)


def _build_human_mesh(xs, ys, zs):
    pts = np.column_stack([xs, ys, zs])
    triangles = []
    for (i, j) in BODY_SEGMENTS:
        r = _capsule_radius(pts[i], pts[j], base=0.02, frac=0.05)
        triangles.extend(_capsule_triangles(pts[i], pts[j], r, n_theta=12))
    head_center = pts[HEAD_LANDMARK]
    shoulder_mid = (pts[11] + pts[12]) * 0.5
    neck_top = shoulder_mid + NECK_FRACTION * (head_center - shoulder_mid)
    r_neck = _capsule_radius(shoulder_mid, neck_top, base=0.02, frac=0.05)
    triangles.extend(_capsule_triangles(shoulder_mid, neck_top, r_neck, n_theta=10))
    triangles.extend(_sphere_triangles(head_center, HEAD_RADIUS, n=10))
    return triangles


def run_viewer(camera_id=0):
    """All pipeline options (model_type, detect_every_n, etc.) come from cv.py config."""
    try:
        core = PoseCore(camera_id=camera_id)
    except RuntimeError as exc:
        print(exc)
        return

    plt.ion()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Animated Human (pose from camera)")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_facecolor((0.15, 0.15, 0.2))
    human_mesh = art3d.Poly3DCollection([], facecolor=(0.9, 0.75, 0.7), edgecolor="none", shade=False)
    ax.add_collection3d(human_mesh)

    last_smoothed = None
    try:
        while True:
            data = core.step()
            if data is None:
                break
            wlm = data["world_landmarks"]
            if wlm:
                xs = [p.x for p in wlm]
                ys = [p.y for p in wlm]
                zs = [p.z for p in wlm]
                xs, ys, zs = _transform_world_coords(xs, ys, zs)
                curr = (xs, ys, zs)
                curr = _smooth_world(last_smoothed, curr)
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
            if plt.get_fignums() == []:
                break
    except KeyboardInterrupt:
        pass
    finally:
        core.close()
        plt.ioff()
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D viewer (all options from cv.py / config.yaml).")
    parser.add_argument("--camera", type=int, default=0, help="Camera device id")
    args = parser.parse_args()
    run_viewer(camera_id=args.camera)
