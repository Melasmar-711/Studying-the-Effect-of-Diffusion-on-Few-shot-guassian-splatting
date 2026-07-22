"""Placement geometry for Zero123 novel views (VERIFIED, azimuth-only).

The real capture is NOT a clean single-axis orbit globally, but LOCALLY (small
azimuth offsets from a source frame) rotating the whole source camera frame about
the scene's orbit axis reproduces true poses to ~2-3 deg (verified against held-out
poses in scripts/verify_zero123_pose.py). Elevation offsets are NOT reliable for
this near-top-down capture, so we only offset azimuth.

Orbit frame: center = least-squares intersection of camera view rays; axis =
smallest-spread PCA direction of camera positions (the turntable axis). Zero123's
azimuth sign matches this azimuth (verified: azim*+1, no flip).
"""
from __future__ import annotations

import numpy as np


def orbit_frame(frames):
    """Return (center C, up axis) for a list of frames (each has transform_matrix)."""
    T = np.array([np.array(f["transform_matrix"]) for f in frames])
    pos = T[:, :3, 3]
    # center = LS intersection of view rays (camera looks down -z)
    A = np.zeros((3, 3)); b = np.zeros(3)
    for M in T:
        o = M[:3, 3]; d = -M[:3, 2]; d = d / np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d); A += P; b += P @ o
    C = np.linalg.solve(A, b)
    v = pos - C
    _, _, vt = np.linalg.svd(v - v.mean(0))
    up = vt[2] / np.linalg.norm(vt[2])
    if (v @ up).mean() < 0:
        up = -up
    return C, up


def _rot_axis(axis, deg):
    axis = axis / np.linalg.norm(axis); a = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def object_centroid(ply_path):
    """Trimmed centroid of the sparse cloud = object 3D center (for projection)."""
    from plyfile import PlyData
    v = PlyData.read(str(ply_path))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1)
    dist = np.linalg.norm(xyz - np.median(xyz, 0), axis=1)
    return xyz[dist < np.percentile(dist, 80)].mean(0)


def project_point(O, c2w, fx, fy, cx, cy):
    """Project world point O into image pixels through a nerfstudio c2w (camera
    looks down -z, +y up). Verified vs real mask centroids to ~15-50px."""
    w2c = np.linalg.inv(np.array(c2w, float))
    x, y, z = (w2c @ np.array([*O, 1.0]))[:3]
    return cx + fx * (x / -z), cy - fy * (y / -z)


def novel_pose_azimuth(src_c2w, C, up, d_azim_deg):
    """Camera-to-world for a view azimuth-offset by ``d_azim_deg`` from src, by
    rotating the WHOLE source camera frame about the orbit axis through C (this
    preserves the source roll, which a fresh look-at would get wrong ~27 deg)."""
    src = np.array(src_c2w, dtype=float)
    R = _rot_axis(up, d_azim_deg)
    out = np.eye(4)
    out[:3, :3] = R @ src[:3, :3]
    out[:3, 3] = C + R @ (src[:3, 3] - C)
    return out.tolist()
