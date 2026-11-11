"""Joint constants, action scaling, and Three.js<->MuJoCo orientation mapping for Steel Sentinel."""

from __future__ import annotations

import math
import numpy as np

JOINT_NAMES = [
    "hy", "hx", "wt",
    "rs", "rr", "re",
    "ls", "lr", "le",
    "rh", "rha", "rk",
    "lh", "lha", "lk",
    "ra", "la",
]

J_MIN = np.array(
    [-1.0, -0.6, -0.8, -2.5, 0.0, 0.0, -2.5, 0.0, 0.0, -1.2, -0.45, 0.0, -1.2, -0.45, 0.0, -0.8, -0.8],
    dtype=np.float32,
)
J_MAX = np.array(
    [1.0, 0.6, 0.8, 2.5, 2.8, 2.4, 2.5, 2.8, 2.4, 1.2, 0.85, 2.2, 1.2, 0.85, 2.2, 0.8, 0.8],
    dtype=np.float32,
)
# Neutral upright standing posture (used only as center for PD target offset and weak posture regularization)
J_NOM = np.array(
    [0.0, 0.0, 0.0, 0.02, 0.10, 0.35, 0.02, 0.10, 0.35, 0.12, 0.04, 0.24, 0.12, 0.04, 0.24, -0.12, -0.12],
    dtype=np.float32,
)
# Full RL action authority per joint (Pure PPO controls all joints directly)
ACT_SCALE = np.array(
    [0.50, 0.50, 0.65, 1.40, 1.00, 0.85, 1.40, 1.00, 0.85, 0.95, 0.50, 1.15, 0.95, 0.50, 1.15, 0.65, 0.65],
    dtype=np.float32,
)


def mj_quat_to_three_eul_yxz(quat_wxyz: np.ndarray) -> tuple[float, float, float]:
    """Convert MuJoCo hips quaternion (facing +X_mj, +Z_mj up) to Three.js YXZ Euler [pitch, yaw, roll]."""
    w, x, y, z = float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r10 = 2.0 * (x * y + w * z)
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)

    pitch = math.asin(max(-1.0, min(1.0, -r20)))
    yaw = math.atan2(r00, -r10)
    roll = math.atan2(r21, r22)
    return pitch, yaw, roll


_INV_SQRT2 = math.sqrt(0.5)


def mj_quat_to_three_quat_xyzw(quat_wxyz: np.ndarray) -> tuple[float, float, float, float]:
    """Convert MuJoCo hips quaternion [w,x,y,z] directly into singularity-free Three.js [qx,qy,qz,qw]."""
    w, x, y, z = float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
    wx, xx, yx, zx = w, x, z, -y
    qw = _INV_SQRT2 * (wx - yx)
    qx = _INV_SQRT2 * (xx - zx)
    qy = _INV_SQRT2 * (wx + yx)
    qz = _INV_SQRT2 * (zx + xx)
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) or 1.0
    return qx / norm, qy / norm, qz / norm, qw / norm
