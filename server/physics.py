"""Vectorized 3D articulated biped rigid-body physics on PyTorch CUDA/CPU."""

from __future__ import annotations

import math
import torch

JOINT_NAMES = [
    "hy", "hx", "wt",
    "rs", "rr", "re",
    "ls", "lr", "le",
    "rh", "rk", "lh", "lk", "ra", "la",
]

# [min, max, sign] matching src/robot/model.js
JOINT_MIN = torch.tensor(
    [-1.0, -0.6, -0.8, -2.5, 0.0, 0.0, -2.5, 0.0, 0.0, -1.2, 0.0, -1.2, 0.0, -0.8, -0.8],
    dtype=torch.float32,
)
JOINT_MAX = torch.tensor(
    [1.0, 0.6, 0.8, 2.5, 2.8, 2.4, 2.5, 2.8, 2.4, 1.2, 2.2, 1.2, 2.2, 0.8, 0.8],
    dtype=torch.float32,
)
JOINT_SIGN = torch.tensor(
    [1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0],
    dtype=torch.float32,
)
# Neutral balanced standing stance (action = 0 commands upright posture with flat feet)
JOINT_NOM = torch.tensor(
    [0.0, 0.0, 0.0, 0.05, 0.14, 0.25, 0.05, 0.14, 0.25, 0.14, 0.28, 0.14, 0.28, -0.14, -0.14],
    dtype=torch.float32,
)
ACTION_SCALE = torch.tensor(
    [0.5, 0.4, 0.5, 0.9, 0.65, 0.7, 0.9, 0.65, 0.7, 0.75, 0.85, 0.75, 0.85, 0.55, 0.55],
    dtype=torch.float32,
)

# Per-joint PD gains (stronger on legs/ankles, lighter on arms/head)
KP = torch.tensor(
    [10.0, 10.0, 18.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 52.0, 56.0, 52.0, 56.0, 42.0, 42.0],
    dtype=torch.float32,
)
KD = torch.tensor(
    [1.5, 1.5, 2.4, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 5.2, 5.6, 5.2, 5.6, 4.4, 4.4],
    dtype=torch.float32,
)
TORQUE_LIMIT = torch.tensor(
    [4.0, 4.0, 8.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 32.0, 34.0, 32.0, 34.0, 26.0, 26.0],
    dtype=torch.float32,
)
JOINT_INERTIA = torch.tensor(
    [0.06, 0.06, 0.14, 0.09, 0.09, 0.07, 0.09, 0.09, 0.07, 0.20, 0.18, 0.20, 0.18, 0.12, 0.12],
    dtype=torch.float32,
)

ROAD_TOP_Y = 0.12
GRAVITY = 18.0
MASS = 1.0
MAX_EPISODE_STEPS = 480


def _rot_yxz(pitch: torch.Tensor, yaw: torch.Tensor, roll: torch.Tensor) -> torch.Tensor:
    """Build [N, 3, 3] rotation matrices matching Three.js Euler order 'YXZ'."""
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)

    # R = Ry(yaw) @ Rx(pitch) @ Rz(roll)
    r00 = cy * cr + sy * sp * sr
    r01 = sy * sp * cr - cy * sr
    r02 = sy * cp
    r10 = cp * sr
    r11 = cp * cr
    r12 = -sp
    r20 = cy * sp * sr - sy * cr
    r21 = sy * sr + cy * sp * cr
    r22 = cy * cp

    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _wrap_angle(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class BipedPhysicsEnv:
    """Batched 3D articulated biped physics simulator running on PyTorch CUDA/CPU."""

    OBS_DIM = 45
    ACT_DIM = 15

    def __init__(self, num_envs: int = 16, device: torch.device | None = None):
        self.num_envs = num_envs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.j_min = JOINT_MIN.to(self.device)
        self.j_max = JOINT_MAX.to(self.device)
        self.j_nom = JOINT_NOM.to(self.device)
        self.act_scale = ACTION_SCALE.to(self.device)
        self.j_mid = 0.5 * (self.j_min + self.j_max)
        self.j_half = 0.5 * (self.j_max - self.j_min)
        self.j_sign = JOINT_SIGN.to(self.device)
        self.kp = KP.to(self.device)
        self.kd = KD.to(self.device)
        self.tau_lim = TORQUE_LIMIT.to(self.device)
        self.j_inertia = JOINT_INERTIA.to(self.device)

        # Effective body inertia around COM [pitch_x, yaw_y, roll_z]
        self.body_inertia = torch.tensor([1.25, 0.60, 1.40], device=self.device)
        self.com_local = torch.tensor([0.0, 0.95, 0.0], device=self.device)
        self.body_radii = torch.tensor([0.22, 0.22, 0.34, 0.45, 0.42], device=self.device).view(1, 5)

        # Foot corner offsets in foot frame: [4, 3] (heel/toe x inner/outer)
        self.foot_corners = torch.tensor(
            [
                [-0.20, -0.10, -0.24],
                [0.20, -0.10, -0.24],
                [-0.20, -0.10, 0.62],
                [0.20, -0.10, 0.62],
            ],
            device=self.device,
        )

        self.stage = 0  # 0 = stand/balance, 1 = walk
        self.target_speed = 2.6

        # State tensors
        n = num_envs
        self.pos = torch.zeros(n, 3, device=self.device)
        self.vel = torch.zeros(n, 3, device=self.device)
        self.eul = torch.zeros(n, 3, device=self.device)  # [pitch, yaw, roll]
        self.ang_vel = torch.zeros(n, 3, device=self.device)  # world angular vel
        self.q = torch.zeros(n, self.ACT_DIM, device=self.device)
        self.qd = torch.zeros(n, self.ACT_DIM, device=self.device)
        self.q_cmd = torch.zeros(n, self.ACT_DIM, device=self.device)
        self.prev_act = torch.zeros(n, self.ACT_DIM, device=self.device)
        self.step_count = torch.zeros(n, dtype=torch.long, device=self.device)
        self.foot_contact = torch.zeros(n, 2, device=self.device)
        self.foot_heights = torch.zeros(n, 2, device=self.device)
        self.gait_phase = torch.zeros(n, device=self.device)
        self.ep_reward = torch.zeros(n, device=self.device)
        self.ep_upright = torch.zeros(n, device=self.device)
        self.start_x = torch.zeros(n, device=self.device)

        self.reset()

    def set_stage(self, stage: int) -> None:
        self.stage = 1 if stage >= 1 else 0

    def apply_impulse(self, dvx: float = 0.0, dvz: float = 0.0, env_idx: int = 0) -> None:
        """Apply a physical external push impulse to test balance recovery."""
        self.vel[env_idx, 0] += dvx
        self.vel[env_idx, 2] += dvz
        self.ang_vel[env_idx, 0] += dvx * 0.4
        self.ang_vel[env_idx, 2] -= dvz * 0.4

    def reset(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        k = env_ids.numel()
        if k == 0:
            return self.get_obs()

        # Spawn upright near road center facing +x (yaw = pi/2), slight perturbation
        self.pos[env_ids, 0] = 0.0
        self.pos[env_ids, 1] = 2.68 + (torch.rand(k, device=self.device) - 0.5) * 0.05
        self.pos[env_ids, 2] = (torch.rand(k, device=self.device) - 0.5) * 0.4
        self.start_x[env_ids] = self.pos[env_ids, 0]

        self.vel[env_ids, 0] = (torch.rand(k, device=self.device) - 0.5) * 0.25
        self.vel[env_ids, 1] = 0.0
        self.vel[env_ids, 2] = (torch.rand(k, device=self.device) - 0.5) * 0.25

        self.eul[env_ids, 0] = (torch.rand(k, device=self.device) - 0.5) * 0.14  # pitch
        self.eul[env_ids, 1] = math.pi / 2.0 + (torch.rand(k, device=self.device) - 0.5) * 0.12  # yaw
        self.eul[env_ids, 2] = (torch.rand(k, device=self.device) - 0.5) * 0.14  # roll

        self.ang_vel[env_ids] = (torch.rand(k, 3, device=self.device) - 0.5) * 0.35

        # Neutral slightly-bent athletic stance with slight noise
        q0 = self.j_nom.unsqueeze(0).expand(k, -1).clone()
        q0 += (torch.rand(k, self.ACT_DIM, device=self.device) - 0.5) * 0.08
        self.q[env_ids] = torch.clamp(q0, self.j_min, self.j_max)
        self.q_cmd[env_ids] = self.q[env_ids]
        self.qd[env_ids] = 0.0
        self.prev_act[env_ids] = 0.0
        self.step_count[env_ids] = 0
        self.gait_phase[env_ids] = torch.rand(k, device=self.device) * 2.0 * math.pi
        self.ep_reward[env_ids] = 0.0
        self.ep_upright[env_ids] = 0.0

        # Place feet cleanly onto the road surface with zero initial penetration spike
        _, _, _, _, _, _, p_corners, _, _ = self._compute_fk()
        min_foot_y = p_corners[env_ids, ..., 1].amin(dim=(1, 2))
        self.pos[env_ids, 1] += (ROAD_TOP_Y + 0.004) - min_foot_y
        self._update_foot_metrics()
        return self.get_obs()

    def _compute_fk(self):
        """Exact 3D forward kinematics for leg joints, foot corners, and body collision spheres."""
        R_root = _rot_yxz(self.eul[:, 0], self.eul[:, 1], self.eul[:, 2])  # [N, 3, 3]
        axis_x_local = R_root[:, :, 0]  # [N, 3] world direction of robot's local X axis

        n = self.num_envs
        # Build both legs simultaneously [N, 2, ...] (0=Right s=+1, 1=Left s=-1)
        hip_x_off = torch.tensor([0.3, -0.3], device=self.device).view(1, 2, 1)
        hip_local = torch.zeros(n, 2, 3, device=self.device)
        hip_local[:, :, 0] = hip_x_off.squeeze(-1)
        hip_local[:, :, 1] = -0.15
        p_hip = self.pos.unsqueeze(1) + torch.einsum("nij,nlj->nli", R_root, hip_local)

        # Signed sagittal angles (model.js convention: rot_x = val * sign)
        a_hip = torch.stack([self.q[:, 9] * -1.0, self.q[:, 11] * -1.0], dim=1)
        a_knee = a_hip + torch.stack([self.q[:, 10], self.q[:, 12]], dim=1)
        a_foot = a_knee + torch.stack([self.q[:, 13], self.q[:, 14]], dim=1)

        # Thigh vector in root frame: Rx(a_hip) @ [0, -1.2, 0] = [0, -1.2*cos, -1.2*sin]
        thigh_local = torch.stack(
            [torch.zeros_like(a_hip), -1.2 * torch.cos(a_hip), -1.2 * torch.sin(a_hip)], dim=-1
        )
        p_knee = p_hip + torch.einsum("nij,nlj->nli", R_root, thigh_local)

        # Shin vector in root frame: Rx(a_knee) @ [0, -1.15, 0]
        shin_local = torch.stack(
            [torch.zeros_like(a_knee), -1.15 * torch.cos(a_knee), -1.15 * torch.sin(a_knee)], dim=-1
        )
        p_ankle = p_knee + torch.einsum("nij,nlj->nli", R_root, shin_local)

        # Foot corners [N, 2, 4, 3]: rotate foot_corners [4, 3] by Rx(a_foot) then R_root
        ca = torch.cos(a_foot).unsqueeze(-1)  # [N, 2, 1]
        sa = torch.sin(a_foot).unsqueeze(-1)
        fc = self.foot_corners.view(1, 1, 4, 3)
        fy = fc[..., 1] * ca - fc[..., 2] * sa
        fz = fc[..., 1] * sa + fc[..., 2] * ca
        fx = fc[..., 0].expand_as(fy)
        corners_local = torch.stack([fx, fy, fz], dim=-1)  # [N, 2, 4, 3]
        p_corners = p_ankle.unsqueeze(2) + torch.einsum("nij,nlcj->nlci", R_root, corners_local)

        # Body collision points in world (chest, head)
        chest_local = torch.tensor([0.0, 1.35, 0.0], device=self.device).view(1, 3)
        head_local = torch.tensor([0.0, 2.35, 0.0], device=self.device).view(1, 3)
        p_chest = self.pos + torch.einsum("nij,nj->ni", R_root, chest_local)
        p_head = self.pos + torch.einsum("nij,nj->ni", R_root, head_local)
        p_com = self.pos + torch.einsum("nij,j->ni", R_root, self.com_local)

        return R_root, axis_x_local, p_com, p_hip, p_knee, p_ankle, p_corners, p_chest, p_head

    def _update_foot_metrics(self):
        _, _, _, _, _, p_ankle, p_corners, _, _ = self._compute_fk()
        min_y = p_corners[..., 1].min(dim=-1).values  # [N, 2]
        self.foot_heights = (min_y - ROAD_TOP_Y).clamp(min=0.0)
        self.foot_contact = (min_y <= ROAD_TOP_Y + 0.02).float()

    def step(self, action: torch.Tensor, dt_ctrl: float = 1.0 / 45.0, substeps: int = 4):
        """Advance physics and return (obs, reward, done, info)."""
        action = torch.clamp(action, -1.0, 1.0)
        q_target_raw = torch.clamp(self.j_nom + action * self.act_scale, self.j_min, self.j_max)
        self.q_cmd = self.q_cmd + 0.45 * (q_target_raw - self.q_cmd)
        q_target = self.q_cmd
        dt = dt_ctrl / substeps

        total_contact = torch.zeros(self.num_envs, 2, device=self.device)
        k_n = 650.0
        c_n = 48.0
        k_f = 130.0
        mu = 1.15

        for _ in range(substeps):
            R_root, axis_x, p_com, p_hip, p_knee, p_ankle, p_corners, p_chest, p_head = self._compute_fk()

            # Joint PD servo torques
            tau_pd = torch.clamp(
                self.kp * (q_target - self.q) - self.kd * self.qd,
                -self.tau_lim,
                self.tau_lim,
            )

            # Foot corner velocities including joint angular velocity contributions
            # qd signs for leg sagittal axes
            qd_rh, qd_rk, qd_ra = self.qd[:, 9] * -1.0, self.qd[:, 10], self.qd[:, 13]
            qd_lh, qd_lk, qd_la = self.qd[:, 11] * -1.0, self.qd[:, 12], self.qd[:, 14]
            w_hip = torch.stack([qd_rh, qd_lh], dim=1).unsqueeze(-1) * axis_x.unsqueeze(1)  # [N,2,3]
            w_knee = torch.stack([qd_rk, qd_lk], dim=1).unsqueeze(-1) * axis_x.unsqueeze(1)
            w_ank = torch.stack([qd_ra, qd_la], dim=1).unsqueeze(-1) * axis_x.unsqueeze(1)

            r_com_c = p_corners - p_com.unsqueeze(1).unsqueeze(2)
            v_root_c = self.vel.unsqueeze(1).unsqueeze(2) + torch.cross(
                self.ang_vel.unsqueeze(1).unsqueeze(2).expand_as(r_com_c), r_com_c, dim=-1
            )
            v_joint_c = (
                torch.cross(w_hip.unsqueeze(2).expand_as(r_com_c), p_corners - p_hip.unsqueeze(2), dim=-1)
                + torch.cross(w_knee.unsqueeze(2).expand_as(r_com_c), p_corners - p_knee.unsqueeze(2), dim=-1)
                + torch.cross(w_ank.unsqueeze(2).expand_as(r_com_c), p_corners - p_ankle.unsqueeze(2), dim=-1)
            )
            v_c = v_root_c + v_joint_c

            pen = (ROAD_TOP_Y - p_corners[..., 1]).clamp(min=0.0)
            active = (pen > 0.0).float()
            fy = (k_n * pen - c_n * v_c[..., 1]).clamp(min=0.0) * active

            # Tangential Coulomb friction
            fx_raw = -k_f * v_c[..., 0] * active
            fz_raw = -k_f * v_c[..., 2] * active
            f_horiz_norm = torch.sqrt(fx_raw * fx_raw + fz_raw * fz_raw + 1e-8)
            f_max = mu * fy
            scale = torch.minimum(torch.ones_like(f_horiz_norm), f_max / f_horiz_norm)
            F_c = torch.stack([fx_raw * scale, fy, fz_raw * scale], dim=-1)  # [N, 2, 4, 3]

            total_contact += fy.sum(dim=-1) / substeps

            # Net ground force & moment on root COM
            F_net = F_c.sum(dim=(1, 2))
            M_net = torch.cross(r_com_c, F_c, dim=-1).sum(dim=(1, 2))

            # Vectorized body sphere safety contacts (knees, hips, chest, head) against road
            pts = torch.stack([p_knee[:, 0], p_knee[:, 1], self.pos, p_chest, p_head], dim=1)
            p_pen = (ROAD_TOP_Y + self.body_radii - pts[..., 1]).clamp(min=0.0)
            p_act = (p_pen > 0.0).float()
            r_pt = pts - p_com.unsqueeze(1)
            v_pt = self.vel.unsqueeze(1) + torch.cross(
                self.ang_vel.unsqueeze(1).expand_as(r_pt), r_pt, dim=-1
            )
            p_fy = (500.0 * p_pen - 40.0 * v_pt[..., 1]).clamp(min=0.0) * p_act
            p_fx = torch.clamp(-80.0 * v_pt[..., 0] * p_act, -0.8 * p_fy, 0.8 * p_fy)
            p_fz = torch.clamp(-80.0 * v_pt[..., 2] * p_act, -0.8 * p_fy, 0.8 * p_fy)
            p_F = torch.stack([p_fx, p_fy, p_fz], dim=-1)
            F_net = F_net + p_F.sum(dim=1)
            M_net = M_net + torch.cross(r_pt, p_F, dim=-1).sum(dim=1)

            # Gravity
            F_net[:, 1] -= MASS * GRAVITY

            # Action-reaction joint contact torques for legs: tau_j = sum_c ((p_c - p_j) x F_c) . axis_x
            tau_c_sum = torch.cross(p_corners - p_hip.unsqueeze(2), F_c, dim=-1).sum(dim=2)
            tau_c_hip = (tau_c_sum * axis_x.unsqueeze(1)).sum(dim=-1)  # [N, 2]
            tau_c_knee = (
                torch.cross(p_corners - p_knee.unsqueeze(2), F_c, dim=-1).sum(dim=2) * axis_x.unsqueeze(1)
            ).sum(dim=-1)
            tau_c_ank = (
                torch.cross(p_corners - p_ankle.unsqueeze(2), F_c, dim=-1).sum(dim=2) * axis_x.unsqueeze(1)
            ).sum(dim=-1)

            # Convert contact reaction into joint coordinate space (sign-aware, opposes penetration push)
            tau_ext = torch.zeros_like(tau_pd)
            tau_ext[:, 9] = tau_c_hip[:, 0] * -1.0
            tau_ext[:, 11] = tau_c_hip[:, 1] * -1.0
            tau_ext[:, 10] = tau_c_knee[:, 0]
            tau_ext[:, 12] = tau_c_knee[:, 1]
            tau_ext[:, 13] = tau_c_ank[:, 0]
            tau_ext[:, 14] = tau_c_ank[:, 1]

            # Slight arm/waist angular-momentum counterbalance reaction on torso pitch/yaw
            arm_pitch_react = -0.05 * (tau_pd[:, 3] + tau_pd[:, 6])
            arm_roll_react = -0.05 * (tau_pd[:, 4] - tau_pd[:, 7])
            waist_yaw_react = -0.08 * tau_pd[:, 2]
            M_local = torch.einsum("nji,nj->ni", R_root, M_net)
            M_local[:, 0] += arm_pitch_react
            M_local[:, 1] += waist_yaw_react
            M_local[:, 2] += arm_roll_react

            # Integrate root linear & angular velocity
            acc = F_net / MASS
            self.vel = torch.clamp(self.vel + acc * dt, -18.0, 18.0)
            self.pos = self.pos + self.vel * dt
            self.pos[:, 0] = torch.clamp(self.pos[:, 0], -325.0, 325.0)
            self.pos[:, 1] = torch.clamp(self.pos[:, 1], 0.35, 8.0)

            ang_acc_local = M_local / self.body_inertia - 0.8 * torch.einsum("nji,nj->ni", R_root, self.ang_vel)
            ang_acc_world = torch.einsum("nij,nj->ni", R_root, ang_acc_local)
            self.ang_vel = torch.clamp(self.ang_vel + ang_acc_world * dt, -14.0, 14.0)

            # Map world angular velocity to YXZ Euler rates
            w_loc = torch.einsum("nji,nj->ni", R_root, self.ang_vel)
            cp = torch.cos(self.eul[:, 0]).clamp(min=0.15)
            sp = torch.sin(self.eul[:, 0])
            cr = torch.cos(self.eul[:, 2])
            sr = torch.sin(self.eul[:, 2])
            # Euler kinematics for YXZ
            pitch_dot = w_loc[:, 0] * cr - w_loc[:, 1] * sr
            yaw_dot = (w_loc[:, 1] * cr + w_loc[:, 0] * sr) / cp
            roll_dot = w_loc[:, 2] + yaw_dot * sp
            self.eul[:, 0] = torch.clamp(self.eul[:, 0] + pitch_dot * dt, -1.5, 1.5)
            self.eul[:, 1] = _wrap_angle(self.eul[:, 1] + yaw_dot * dt)
            self.eul[:, 2] = torch.clamp(self.eul[:, 2] + roll_dot * dt, -1.5, 1.5)

            # Integrate joints with physical limits
            qdd = (tau_pd + 0.18 * tau_ext - 0.35 * self.qd) / self.j_inertia
            self.qd = torch.clamp(self.qd + qdd * dt, -16.0, 16.0)
            q_next = self.q + self.qd * dt
            hit_lo = q_next < self.j_min
            hit_hi = q_next > self.j_max
            self.qd = torch.where(hit_lo | hit_hi, self.qd * -0.1, self.qd)
            self.q = torch.clamp(q_next, self.j_min, self.j_max)

        self.step_count += 1
        self.gait_phase = (self.gait_phase + dt_ctrl * 6.8) % (2.0 * math.pi)
        self._update_foot_metrics()

        # Compute rewards & terminations
        pitch = self.eul[:, 0]
        yaw_err = _wrap_angle(self.eul[:, 1] - math.pi / 2.0)
        roll = self.eul[:, 2]
        upright = torch.exp(-4.5 * (pitch * pitch + roll * roll))
        height_ok = torch.exp(-6.0 * (self.pos[:, 1] - 2.64) ** 2)
        ang_calm = torch.exp(-0.12 * (self.ang_vel ** 2).sum(dim=-1))

        energy_pen = 0.015 * (action ** 2).mean(dim=-1) + 0.03 * ((action - self.prev_act) ** 2).mean(dim=-1)
        self.prev_act = action.clone()

        both_feet = (total_contact[:, 0] > 1.0).float() * (total_contact[:, 1] > 1.0).float()
        any_foot = ((total_contact[:, 0] > 1.0) | (total_contact[:, 1] > 1.0)).float()

        if self.stage == 0:
            # Stage 0: Learn to stand balanced on both feet
            still_bonus = torch.exp(-0.8 * (self.vel[:, 0] ** 2 + self.vel[:, 2] ** 2))
            reward = (
                0.45
                + 1.3 * upright
                + 1.1 * height_ok
                + 0.4 * both_feet
                + 0.3 * still_bonus
                + 0.25 * ang_calm
                - energy_pen
            )
        else:
            # Stage 1: Walk forward along +x road
            vx = self.vel[:, 0]
            vz = self.vel[:, 2]
            vel_match = torch.exp(-1.1 * (vx - self.target_speed) ** 2)
            fwd_progress = torch.clamp(vx, -1.0, 3.6) * 0.65
            heading = torch.exp(-3.0 * yaw_err * yaw_err - 0.8 * vz * vz)

            # Encourage alternating stride & swing foot clearance when moving
            s_phase = torch.sin(self.gait_phase)
            stride_sep = torch.clamp(torch.abs(self.q[:, 9] - self.q[:, 11]), 0.0, 0.9)
            swing_lift = torch.clamp(torch.abs(self.foot_heights[:, 0] - self.foot_heights[:, 1]), 0.0, 0.35)
            gait_guide = 0.4 * stride_sep + 0.8 * swing_lift + 0.15 * s_phase * (self.q[:, 9] - self.q[:, 11])

            reward = (
                0.35
                + 1.5 * vel_match
                + fwd_progress
                + 0.95 * upright
                + 0.75 * height_ok
                + 0.45 * heading
                + 0.25 * any_foot
                + gait_guide
                + 0.15 * ang_calm
                - energy_pen
            )

        fallen = (
            (self.pos[:, 1] < 1.65)
            | (torch.abs(pitch) > 0.95)
            | (torch.abs(roll) > 0.95)
            | (torch.abs(self.pos[:, 2]) > 3.7)
            | (torch.abs(self.pos[:, 0]) > 318.0)
        )
        timeout = self.step_count >= MAX_EPISODE_STEPS
        done = fallen | timeout

        self.ep_reward += reward
        self.ep_upright += upright

        completed_info = []
        done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            for idx in done_ids.tolist():
                steps = max(1, int(self.step_count[idx].item()))
                completed_info.append(
                    {
                        "env": idx,
                        "reward": float(self.ep_reward[idx].item()),
                        "steps": steps,
                        "upright": float(self.ep_upright[idx].item() / steps),
                        "distance": float((self.pos[idx, 0] - self.start_x[idx]).item()),
                        "fallen": bool(fallen[idx].item()),
                    }
                )
            self.reset(done_ids)

        obs = self.get_obs()
        return obs, reward, done, completed_info

    def get_obs(self) -> torch.Tensor:
        """Build [N, 45] policy observation vector on GPU."""
        q_norm = (self.q - self.j_mid) / (self.j_half + 1e-6)
        qd_norm = self.qd / 10.0
        h_norm = (self.pos[:, 1:2] - 2.65) / 1.0
        vel_norm = self.vel / 5.0
        yaw_err = _wrap_angle(self.eul[:, 1] - math.pi / 2.0)
        eul_obs = torch.stack([self.eul[:, 0], yaw_err, self.eul[:, 2]], dim=-1)
        ang_norm = self.ang_vel / 8.0
        stage_feat = torch.full((self.num_envs, 1), float(self.stage), device=self.device)
        gait_sin = torch.sin(self.gait_phase).unsqueeze(-1) * float(self.stage)
        gait_cos = torch.cos(self.gait_phase).unsqueeze(-1) * float(self.stage)

        return torch.cat(
            [
                h_norm,          # 1
                vel_norm,        # 3
                eul_obs,         # 3
                ang_norm,        # 3
                q_norm,          # 15
                qd_norm,         # 15
                self.foot_contact,  # 2
                stage_feat,      # 1
                gait_sin,        # 1
                gait_cos,        # 1
            ],
            dim=-1,
        )

    def get_hero_snapshot(self) -> dict:
        """Extract environment 0 state for smooth 3D rendering in Three.js."""
        return {
            "pos": [round(float(v), 4) for v in self.pos[0].tolist()],
            "eul": [round(float(v), 4) for v in self.eul[0].tolist()],
            "joints": [round(float(v), 4) for v in self.q[0].tolist()],
            "contacts": [int(v) for v in self.foot_contact[0].tolist()],
            "speed": round(float(self.vel[0, 0].item()), 3),
            "step": int(self.step_count[0].item()),
        }
