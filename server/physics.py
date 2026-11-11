"""Vectorized 3D articulated biped rigid-body physics on PyTorch CUDA/CPU."""

from __future__ import annotations

import math
import torch

JOINT_NAMES = [
    "hy", "hx", "wt",
    "rs", "rr", "re",
    "ls", "lr", "le",
    "rh", "rha", "rk",
    "lh", "lha", "lk",
    "ra", "la",
]

# [min, max, sign] matching src/robot/model.js (17 DOFs including lateral hip abduction/adduction)
JOINT_MIN = torch.tensor(
    [-1.0, -0.6, -0.8, -2.5, 0.0, 0.0, -2.5, 0.0, 0.0, -1.2, -0.45, 0.0, -1.2, -0.45, 0.0, -0.8, -0.8],
    dtype=torch.float32,
)
JOINT_MAX = torch.tensor(
    [1.0, 0.6, 0.8, 2.5, 2.8, 2.4, 2.5, 2.8, 2.4, 1.2, 0.85, 2.2, 1.2, 0.85, 2.2, 0.8, 0.8],
    dtype=torch.float32,
)
JOINT_SIGN = torch.tensor(
    [1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
    dtype=torch.float32,
)
# Neutral human-like athletic stance (soft knees, flexed ankles, natural hip stance width, bent elbows)
JOINT_NOM = torch.tensor(
    [0.0, 0.0, 0.0, 0.0, 0.08, 0.36, 0.0, 0.08, 0.36, 0.12, 0.04, 0.24, 0.12, 0.04, 0.24, -0.12, -0.12],
    dtype=torch.float32,
)
ACTION_SCALE = torch.tensor(
    [0.20, 0.20, 0.25, 1.05, 0.90, 0.85, 1.05, 0.90, 0.85, 0.85, 0.55, 0.95, 0.85, 0.55, 0.95, 0.62, 0.62],
    dtype=torch.float32,
)

# Compliant human-like PD joint stiffness (front/back + left/right hips, flexible knees/ankles/elbows)
KP = torch.tensor(
    [12.0, 12.0, 20.0, 22.0, 22.0, 18.0, 22.0, 22.0, 18.0, 46.0, 44.0, 48.0, 46.0, 44.0, 48.0, 34.0, 34.0],
    dtype=torch.float32,
)
KD = torch.tensor(
    [1.8, 1.8, 2.8, 2.6, 2.6, 2.2, 2.6, 2.6, 2.2, 4.8, 4.6, 5.0, 4.8, 4.6, 5.0, 3.8, 3.8],
    dtype=torch.float32,
)
TORQUE_LIMIT = torch.tensor(
    [4.0, 4.0, 8.0, 11.0, 11.0, 8.0, 11.0, 11.0, 8.0, 30.0, 26.0, 32.0, 30.0, 26.0, 32.0, 24.0, 24.0],
    dtype=torch.float32,
)
JOINT_INERTIA = torch.tensor(
    [0.06, 0.06, 0.14, 0.09, 0.09, 0.07, 0.09, 0.09, 0.07, 0.20, 0.18, 0.18, 0.20, 0.18, 0.18, 0.12, 0.12],
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


def _rot_z(az: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate 3D vectors v [..., 3] around local Z by angles az [...] (broadcastable)."""
    cz = torch.cos(az)
    sz = torch.sin(az)
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    return torch.stack([vx * cz - vy * sz, vx * sz + vy * cz, vz], dim=-1)


def _wrap_angle(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class BipedPhysicsEnv:
    """Batched 3D articulated biped physics simulator running on PyTorch CUDA/CPU."""

    OBS_DIM = 49
    ACT_DIM = 17

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
        self.hip_local_buf = torch.tensor(
            [[0.3, -0.15, 0.0], [-0.3, -0.15, 0.0]], device=self.device
        ).unsqueeze(0).expand(num_envs, 2, 3).contiguous()
        self.chest_local = torch.tensor([[0.0, 1.35, 0.0]], device=self.device)
        self.head_local = torch.tensor([[0.0, 2.35, 0.0]], device=self.device)
        self.arm_rest_vec = torch.tensor([[0.0, -2.2, 0.0]], device=self.device)

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
        self.stage_t = torch.zeros(1, device=self.device)
        self.target_speed = 1.65

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

        # GPU-resident accumulators & hero snapshot vector (eliminates per-step CPU syncs)
        self.ep_count_acc = torch.zeros(1, device=self.device)
        self.ep_rew_acc = torch.zeros(1, device=self.device)
        self.ep_up_acc = torch.zeros(1, device=self.device)
        self.ep_len_acc = torch.zeros(1, device=self.device)
        self.ep_dist_acc = torch.zeros(1, device=self.device)
        self.best_rew_t = torch.full((1,), -1e9, device=self.device)
        self.best_dist_t = torch.zeros(1, device=self.device)
        self.hero_vec = torch.zeros(27, device=self.device)

        self.crouch_delta = torch.zeros(self.ACT_DIM, device=self.device)
        self.crouch_delta[9] = 0.36   # rh forward flex
        self.crouch_delta[11] = 0.68  # rk knee bend
        self.crouch_delta[12] = 0.36  # lh forward flex
        self.crouch_delta[14] = 0.68  # lk knee bend
        self.crouch_delta[15] = -0.32 # ra ankle dorsiflexion
        self.crouch_delta[16] = -0.32 # la ankle dorsiflexion
        self.crouch_delta[3] = -0.18  # rs arms slightly back
        self.crouch_delta[6] = -0.18  # ls arms slightly back
        self.crouch_delta[4] = 0.20   # rr arms slightly out
        self.crouch_delta[7] = 0.20   # lr arms slightly out

        # Precompute exact nominal standing height so feet rest cleanly on road on branchless resets
        self.pos[:, 1] = 2.70
        self.eul[:, 1] = math.pi / 2.0
        self.q.copy_(self.j_nom.unsqueeze(0).expand(n, -1))
        _, _, _, _, _, _, _, p_corners, _, _ = self._compute_fk()
        min_foot_y = float(p_corners[0, ..., 1].min().item())
        self.nom_spawn_y = 2.70 + (ROAD_TOP_Y + 0.004) - min_foot_y

        self.reset()

    def set_stage(self, stage: int) -> None:
        self.stage = 1 if stage >= 1 else 0
        self.stage_t.fill_(float(self.stage))

    def apply_impulse(self, dvx: float = 0.0, dvz: float = 0.0, env_idx: int = 0) -> None:
        """Apply a physical external push impulse matching right-hand chest torque axes."""
        self.vel[env_idx, 0] += dvx
        self.vel[env_idx, 2] += dvz
        self.ang_vel[env_idx, 2] -= dvx * 0.22
        self.ang_vel[env_idx, 0] += dvz * 0.22

    def _reset_masked(self, done: torch.Tensor) -> None:
        """Branchless pure-GPU masked reset: Stand starts crouched/tilted so rising & balance are visibly learned."""
        n = self.num_envs
        m1 = done.unsqueeze(-1)
        stand_w = 1.0 - self.stage_t[0]
        crouch_frac = stand_w * (0.35 + 0.45 * torch.rand(n, device=self.device))

        rand_pos = torch.stack(
            [
                torch.zeros(n, device=self.device),
                self.nom_spawn_y - 0.13 * crouch_frac + (torch.rand(n, device=self.device) - 0.5) * 0.015,
                (torch.rand(n, device=self.device) - 0.5) * 0.30,
            ],
            dim=-1,
        )
        rand_vel = torch.stack(
            [
                (torch.rand(n, device=self.device) - 0.5) * (0.20 + 0.35 * stand_w),
                torch.zeros(n, device=self.device),
                (torch.rand(n, device=self.device) - 0.5) * (0.20 + 0.35 * stand_w),
            ],
            dim=-1,
        )
        rand_eul = torch.stack(
            [
                (torch.rand(n, device=self.device) - 0.5) * (0.10 + 0.22 * stand_w) + 0.08 * crouch_frac,
                math.pi / 2.0 + (torch.rand(n, device=self.device) - 0.5) * 0.10,
                (torch.rand(n, device=self.device) - 0.5) * (0.10 + 0.18 * stand_w),
            ],
            dim=-1,
        )
        rand_ang = (torch.rand(n, 3, device=self.device) - 0.5) * (0.20 + 0.30 * stand_w)
        q0 = torch.clamp(
            self.j_nom.unsqueeze(0)
            + crouch_frac.unsqueeze(-1) * self.crouch_delta.unsqueeze(0)
            + (torch.rand(n, self.ACT_DIM, device=self.device) - 0.5) * 0.08,
            self.j_min,
            self.j_max,
        )

        self.pos.copy_(torch.where(m1, rand_pos, self.pos))
        self.vel.copy_(torch.where(m1, rand_vel, self.vel))
        self.eul.copy_(torch.where(m1, rand_eul, self.eul))
        self.ang_vel.copy_(torch.where(m1, rand_ang, self.ang_vel))
        self.q.copy_(torch.where(m1, q0, self.q))
        self.q_cmd.copy_(torch.where(m1, q0, self.q_cmd))
        self.qd.copy_(torch.where(m1, torch.zeros_like(self.qd), self.qd))
        self.prev_act.copy_(torch.where(m1, torch.zeros_like(self.prev_act), self.prev_act))
        self.step_count.copy_(torch.where(done, torch.zeros_like(self.step_count), self.step_count))
        self.gait_phase.copy_(
            torch.where(done, torch.rand(n, device=self.device) * (2.0 * math.pi), self.gait_phase)
        )
        self.ep_reward.copy_(torch.where(done, torch.zeros_like(self.ep_reward), self.ep_reward))
        self.ep_upright.copy_(torch.where(done, torch.zeros_like(self.ep_upright), self.ep_upright))
        self.start_x.copy_(torch.where(done, rand_pos[:, 0], self.start_x))

    def reset(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            done = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            if env_ids.numel() > 0:
                done[env_ids] = True
        self._reset_masked(done)
        self._update_foot_metrics()
        self._write_hero_vec()
        return self.get_obs()

    def _compute_fk(self):
        """Exact 3D forward kinematics for 17-DOF legs (front/back + left/right hips), feet, and arms."""
        R_root = _rot_yxz(self.eul[:, 0], self.eul[:, 1], self.eul[:, 2])  # [N, 3, 3]
        axis_z_local = R_root[:, :, 2]  # [N, 3] hip abduction/adduction axis

        # Build both legs simultaneously [N, 2, ...] (0=Right s=+1, 1=Left s=-1)
        p_hip = self.pos.unsqueeze(1) + torch.einsum("nij,nlj->nli", R_root, self.hip_local_buf)

        # Signed sagittal + lateral hip angles (matching model.js ZYX hip rotation order)
        a_hip = torch.stack([self.q[:, 9] * -1.0, self.q[:, 12] * -1.0], dim=1)
        az_hip = torch.stack([self.q[:, 10], self.q[:, 13] * -1.0], dim=1)
        a_knee = a_hip + torch.stack([self.q[:, 11], self.q[:, 14]], dim=1)
        a_foot = a_knee + torch.stack([self.q[:, 15], self.q[:, 16]], dim=1)
        axis_x_leg = (
            torch.cos(az_hip).unsqueeze(-1) * R_root[:, :, 0].unsqueeze(1)
            + torch.sin(az_hip).unsqueeze(-1) * R_root[:, :, 1].unsqueeze(1)
        )

        # Thigh vector in root frame: Rz(az_hip) @ Rx(a_hip) @ [0, -1.2, 0]
        thigh_sag = torch.stack(
            [torch.zeros_like(a_hip), -1.2 * torch.cos(a_hip), -1.2 * torch.sin(a_hip)], dim=-1
        )
        p_knee = p_hip + torch.einsum("nij,nlj->nli", R_root, _rot_z(az_hip, thigh_sag))

        # Shin vector in root frame: Rz(az_hip) @ Rx(a_knee) @ [0, -1.15, 0]
        shin_sag = torch.stack(
            [torch.zeros_like(a_knee), -1.15 * torch.cos(a_knee), -1.15 * torch.sin(a_knee)], dim=-1
        )
        p_ankle = p_knee + torch.einsum("nij,nlj->nli", R_root, _rot_z(az_hip, shin_sag))

        # Foot corners [N, 2, 4, 3]: Rz(az_hip) @ Rx(a_foot) @ foot_corners
        ca = torch.cos(a_foot).unsqueeze(-1)
        sa = torch.sin(a_foot).unsqueeze(-1)
        fc = self.foot_corners.view(1, 1, 4, 3)
        fy = fc[..., 1] * ca - fc[..., 2] * sa
        fz = fc[..., 1] * sa + fc[..., 2] * ca
        fx = fc[..., 0].expand_as(fy)
        corners_local = _rot_z(az_hip.unsqueeze(-1), torch.stack([fx, fy, fz], dim=-1))
        p_corners = p_ankle.unsqueeze(2) + torch.einsum("nij,nlcj->nlci", R_root, corners_local)

        # Body collision points in world (chest, head)
        p_chest = self.pos + torch.einsum("nij,nj->ni", R_root, self.chest_local)
        p_head = self.pos + torch.einsum("nij,nj->ni", R_root, self.head_local)

        # Include arm mass COM shift so raising/swinging arms physically shifts balance
        rs, rr = self.q[:, 3], self.q[:, 4]
        ls, lr = self.q[:, 6], self.q[:, 7]
        ay_r = -1.1 * torch.cos(rr)
        ay_l = -1.1 * torch.cos(lr)
        arm_r_off = torch.stack([1.1 * torch.sin(rr), ay_r * torch.cos(rs), -ay_r * torch.sin(rs)], dim=-1)
        arm_l_off = torch.stack([-1.1 * torch.sin(lr), ay_l * torch.cos(ls), -ay_l * torch.sin(ls)], dim=-1)
        arm_com_shift = 0.10 * ((arm_r_off + arm_l_off) - self.arm_rest_vec)
        p_com = self.pos + torch.einsum("nij,nj->ni", R_root, self.com_local.unsqueeze(0) + arm_com_shift)

        return R_root, axis_x_leg, axis_z_local, p_com, p_hip, p_knee, p_ankle, p_corners, p_chest, p_head

    def _update_foot_metrics(self):
        _, _, _, _, _, _, _, p_corners, _, _ = self._compute_fk()
        min_y = p_corners[..., 1].min(dim=-1).values  # [N, 2]
        self.foot_heights.copy_((min_y - ROAD_TOP_Y).clamp(min=0.0))
        self.foot_contact.copy_((min_y <= ROAD_TOP_Y + 0.025).float())

    def step(self, action: torch.Tensor, dt_ctrl: float = 1.0 / 45.0, substeps: int = 4):
        """Compatibility step returning (obs, reward, done, info_list)."""
        obs, rew, done = self._step_core(action, dt_ctrl, substeps)
        return obs, rew, done, []

    def _step_core(self, action: torch.Tensor, dt_ctrl: float = 1.0 / 45.0, substeps: int = 4):
        """Advance physics purely in-place in GPU tensors without CPU synchronization."""
        action = torch.clamp(action, -1.0, 1.0)
        q_target_raw = torch.clamp(self.j_nom + action * self.act_scale, self.j_min, self.j_max)
        self.q_cmd.add_(0.45 * (q_target_raw - self.q_cmd))
        q_target = self.q_cmd
        dt = dt_ctrl / substeps

        total_contact = torch.zeros(self.num_envs, 2, device=self.device)
        k_n = 650.0
        c_n = 48.0
        k_f = 130.0
        mu = 1.15

        for _ in range(substeps):
            R_root, axis_x_leg, axis_z, p_com, p_hip, p_knee, p_ankle, p_corners, p_chest, p_head = self._compute_fk()

            # Joint PD servo torques
            tau_pd = torch.clamp(
                self.kp * (q_target - self.q) - self.kd * self.qd,
                -self.tau_lim,
                self.tau_lim,
            )

            # Foot corner velocities including front/back + left/right hip velocities
            qd_rh, qd_rk, qd_ra = self.qd[:, 9] * -1.0, self.qd[:, 11], self.qd[:, 15]
            qd_lh, qd_lk, qd_la = self.qd[:, 12] * -1.0, self.qd[:, 14], self.qd[:, 16]
            qd_rha, qd_lha = self.qd[:, 10], self.qd[:, 13] * -1.0
            w_hip_x = torch.stack([qd_rh, qd_lh], dim=1).unsqueeze(-1) * axis_x_leg
            w_hip_z = torch.stack([qd_rha, qd_lha], dim=1).unsqueeze(-1) * axis_z.unsqueeze(1)
            w_knee = torch.stack([qd_rk, qd_lk], dim=1).unsqueeze(-1) * axis_x_leg
            w_ank = torch.stack([qd_ra, qd_la], dim=1).unsqueeze(-1) * axis_x_leg

            r_com_c = p_corners - p_com.unsqueeze(1).unsqueeze(2)
            v_root_c = self.vel.unsqueeze(1).unsqueeze(2) + torch.cross(
                self.ang_vel.unsqueeze(1).unsqueeze(2).expand_as(r_com_c), r_com_c, dim=-1
            )
            v_joint_c = (
                torch.cross((w_hip_x + w_hip_z).unsqueeze(2).expand_as(r_com_c), p_corners - p_hip.unsqueeze(2), dim=-1)
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

            # Action-reaction joint contact torques for legs (sagittal + lateral hip abduction)
            tau_c_sum = torch.cross(p_corners - p_hip.unsqueeze(2), F_c, dim=-1).sum(dim=2)
            tau_c_hip = (tau_c_sum * axis_x_leg).sum(dim=-1)  # [N, 2]
            tau_c_hip_z = (tau_c_sum * axis_z.unsqueeze(1)).sum(dim=-1)  # [N, 2]
            tau_c_knee = (
                torch.cross(p_corners - p_knee.unsqueeze(2), F_c, dim=-1).sum(dim=2) * axis_x_leg
            ).sum(dim=-1)
            tau_c_ank = (
                torch.cross(p_corners - p_ankle.unsqueeze(2), F_c, dim=-1).sum(dim=2) * axis_x_leg
            ).sum(dim=-1)

            # Convert contact reaction into 17-joint coordinate space
            tau_ext = torch.zeros_like(tau_pd)
            tau_ext[:, 9] = tau_c_hip[:, 0] * -1.0
            tau_ext[:, 10] = tau_c_hip_z[:, 0]
            tau_ext[:, 11] = tau_c_knee[:, 0]
            tau_ext[:, 12] = tau_c_hip[:, 1] * -1.0
            tau_ext[:, 13] = tau_c_hip_z[:, 1] * -1.0
            tau_ext[:, 14] = tau_c_knee[:, 1]
            tau_ext[:, 15] = tau_c_ank[:, 0]
            tau_ext[:, 16] = tau_c_ank[:, 1]

            # Human arm & waist angular-momentum counterbalance reaction on torso (proportional to arm inertia)
            arm_pitch_react = 0.11 * (tau_pd[:, 3] + tau_pd[:, 6])
            arm_roll_react = -0.16 * tau_pd[:, 4] + 0.16 * tau_pd[:, 7]
            waist_yaw_react = -0.08 * tau_pd[:, 2]
            M_local = torch.einsum("nji,nj->ni", R_root, M_net)
            M_local[:, 0] += arm_pitch_react
            M_local[:, 1] += waist_yaw_react
            M_local[:, 2] += arm_roll_react

            # Integrate root linear & angular velocity strictly in-place (CUDA Graph safe!)
            acc = F_net / MASS
            self.vel.copy_(torch.clamp(self.vel + acc * dt, -18.0, 18.0))
            self.pos.add_(self.vel * dt)
            self.pos[:, 0].clamp_(-325.0, 325.0)
            self.pos[:, 1].clamp_(0.35, 8.0)

            ang_acc_local = M_local / self.body_inertia - 0.8 * torch.einsum("nji,nj->ni", R_root, self.ang_vel)
            ang_acc_world = torch.einsum("nij,nj->ni", R_root, ang_acc_local)
            self.ang_vel.copy_(torch.clamp(self.ang_vel + ang_acc_world * dt, -14.0, 14.0))

            # Map world angular velocity to YXZ Euler rates in-place
            w_loc = torch.einsum("nji,nj->ni", R_root, self.ang_vel)
            cp = torch.cos(self.eul[:, 0]).clamp(min=0.15)
            sp = torch.sin(self.eul[:, 0])
            cr = torch.cos(self.eul[:, 2])
            sr = torch.sin(self.eul[:, 2])
            pitch_dot = w_loc[:, 0] * cr - w_loc[:, 1] * sr
            yaw_dot = (w_loc[:, 1] * cr + w_loc[:, 0] * sr) / cp
            roll_dot = w_loc[:, 2] + yaw_dot * sp
            self.eul[:, 0].copy_(torch.clamp(self.eul[:, 0] + pitch_dot * dt, -1.5, 1.5))
            self.eul[:, 1].copy_(_wrap_angle(self.eul[:, 1] + yaw_dot * dt))
            self.eul[:, 2].copy_(torch.clamp(self.eul[:, 2] + roll_dot * dt, -1.5, 1.5))

            # Integrate joints strictly in-place with physical limits
            qdd = (tau_pd + 0.18 * tau_ext - 0.35 * self.qd) / self.j_inertia
            qd_next = torch.clamp(self.qd + qdd * dt, -16.0, 16.0)
            q_next = self.q + qd_next * dt
            hit_lo = q_next < self.j_min
            hit_hi = q_next > self.j_max
            self.qd.copy_(torch.where(hit_lo | hit_hi, qd_next * -0.1, qd_next))
            self.q.copy_(torch.clamp(q_next, self.j_min, self.j_max))

        self.step_count += 1
        self.gait_phase.copy_((self.gait_phase + dt_ctrl * 5.0) % (2.0 * math.pi))

        # Periodic directional pushes in Stand training (Stage 0) only after standing upright & calm
        n = self.num_envs
        stand_w = 1.0 - self.stage_t[0]
        can_push = (
            (self.step_count > 60)
            & (torch.abs(self.eul[:, 0]) < 0.14)
            & (torch.abs(self.eul[:, 2]) < 0.14)
            & (torch.abs(self.vel[:, 0]) < 0.25)
            & (torch.abs(self.vel[:, 2]) < 0.25)
        )
        trigger = (can_push & (torch.rand(n, device=self.device) < 0.0040)).float() * stand_w
        is_sag = (torch.rand(n, device=self.device) < 0.5).float()
        p_sign = torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0)
        sag_p = 0.70 + 0.45 * torch.rand(n, device=self.device)
        lat_p = 0.58 + 0.34 * torch.rand(n, device=self.device)
        dvx_p = trigger * is_sag * p_sign * sag_p
        dvz_p = trigger * (1.0 - is_sag) * p_sign * lat_p
        self.vel[:, 0].add_(dvx_p)
        self.vel[:, 2].add_(dvz_p)
        self.ang_vel[:, 2].sub_(dvx_p * 0.22)
        self.ang_vel[:, 0].add_(dvz_p * 0.22)

        self._update_foot_metrics()

        # Compute human-like rewards & terminations
        pitch = self.eul[:, 0]
        yaw_err = _wrap_angle(self.eul[:, 1] - math.pi / 2.0)
        roll = self.eul[:, 2]
        vx = self.vel[:, 0]
        vz = self.vel[:, 2]
        settled = (self.step_count > 55).float()
        sag_sig = vx + 1.4 * pitch * settled
        lat_sig = vz + 1.5 * roll * settled
        sag_mag = torch.clamp((torch.abs(sag_sig) - 0.18) / 0.75, 0.0, 1.0)
        lat_mag = torch.clamp((torch.abs(lat_sig) - 0.22) / 0.70, 0.0, 1.0)
        calm_w = 1.0 - torch.maximum(sag_mag, lat_mag)

        # Straight upright torso posture (allows knee/hip dip when absorbing a directional push)
        upright = torch.exp(-6.5 * (pitch * pitch + roll * roll))
        torso_straight_pen = 0.65 * (pitch * pitch + roll * roll) + 0.25 * (self.q[:, 0] ** 2 + self.q[:, 1] ** 2)
        target_h = 2.68 - 0.14 * (1.0 - calm_w)
        height_ok = torch.exp(-7.0 * (self.pos[:, 1] - target_h) ** 2)
        ang_calm = torch.exp(-0.15 * (self.ang_vel ** 2).sum(dim=-1))

        energy_pen = 0.010 * (action ** 2).mean(dim=-1) + 0.020 * ((action - self.prev_act) ** 2).mean(dim=-1)
        self.prev_act.copy_(action)

        both_feet = (total_contact[:, 0] > 1.0).float() * (total_contact[:, 1] > 1.0).float()
        any_foot = ((total_contact[:, 0] > 1.0) | (total_contact[:, 1] > 1.0)).float()

        # Arm counterbalance reward: extend arms laterally to counter roll, relax down when upright
        desired_rr = torch.clamp(0.08 + torch.clamp(roll, min=0.0) * 2.4, 0.04, 1.5)
        desired_lr = torch.clamp(0.08 + torch.clamp(-roll, min=0.0) * 2.4, 0.04, 1.5)
        arm_balance_pose = torch.exp(
            -3.0 * ((self.q[:, 4] - desired_rr) ** 2 + (self.q[:, 7] - desired_lr) ** 2)
        )
        elbow_natural = torch.exp(-2.5 * ((self.q[:, 5] - 0.42) ** 2 + (self.q[:, 8] - 0.42) ** 2))
        # Directional human push-recovery targets in Stand (front/back knee-step + left/right hip abduct/adduct)
        push_back = (sag_sig < -0.18).float()
        push_fwd = (sag_sig > 0.18).float()
        push_from_right = (lat_sig > 0.22).float()
        push_from_left = (lat_sig < -0.22).float()
        t_rh = 0.12 + 0.38 * push_fwd + 0.06 * push_back
        t_lh = 0.12 - 0.56 * push_back - 0.28 * push_fwd
        t_rk = 0.24 + 0.40 * push_fwd + 0.06 * push_back + 0.22 * push_from_left
        t_lk = 0.24 + 0.46 * push_back + 0.10 * push_fwd + 0.22 * push_from_right
        sag_recover_pose = torch.exp(
            -2.8 * ((self.q[:, 9] - t_rh) ** 2 + (self.q[:, 12] - t_lh) ** 2 + (self.q[:, 11] - t_rk) ** 2 + (self.q[:, 14] - t_lk) ** 2)
        )

        t_rha = 0.04 + 0.44 * push_from_left - 0.18 * push_from_right
        t_lha = 0.04 + 0.44 * push_from_right - 0.18 * push_from_left
        lat_recover_pose = torch.exp(-3.5 * ((self.q[:, 10] - t_rha) ** 2 + (self.q[:, 13] - t_lha) ** 2))

        ankle_flex_stand = torch.exp(-3.5 * ((self.q[:, 15] + 0.12) ** 2 + (self.q[:, 16] + 0.12) ** 2))
        still_bonus = torch.exp(-0.8 * (vx * vx + vz * vz))
        waist_straight = torch.exp(-8.0 * (self.q[:, 2] ** 2))
        arm_pitch_calm = torch.exp(-2.0 * ((self.q[:, 3] + 1.5 * pitch) ** 2 + (self.q[:, 6] + 1.5 * pitch) ** 2))
        rew_stand = (
            0.45
            + 1.5 * upright
            + 1.15 * height_ok
            + 0.40 * calm_w * both_feet
            + 0.35 * calm_w * still_bonus
            + 0.35 * waist_straight
            + 0.45 * arm_balance_pose
            + 0.35 * arm_pitch_calm
            + 0.25 * elbow_natural
            + (0.40 + 0.55 * sag_mag) * sag_recover_pose
            + (0.40 + 0.55 * lat_mag) * lat_recover_pose
            + 0.20 * ankle_flex_stand
            + 0.20 * ang_calm
            - torso_straight_pen
            - energy_pen
        )

        desired_rha = t_rha
        desired_lha = t_lha
        hip_side_pose = lat_recover_pose
        moving_gate = torch.clamp(vx / 0.75, 0.0, 1.0)
        vel_match = 3.2 * torch.exp(-1.1 * (vx - self.target_speed) ** 2)
        fwd_progress = 2.6 * torch.clamp(vx, -1.2, 2.5)
        stand_still_pen = 0.85 * torch.exp(-3.5 * vx * vx)
        heading = torch.exp(-3.5 * yaw_err * yaw_err - 0.9 * vz * vz)
        arm_counter_err = (self.q[:, 3] + 0.90 * self.q[:, 9]) ** 2 + (self.q[:, 6] + 0.90 * self.q[:, 12]) ** 2
        arm_swing_reward = 0.95 * torch.exp(-3.5 * arm_counter_err)
        waist_natural = 0.25 * torch.exp(-6.0 * (self.q[:, 2] - 0.14 * (self.q[:, 12] - self.q[:, 9])) ** 2)
        s_phase = torch.sin(self.gait_phase)
        c_phase = torch.cos(self.gait_phase)
        # Dynamic knee flexion on swing leg + ankle compensation (matching control.md gait!)
        target_rk = 0.20 + torch.clamp(c_phase, min=0.0) * 0.65
        target_lk = 0.20 + torch.clamp(-c_phase, min=0.0) * 0.65
        knee_flex_walk = 0.55 * torch.exp(-3.0 * ((self.q[:, 11] - target_rk) ** 2 + (self.q[:, 14] - target_lk) ** 2))
        ankle_flex_walk = 0.35 * torch.exp(
            -3.0 * ((self.q[:, 15] - 0.6 * (self.q[:, 9] - self.q[:, 11])) ** 2 + (self.q[:, 16] - 0.6 * (self.q[:, 12] - self.q[:, 14])) ** 2)
        )
        stride_sep = torch.clamp(torch.abs(self.q[:, 9] - self.q[:, 12]), 0.0, 0.9)
        swing_lift = torch.clamp(torch.abs(self.foot_heights[:, 0] - self.foot_heights[:, 1]), 0.0, 0.35)
        gait_guide = 0.50 * stride_sep + 0.95 * swing_lift + 0.35 * s_phase * (self.q[:, 9] - self.q[:, 12])
        posture_base = 1.05 * upright + 0.70 * height_ok + 0.45 * heading + 0.25 * arm_balance_pose + 0.20 * elbow_natural + 0.20 * hip_side_pose
        gait_bonus = moving_gate * (
            arm_swing_reward + knee_flex_walk + ankle_flex_walk + gait_guide + waist_natural
        )
        rew_walk = (
            0.10
            + vel_match
            + fwd_progress
            + (0.30 + 0.70 * moving_gate) * posture_base
            + 0.20 * any_foot
            + gait_bonus
            + 0.10 * ang_calm
            - stand_still_pen
            - 0.65 * torso_straight_pen
            - energy_pen
        )

        stage_w = self.stage_t[0]
        reward = (1.0 - stage_w) * rew_stand + stage_w * rew_walk

        fallen = (
            (self.pos[:, 1] < 1.50)
            | (torch.abs(pitch) > 0.95)
            | (torch.abs(roll) > 0.95)
            | (torch.abs(self.pos[:, 2]) > 3.7)
            | (torch.abs(self.pos[:, 0]) > 318.0)
        )
        timeout = self.step_count >= MAX_EPISODE_STEPS
        done = fallen | timeout

        self.ep_reward += reward
        self.ep_upright += upright

        # Accumulate episode summaries directly on GPU with zero CPU-GPU syncs
        d_f = done.float()
        steps_f = self.step_count.float().clamp(min=1.0)
        dist = self.pos[:, 0] - self.start_x
        self.ep_count_acc += d_f.sum()
        self.ep_rew_acc += (self.ep_reward * d_f).sum()
        self.ep_up_acc += ((self.ep_upright / steps_f) * d_f).sum()
        self.ep_len_acc += (steps_f * d_f).sum()
        self.ep_dist_acc += (dist * d_f).sum()

        neg_inf = torch.full_like(self.ep_reward, -1e9)
        max_rew = torch.where(done, self.ep_reward, neg_inf).max()
        self.best_rew_t.copy_(torch.maximum(self.best_rew_t, max_rew))
        walk_dist = torch.where(done & (stage_w > 0.5), dist, torch.zeros_like(dist)).max()
        self.best_dist_t.copy_(torch.maximum(self.best_dist_t, walk_dist))

        self._reset_masked(done)
        self._update_foot_metrics()
        self._write_hero_vec()
        obs = self.get_obs()
        return obs, reward, done

    def _write_hero_vec(self) -> None:
        self.hero_vec[:3].copy_(self.pos[0])
        self.hero_vec[3:6].copy_(self.eul[0])
        self.hero_vec[6:23].copy_(self.q[0])
        self.hero_vec[23:25].copy_(self.foot_contact[0])
        self.hero_vec[25] = self.vel[0, 0]
        self.hero_vec[26] = self.step_count[0].float()

    def step_gpu(self, action: torch.Tensor, dt_ctrl: float = 1.0 / 45.0, substeps: int = 4):
        """Pure-GPU step without CPU synchronization (CUDA-Graph capturable)."""
        return self._step_core(action, dt_ctrl, substeps)

    def consume_gpu_stats(self) -> dict | None:
        """Read and reset accumulated GPU episode metrics once per rollout window."""
        vec = torch.stack(
            [
                self.ep_count_acc[0],
                self.ep_rew_acc[0],
                self.ep_up_acc[0],
                self.ep_len_acc[0],
                self.ep_dist_acc[0],
                self.best_rew_t[0],
                self.best_dist_t[0],
            ]
        ).tolist()
        self.ep_count_acc.zero_()
        self.ep_rew_acc.zero_()
        self.ep_up_acc.zero_()
        self.ep_len_acc.zero_()
        self.ep_dist_acc.zero_()
        count = int(vec[0])
        if count <= 0:
            return {
                "count": 0,
                "best_reward": vec[5],
                "best_distance": vec[6],
            }
        return {
            "count": count,
            "avg_reward": vec[1] / count,
            "avg_upright": vec[2] / count,
            "avg_length": vec[3] / count,
            "avg_distance": vec[4] / count,
            "best_reward": vec[5],
            "best_distance": vec[6],
        }

    def get_obs(self) -> torch.Tensor:
        """Build [N, 45] policy observation vector on GPU."""
        q_norm = (self.q - self.j_mid) / (self.j_half + 1e-6)
        qd_norm = self.qd / 10.0
        h_norm = (self.pos[:, 1:2] - 2.65) / 1.0
        vel_norm = self.vel / 5.0
        yaw_err = _wrap_angle(self.eul[:, 1] - math.pi / 2.0)
        eul_obs = torch.stack([self.eul[:, 0], yaw_err, self.eul[:, 2]], dim=-1)
        ang_norm = self.ang_vel / 8.0
        stage_feat = self.stage_t.view(1, 1).expand(self.num_envs, 1)
        gait_sin = torch.sin(self.gait_phase).unsqueeze(-1) * stage_feat
        gait_cos = torch.cos(self.gait_phase).unsqueeze(-1) * stage_feat

        return torch.cat(
            [
                h_norm,
                vel_norm,
                eul_obs,
                ang_norm,
                q_norm,
                qd_norm,
                self.foot_contact,
                stage_feat,
                gait_sin,
                gait_cos,
            ],
            dim=-1,
        )

    def get_hero_snapshot(self) -> dict:
        """Extract environment 0 state via single batched 27-float copy."""
        v = self.hero_vec.tolist()
        return {
            "pos": [round(v[0], 4), round(v[1], 4), round(v[2], 4)],
            "eul": [round(v[3], 4), round(v[4], 4), round(v[5], 4)],
            "joints": [round(x, 4) for x in v[6:23]],
            "contacts": [int(v[23]), int(v[24])],
            "speed": round(v[25], 3),
            "step": int(v[26]),
        }
