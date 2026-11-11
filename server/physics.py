"""Multi-threaded Pure MuJoCo rigid-body humanoid environment for RL from scratch."""

from __future__ import annotations

import concurrent.futures
import math
from pathlib import Path

import mujoco
import numpy as np
import torch

from .reflex import (
    ACT_SCALE,
    JOINT_NAMES,
    J_MAX,
    J_MIN,
    J_NOM,
    mj_quat_to_three_eul_yxz,
    mj_quat_to_three_quat_xyzw,
)

MAX_EPISODE_STEPS = 480
XML_PATH = Path(__file__).resolve().parent / "sentinel.xml"


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _axis_angle_quat(ax: float, ay: float, az: float, ang: float) -> np.ndarray:
    half = 0.5 * ang
    s = math.sin(half)
    return np.array([math.cos(half), ax * s, ay * s, az * s], dtype=np.float64)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


class BipedPhysicsEnv:
    """Vectorized 3D MuJoCo biped environment stepped across C threads for pure RL from scratch."""

    OBS_DIM = 49
    ACT_DIM = 17

    def __init__(self, num_envs: int = 24, device: torch.device | None = None):
        self.num_envs = num_envs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.datas = [mujoco.MjData(self.model) for _ in range(num_envs)]
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=min(12, max(2, num_envs)))

        # Precompute joint / body / geom lookup indices once
        self.qpos_ids = np.array(
            [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES],
            dtype=np.int32,
        )
        self.qvel_ids = np.array(
            [self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES],
            dtype=np.int32,
        )
        self.hips_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hips")
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.head_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "head")
        self.rf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "r_foot")
        self.lf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "l_foot")
        self.chest_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "chest_geom")
        self.r_farm_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "r_farm_geom")
        self.l_farm_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "l_farm_geom")
        self.r_uarm_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "r_uarm_geom")
        self.l_uarm_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "l_uarm_geom")
        self.r_thigh_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "r_thigh_geom")
        self.l_thigh_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "l_thigh_geom")
        self.r_sole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "r_sole")
        self.l_sole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "l_sole")
        self.road_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "road")

        # Keep exact baseline PD gains & torque limits for Stand/Walk while allowing full-body push-up torque in Stand Up
        self._base_gainprm = self.model.actuator_gainprm.copy()
        self._base_biasprm = self.model.actuator_biasprm.copy()
        self._base_forcerange = self.model.actuator_forcerange.copy()

        self.j_mid = 0.5 * (J_MIN + J_MAX)
        self.j_half = 0.5 * (J_MAX - J_MIN)

        self.stage = 0  # 0 = Stand & Push Recovery, 1 = Walk, 2 = Stand Up from Ground
        self.target_speed = 2.95
        self.walk_cadence = 4.9
        self.training_pushes = True
        self.hero_getup_idx = 0

        # Per-env telemetry and smoothing buffers
        n = num_envs
        self.q_cmd = np.tile(J_NOM, (n, 1))
        self.prev_act = np.zeros((n, self.ACT_DIM), dtype=np.float32)
        self.step_count = np.zeros(n, dtype=np.int64)
        self.walk_phase = np.zeros(n, dtype=np.float32)
        self.foot_contact = np.zeros((n, 2), dtype=np.float32)
        self.last_single_foot = np.zeros(n, dtype=np.int8)
        self.last_step_tick = np.zeros(n, dtype=np.int64)
        self.prev_rf_x = np.zeros(n, dtype=np.float32)
        self.prev_lf_x = np.zeros(n, dtype=np.float32)
        self.ep_reward = np.zeros(n, dtype=np.float32)
        self.ep_upright = np.zeros(n, dtype=np.float32)
        self.start_x = np.zeros(n, dtype=np.float32)

        self.best_reward = -1e9
        self.best_distance = 0.0
        self.best_rew_t = torch.full((1,), -1e9, device=self.device)
        self.best_dist_t = torch.zeros(1, device=self.device)

        self._completed: list[tuple[float, float, int, float]] = []
        self._obs_np = np.zeros((n, self.OBS_DIM), dtype=np.float32)
        self._rew_np = np.zeros(n, dtype=np.float32)
        self._done_np = np.zeros(n, dtype=np.float32)

        self.reset()

    def set_stage(self, stage: int) -> None:
        self.stage = int(stage) if int(stage) in (0, 1, 2) else 0
        self.model.actuator_gainprm[:] = self._base_gainprm
        self.model.actuator_biasprm[:] = self._base_biasprm
        self.model.actuator_forcerange[:] = self._base_forcerange
        if self.stage == 2:
            # Give waist, shoulders, and elbows athletic push-up strength exclusively during Stand Up mode
            boost = (
                (2, 900.0, 60.0, 1200.0),
                (3, 950.0, 52.0, 1200.0),
                (4, 850.0, 48.0, 1100.0),
                (5, 800.0, 42.0, 1000.0),
                (6, 950.0, 52.0, 1200.0),
                (7, 850.0, 48.0, 1100.0),
                (8, 800.0, 42.0, 1000.0),
            )
            for idx, kp, kv, fm in boost:
                self.model.actuator_gainprm[idx, 0] = kp
                self.model.actuator_biasprm[idx, 1] = -kp
                self.model.actuator_biasprm[idx, 2] = -kv
                self.model.actuator_forcerange[idx] = [-fm, fm]

    def apply_impulse(self, dvx: float = 0.0, dvz: float = 0.0, env_idx: int = 0) -> None:
        """Apply physical directional impulse in Three.js world frame (X along road, Z lateral)."""
        d = self.datas[env_idx]
        dvy = -float(dvz)
        d.qvel[0] += float(dvx)
        d.qvel[1] += dvy
        d.qvel[3] -= dvy * 0.14
        d.qvel[4] += float(dvx) * 0.14

    def _reset_env(self, i: int) -> None:
        d = self.datas[i]
        mujoco.mj_resetData(self.model, d)

        d.qpos[0] = 0.0
        d.qpos[1] = float(np.random.uniform(-0.08, 0.08))
        if self.stage == 2:
            if i == 0:
                pose_kind = self.hero_getup_idx % 4
                self.hero_getup_idx += 1
            else:
                pose_kind = i % 4
            yaw_q = _axis_angle_quat(0.0, 0.0, 1.0, float(np.random.uniform(-0.35, 0.35)))
            if pose_kind == 0:
                tilt_q = _axis_angle_quat(0.0, 1.0, 0.0, float(np.random.uniform(-1.54, -1.40)))
                d.qpos[2] = 0.62
            elif pose_kind == 1:
                tilt_q = _axis_angle_quat(0.0, 1.0, 0.0, float(np.random.uniform(1.40, 1.54)))
                d.qpos[2] = 0.62
            elif pose_kind == 2:
                tilt_q = _axis_angle_quat(1.0, 0.0, 0.0, float(np.random.uniform(1.38, 1.52)))
                d.qpos[2] = 1.10
            else:
                tilt_q = _axis_angle_quat(1.0, 0.0, 0.0, float(np.random.uniform(-1.52, -1.38)))
                d.qpos[2] = 1.10
            d.qpos[3:7] = _quat_mul(yaw_q, tilt_q)
            q0 = np.clip(J_NOM + np.random.uniform(-0.22, 0.22, size=self.ACT_DIM), J_MIN, J_MAX)
            d.qpos[self.qpos_ids] = q0
            d.ctrl[:] = q0
            d.qvel[:] = 0.0
            mujoco.mj_step(self.model, d, 10)
            d.qvel[:] *= 0.2
        else:
            d.qpos[2] = 2.695
            d.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            q0 = np.clip(J_NOM + np.random.uniform(-0.04, 0.04, size=self.ACT_DIM), J_MIN, J_MAX)
            d.qpos[self.qpos_ids] = q0
            d.ctrl[:] = q0
            if self.stage == 1 and i > 0:
                d.qvel[0] = float(np.random.uniform(0.0, 0.55))
            else:
                d.qvel[0] = float(np.random.uniform(-0.06, 0.06))
            d.qvel[1] = float(np.random.uniform(-0.06, 0.06))
            d.qvel[3:6] = np.random.uniform(-0.05, 0.05, size=3)
            mujoco.mj_forward(self.model, d)

        self.q_cmd[i] = q0.astype(np.float32)
        self.prev_act[i].fill(0.0)
        self.step_count[i] = 0
        self.walk_phase[i] = 0.0 if (i == 0 or i % 2 == 0) else math.pi
        self.foot_contact[i] = 1.0
        self.last_single_foot[i] = 0
        self.last_step_tick[i] = 0
        self.prev_rf_x[i] = float(d.xpos[self.rf_id, 0])
        self.prev_lf_x[i] = float(d.xpos[self.lf_id, 0])
        self.ep_reward[i] = 0.0
        self.ep_upright[i] = 0.0
        self.start_x[i] = float(d.qpos[0])

    def reset(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            for i in range(self.num_envs):
                self._reset_env(i)
        else:
            for i in env_ids.tolist():
                self._reset_env(int(i))
        self._build_obs()
        return self.get_obs()

    def _measure_road_contacts(self, d: mujoco.MjData) -> tuple[float, float, int, int]:
        cr, cl = 0.0, 0.0
        arm_r, arm_l, th_r, th_l = 0, 0, 0, 0
        for c_idx in range(d.ncon):
            con = d.contact[c_idx]
            g1, g2 = con.geom1, con.geom2
            if g1 == self.road_geom_id or g2 == self.road_geom_id:
                other = g2 if g1 == self.road_geom_id else g1
                if other == self.r_sole_id:
                    cr = 1.0
                elif other == self.l_sole_id:
                    cl = 1.0
                elif other in (self.r_farm_id, self.r_uarm_id):
                    arm_r = 1
                elif other in (self.l_farm_id, self.l_uarm_id):
                    arm_l = 1
                elif other == self.r_thigh_id:
                    th_r = 1
                elif other == self.l_thigh_id:
                    th_l = 1
        return cr, cl, arm_r + arm_l, th_r + th_l

    def _measure_foot_contacts(self, d: mujoco.MjData) -> tuple[float, float]:
        cr, cl, _, _ = self._measure_road_contacts(d)
        return cr, cl

    def _build_obs(self) -> None:
        for i, d in enumerate(self.datas):
            cr, cl = self._measure_foot_contacts(d)
            self.foot_contact[i, 0] = cr
            self.foot_contact[i, 1] = cl
            pitch, yaw, roll = mj_quat_to_three_eul_yxz(d.qpos[3:7])
            yaw_err = _wrap_angle(yaw - math.pi / 2.0)
            cy, sy = math.cos(yaw_err), math.sin(yaw_err)

            vx_w, vy_w = float(d.qvel[0]), float(d.qvel[1])
            vx_loc = vx_w * cy + vy_w * sy
            vy_loc = -vx_w * sy + vy_w * cy

            q17 = d.qpos[self.qpos_ids].astype(np.float32)
            qd17 = d.qvel[self.qvel_ids].astype(np.float32)
            q_norm = (q17 - self.j_mid) / (self.j_half + 1e-6)
            qd_norm = qd17 / 10.0

            o = self._obs_np[i]
            o[0] = (float(d.qpos[2]) - 2.65) / 1.0
            o[1] = vx_loc / 5.0
            o[2] = float(d.qvel[2]) / 5.0
            o[3] = -vy_loc / 5.0
            o[4] = pitch
            o[5] = yaw_err
            o[6] = roll
            o[7] = float(d.qvel[4]) / 8.0
            o[8] = float(d.qvel[5]) / 8.0
            o[9] = float(d.qvel[3]) / 8.0
            o[10:27] = q_norm
            o[27:44] = qd_norm
            o[44] = cr
            o[45] = cl
            if self.stage == 1:
                # Provide purely clock-phase (sin, cos) + lateral road offset for Walk RL
                o[46] = math.sin(float(self.walk_phase[i]))
                o[47] = math.cos(float(self.walk_phase[i]))
                o[48] = -float(d.qpos[1]) / 3.0
            elif self.stage == 2:
                # Singularity-free world +Z up vector expressed in pelvis body frame
                w, x, y, qz = d.qpos[3:7]
                o[46] = float(2.0 * (x * qz - w * y))
                o[47] = float(2.0 * (y * qz + w * x))
                o[48] = float(1.0 - 2.0 * (x * x + y * y))
            else:
                o[46] = 0.0
                o[47] = float(d.xpos[self.rf_id, 0] - d.xpos[self.lf_id, 0])
                o[48] = -float(d.qpos[1]) / 3.0

    def get_obs(self) -> torch.Tensor:
        return torch.from_numpy(self._obs_np.copy()).to(self.device)

    def _prepare_ctrl(self, i: int, act_i: np.ndarray) -> None:
        d = self.datas[i]

        # Regular directional pushes in Stand/Walk so pure RL handles balance (off during ground get-up)
        if self.training_pushes and self.stage != 2 and self.step_count[i] > 28:
            p_thresh = 0.015 if self.stage == 0 else 0.0015
            if np.random.random() < p_thresh:
                pitch, _, roll = mj_quat_to_three_eul_yxz(d.qpos[3:7])
                if abs(pitch) < 0.20 and abs(roll) < 0.20:
                    axis_sag = np.random.random() < 0.5
                    sign = -1.0 if np.random.random() < 0.5 else 1.0
                    if axis_sag:
                        dvx = sign * float(np.random.uniform(0.55, 1.18))
                        dvy = 0.0
                    else:
                        dvx = 0.0
                        dvy = sign * float(np.random.uniform(0.55, 1.10))
                    d.qvel[0] += dvx
                    d.qvel[1] += dvy
                    d.qvel[3] -= dvy * 0.14
                    d.qvel[4] += dvx * 0.14

        raw_target = np.clip(J_NOM + act_i * ACT_SCALE, J_MIN, J_MAX)
        self.q_cmd[i] += 0.65 * (raw_target - self.q_cmd[i])
        d.ctrl[:] = self.q_cmd[i]

    def step_gpu(self, action: torch.Tensor, dt_ctrl: float = 0.02, substeps: int = 4):
        """Step all parallel MuJoCo worlds across C threads and return GPU (obs, reward, done)."""
        act_np = np.clip(action.detach().cpu().numpy(), -1.0, 1.0).astype(np.float32)

        for i in range(self.num_envs):
            self._prepare_ctrl(i, act_np[i])

        def _step_one(d: mujoco.MjData) -> None:
            mujoco.mj_step(self.model, d, substeps)

        list(self._pool.map(_step_one, self.datas))

        best_rew = float(self.best_rew_t[0].item())
        best_dist = float(self.best_dist_t[0].item())

        for i, d in enumerate(self.datas):
            self.step_count[i] += 1
            pitch, yaw, roll = mj_quat_to_three_eul_yxz(d.qpos[3:7])
            yaw_err = _wrap_angle(yaw - math.pi / 2.0)
            cy, sy = math.cos(yaw_err), math.sin(yaw_err)

            vx_w, vy_w = float(d.qvel[0]), float(d.qvel[1])
            vx_loc = vx_w * cy + vy_w * sy
            vy_loc = -vx_w * sy + vy_w * cy
            z = float(d.qpos[2])
            q17 = d.qpos[self.qpos_ids]

            upright = math.exp(-5.0 * (pitch * pitch + roll * roll))
            heading_ok = math.exp(-3.0 * yaw_err * yaw_err)
            posture_pen = 0.004 * float(np.mean((q17 - J_NOM) ** 2))
            energy_pen = (
                0.005 * float(np.mean(act_np[i] ** 2))
                + 0.020 * float(np.mean((act_np[i] - self.prev_act[i]) ** 2))
            )
            self.prev_act[i] = act_np[i]

            if self.stage == 0:
                height_ok = math.exp(-4.5 * (z - 2.64) ** 2)
                calm = math.exp(-0.35 * (vx_w * vx_w + vy_w * vy_w) - 0.06 * float(np.sum(d.qvel[3:6] ** 2)))
                rew = 0.30 + 1.8 * upright * height_ok + 0.6 * heading_ok + 0.6 * calm - posture_pen - energy_pen
            elif self.stage == 1:
                # Never penalize natural pelvis drop (z down to 2.28) during Manual-Sandbox human strides
                walk_h_ok = math.exp(-3.2 * max(0.0, 2.28 - z) ** 2 - 2.0 * max(0.0, z - 2.74) ** 2)
                self.walk_phase[i] = (self.walk_phase[i] + dt_ctrl * self.walk_cadence) % (2.0 * math.pi)
                cr, cl = self._measure_foot_contacts(d)
                rf_x, lf_x = float(d.xpos[self.rf_id, 0]), float(d.xpos[self.lf_id, 0])
                rf_y, lf_y = float(d.xpos[self.rf_id, 1]), float(d.xpos[self.lf_id, 1])
                foot_sep_loc = (rf_x - lf_x) * cy + (rf_y - lf_y) * sy

                # Smooth ramp from stand into full Manual Sandbox stride (~0.40s)
                wb = min(1.0, (float(self.step_count[i]) * dt_ctrl) / 0.40)
                s_ph = math.sin(float(self.walk_phase[i]))
                c_ph = math.cos(float(self.walk_phase[i]))

                # Prevent double-foot jumping/bunny-hopping cleanly (walking keeps >=1 foot on road)
                airborne = cr < 0.5 and cl < 0.5
                sym_knee_pen = 1.25 * max(0.0, min(float(q17[11]), float(q17[14])) - 0.35)

                # Enforce unhurried alternating human strides (~0.64s per step, wide foot span >= 0.85u)
                stride_bonus = 0.0
                tiny_step_pen = 0.0
                span = abs(foot_sep_loc)
                if not airborne:
                    new_stance = 1 if (cr > 0.5 and cl < 0.5) else (2 if (cl > 0.5 and cr < 0.5) else 0)
                    if new_stance != 0 and new_stance != self.last_single_foot[i]:
                        ticks_since = int(self.step_count[i] - self.last_step_tick[i])
                        if self.last_single_foot[i] != 0:
                            if wb >= 1.0 and (ticks_since < 17 or span < 0.68):
                                tiny_step_pen = 0.80
                            elif ticks_since >= 20 and span >= 0.85 and vx_loc > 1.50 and upright > 0.60:
                                stride_bonus = 3.20 * min(1.6, span / 1.10)
                        self.last_single_foot[i] = new_stance
                        self.last_step_tick[i] = int(self.step_count[i])

                # Non-negative phase alignment reward for single-foot swing clearance
                stance_align = 0.0
                if wb >= 1.0:
                    if (c_ph > 0.35 and cl > 0.5 and cr < 0.5) or (c_ph < -0.35 and cr > 0.5 and cl < 0.5):
                        stance_align = 0.75

                # Keep gait_rew strictly >= 0 so higher forward speed always increases total return
                speed_gate = min(1.0, max(0.0, (vx_loc - 0.50) / 1.50))

                rh_ref = 0.08 + 0.45 * s_ph * wb
                lh_ref = 0.08 - 0.45 * s_ph * wb
                rk_ref = 0.14 + max(0.0, c_ph) * 0.90 * wb
                lk_ref = 0.14 + max(0.0, -c_ph) * 0.90 * wb
                ra_ref = max(-0.50, min(0.50, 0.70 * (rh_ref - rk_ref)))
                la_ref = max(-0.50, min(0.50, 0.70 * (lh_ref - lk_ref)))
                rs_ref = -0.45 * s_ph * wb
                ls_ref = 0.45 * s_ph * wb
                arm_elb_ref = 0.15 + 0.42 * wb
                wt_ref = 0.10 * s_ph * wb

                # Exact kinematic foot-separation target matching Manual Sandbox joint angles
                rf_kin = 1.20 * math.sin(rh_ref) + 1.15 * math.sin(rh_ref - rk_ref)
                lf_kin = 1.20 * math.sin(lh_ref) + 1.15 * math.sin(lh_ref - lk_ref)
                stride_target = rf_kin - lf_kin
                sep_err = (foot_sep_loc - stride_target) ** 2

                # Bounded tent + Gaussian terms: constant non-vanishing gradient inside basin, zero overshoot past target
                sep_tri = max(0.0, 1.0 - abs(foot_sep_loc - stride_target) / 2.20)
                hip_tri = max(0.0, 1.0 - (abs(float(q17[9]) - rh_ref) + abs(float(q17[12]) - lh_ref)) / 1.40)
                knee_tri = max(0.0, 1.0 - (abs(float(q17[11]) - rk_ref) + abs(float(q17[14]) - lk_ref)) / 1.80)
                arm_tri = max(0.0, 1.0 - (abs(float(q17[3]) - rs_ref) + abs(float(q17[6]) - ls_ref)) / 1.50)

                leg_err = (
                    1.6 * ((q17[9] - rh_ref) ** 2 + (q17[12] - lh_ref) ** 2)
                    + 1.0 * ((q17[11] - rk_ref) ** 2 + (q17[14] - lk_ref) ** 2)
                    + 0.35 * ((q17[15] - ra_ref) ** 2 + (q17[16] - la_ref) ** 2)
                )
                arm_err = (
                    0.9 * ((q17[3] - rs_ref) ** 2 + (q17[6] - ls_ref) ** 2)
                    + 0.4 * ((q17[5] - arm_elb_ref) ** 2 + (q17[8] - arm_elb_ref) ** 2)
                    + 0.3 * ((q17[2] - wt_ref) ** 2)
                )

                stride_match = 1.15 * sep_tri + 0.95 * math.exp(-1.6 * sep_err)
                leg_style = 0.95 * hip_tri + 0.75 * knee_tri + 0.85 * math.exp(-1.5 * leg_err)
                arm_style = 0.35 * arm_tri + 0.35 * math.exp(-0.9 * arm_err)
                gait_rew = speed_gate * (stride_match + leg_style + arm_style + stance_align)

                fwd_prog = max(-0.3, min(1.15, vx_loc / self.target_speed))
                vel_match = math.exp(-0.9 * (vx_loc - self.target_speed) ** 2)
                lat_pen = 0.22 * abs(float(d.qpos[1])) + 0.08 * abs(vy_loc)
                if airborne:
                    rew = -0.60 - sym_knee_pen - lat_pen - energy_pen
                else:
                    rew = (
                        (0.20 + 1.25 * fwd_prog + 1.00 * vel_match + gait_rew) * (0.25 + 0.75 * upright * walk_h_ok)
                        + 1.15 * upright * heading_ok
                        + stride_bonus
                        - tiny_step_pen
                        - sym_knee_pen
                        - lat_pen
                        - energy_pen
                    )
            else:
                # Stage 2: Pure-RL full-body get-up (arms/forearms + knees/thighs + torso + legs)
                cr, cl, arm_cnt, thigh_cnt = self._measure_road_contacts(d)
                _, qx, qy, _ = d.qpos[3:7]
                r22 = float(1.0 - 2.0 * (qx * qx + qy * qy))
                torso_up = float(d.xmat[self.torso_id, 8])
                upright = max(0.0, 0.5 * (r22 + torso_up)) ** 2

                pelvis_h = max(0.0, min(1.0, (z - 0.45) / (2.60 - 0.45)))
                chest_z = float(d.geom_xpos[self.chest_geom_id, 2])
                chest_h = max(0.0, min(1.0, (chest_z - 0.50) / (3.80 - 0.50)))
                head_z = float(d.xpos[self.head_id, 2])
                head_h = max(0.0, min(1.0, (head_z - 0.55) / (4.70 - 0.55)))
                body_lift = min(chest_h, pelvis_h)

                lift_active = min(1.0, max(0.0, (chest_z - 0.58) / 0.65))
                arm_push = 0.70 * (arm_cnt / 2.0) * lift_active if z < 2.30 else 0.0
                knee_brace = 0.50 * (thigh_cnt / 2.0) * min(1.0, pelvis_h * 2.5) if z < 2.20 else 0.0

                foot_support = 1.0 if (cr > 0.5 and cl > 0.5) else (0.70 if (cr > 0.5 or cl > 0.5) else 0.15)
                tilt_upright = math.exp(-5.0 * (max(0.0, 1.0 - r22) + max(0.0, 1.0 - torso_up)))
                stand_height = math.exp(-4.0 * max(0.0, 2.58 - z) ** 2 - 2.0 * max(0.0, z - 2.75) ** 2)
                stand_ok = tilt_upright * stand_height * foot_support

                calm = math.exp(-0.25 * (vx_w * vx_w + vy_w * vy_w) - 0.05 * float(np.sum(d.qvel[3:6] ** 2)))
                posture_hold = math.exp(-0.5 * float(np.mean((q17 - J_NOM) ** 2)))
                rise_vel = max(-0.3, min(0.8, float(d.qvel[2]) * 0.40)) if z < 2.45 else 0.0
                up_prog = 0.5 * (max(0.0, torso_up) + max(0.0, r22))
                spin_pen = 0.03 * min(6.0, float(np.sum(d.qvel[3:6] ** 2))) * (1.0 - stand_ok)

                rew = (
                    0.10
                    + 1.50 * body_lift
                    + 1.35 * chest_h
                    + 1.05 * head_h
                    + 0.70 * rise_vel
                    + 1.25 * up_prog * (0.20 + 0.80 * chest_h)
                    + arm_push
                    + knee_brace
                    + 4.80 * stand_ok
                    + 1.20 * stand_ok * (0.5 * calm + 0.5 * posture_hold)
                    - spin_pen
                    - energy_pen
                    - stand_ok * posture_pen
                )

            if self.stage == 2:
                fallen = abs(float(d.qpos[0])) > 318.0 or abs(float(d.qpos[1])) > 8.0 or z < -0.5
            else:
                fallen = (
                    z < 1.50
                    or abs(pitch) > 0.88
                    or abs(roll) > 0.88
                    or abs(float(d.qpos[1])) > 3.6
                    or abs(float(d.qpos[0])) > 318.0
                )
            done = fallen or (self.step_count[i] >= MAX_EPISODE_STEPS)

            self.ep_reward[i] += rew
            self.ep_upright[i] += upright
            self._rew_np[i] = rew
            self._done_np[i] = 1.0 if done else 0.0

            if done:
                steps_f = max(1, int(self.step_count[i]))
                dist = float(d.qpos[0]) - float(self.start_x[i])
                self._completed.append((float(self.ep_reward[i]), float(self.ep_upright[i]) / steps_f, steps_f, dist))
                if self.ep_reward[i] > best_rew:
                    best_rew = float(self.ep_reward[i])
                if self.stage == 1 and dist > best_dist:
                    best_dist = dist
                self._reset_env(i)

        self.best_reward = best_rew
        self.best_distance = best_dist
        self.best_rew_t.fill_(best_rew)
        self.best_dist_t.fill_(best_dist)

        self._build_obs()
        obs_t = torch.from_numpy(self._obs_np.copy()).to(self.device)
        rew_t = torch.from_numpy(self._rew_np.copy()).to(self.device)
        done_t = torch.from_numpy(self._done_np.copy()).to(self.device) > 0.5
        return obs_t, rew_t, done_t

    def step(self, action: torch.Tensor, dt_ctrl: float = 0.02, substeps: int = 4):
        obs, rew, done = self.step_gpu(action, dt_ctrl, substeps)
        return obs, rew, done, []

    def consume_gpu_stats(self) -> dict | None:
        if not self._completed:
            return {
                "count": 0,
                "best_reward": float(self.best_rew_t[0].item()),
                "best_distance": float(self.best_dist_t[0].item()),
            }
        cnt = len(self._completed)
        arr = np.array(self._completed, dtype=np.float32)
        self._completed.clear()
        return {
            "count": cnt,
            "avg_reward": float(arr[:, 0].mean()),
            "avg_upright": float(arr[:, 1].mean()),
            "avg_length": float(arr[:, 2].mean()),
            "avg_distance": float(arr[:, 3].mean()),
            "best_reward": float(self.best_rew_t[0].item()),
            "best_distance": float(self.best_dist_t[0].item()),
        }

    def get_hero_snapshot(self) -> dict:
        """Extract environment 0 pose mapped 1:1 into Three.js coordinate convention."""
        d = self.datas[0]
        pitch, yaw, roll = mj_quat_to_three_eul_yxz(d.qpos[3:7])
        q_three = mj_quat_to_three_quat_xyzw(d.qpos[3:7])
        q17 = d.qpos[self.qpos_ids]
        cr, cl = self._measure_foot_contacts(d)
        return {
            "pos": [round(float(d.qpos[0]), 4), round(float(d.qpos[2]), 4), round(-float(d.qpos[1]), 4)],
            "eul": [round(pitch, 4), round(yaw, 4), round(roll, 4)],
            "quat": [round(float(q_three[0]), 5), round(float(q_three[1]), 5), round(float(q_three[2]), 5), round(float(q_three[3]), 5)],
            "joints": [round(float(x), 4) for x in q17],
            "contacts": [int(cr > 0.5), int(cl > 0.5)],
            "speed": round(float(d.qvel[0]), 3),
            "step": int(self.step_count[0]),
        }
