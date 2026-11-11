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
)

MAX_EPISODE_STEPS = 480
XML_PATH = Path(__file__).resolve().parent / "sentinel.xml"


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


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
        self.rf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "r_foot")
        self.lf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "l_foot")
        self.r_sole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "r_sole")
        self.l_sole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "l_sole")
        self.road_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "road")

        self.j_mid = 0.5 * (J_MIN + J_MAX)
        self.j_half = 0.5 * (J_MAX - J_MIN)

        self.stage = 0  # 0 = Stand & 4-way Push Recovery, 1 = Walk
        self.target_speed = 1.25
        self.training_pushes = True

        # Per-env telemetry and smoothing buffers
        n = num_envs
        self.q_cmd = np.tile(J_NOM, (n, 1))
        self.prev_act = np.zeros((n, self.ACT_DIM), dtype=np.float32)
        self.step_count = np.zeros(n, dtype=np.int64)
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
        self.stage = 1 if stage >= 1 else 0

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
        d.qpos[2] = 2.695
        d.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        q0 = np.clip(J_NOM + np.random.uniform(-0.04, 0.04, size=self.ACT_DIM), J_MIN, J_MAX)
        d.qpos[self.qpos_ids] = q0
        d.ctrl[:] = q0
        if self.stage == 1 and i > 0:
            d.qvel[0] = float(np.random.uniform(0.0, 0.45))
        else:
            d.qvel[0] = float(np.random.uniform(-0.06, 0.06))
        d.qvel[1] = float(np.random.uniform(-0.06, 0.06))
        d.qvel[3:6] = np.random.uniform(-0.05, 0.05, size=3)
        mujoco.mj_forward(self.model, d)

        self.q_cmd[i] = q0.astype(np.float32)
        self.prev_act[i].fill(0.0)
        self.step_count[i] = 0
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

    def _measure_foot_contacts(self, d: mujoco.MjData) -> tuple[float, float]:
        cr, cl = 0.0, 0.0
        for c_idx in range(d.ncon):
            con = d.contact[c_idx]
            g1, g2 = con.geom1, con.geom2
            if g1 == self.road_geom_id or g2 == self.road_geom_id:
                other = g2 if g1 == self.road_geom_id else g1
                if other == self.r_sole_id:
                    cr = 1.0
                elif other == self.l_sole_id:
                    cl = 1.0
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
            o[46] = float(self.stage)
            # Heading & foot separation symmetry hints (purely kinematic sensor readings)
            o[47] = float(d.xpos[self.rf_id, 0] - d.xpos[self.lf_id, 0])
            o[48] = -float(d.qpos[1]) / 3.0

    def get_obs(self) -> torch.Tensor:
        return torch.from_numpy(self._obs_np.copy()).to(self.device)

    def _prepare_ctrl(self, i: int, act_i: np.ndarray) -> None:
        d = self.datas[i]

        # Regular directional pushes in Stand mode so pure RL learns Front/Back/Left/Right recovery
        if self.training_pushes and self.step_count[i] > 28:
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
            height_ok = math.exp(-4.5 * (z - 2.64) ** 2)
            heading_ok = math.exp(-3.0 * yaw_err * yaw_err)
            posture_pen = 0.004 * float(np.mean((q17 - J_NOM) ** 2))
            energy_pen = (
                0.005 * float(np.mean(act_np[i] ** 2))
                + 0.020 * float(np.mean((act_np[i] - self.prev_act[i]) ** 2))
            )
            self.prev_act[i] = act_np[i]

            if self.stage == 0:
                calm = math.exp(-0.35 * (vx_w * vx_w + vy_w * vy_w) - 0.06 * float(np.sum(d.qvel[3:6] ** 2)))
                rew = 0.30 + 1.8 * upright * height_ok + 0.6 * heading_ok + 0.6 * calm - posture_pen - energy_pen
            else:
                cr, cl = self._measure_foot_contacts(d)
                rf_x, lf_x = float(d.xpos[self.rf_id, 0]), float(d.xpos[self.lf_id, 0])
                step_bonus = 0.0
                if vx_loc > 0.30 and upright > 0.45:
                    if cr > 0.5 and cl < 0.5 and self.last_single_foot[i] != 1 and (self.step_count[i] - self.last_step_tick[i]) >= 7:
                        if self.last_single_foot[i] == 2:
                            step_bonus = 0.45
                        self.last_single_foot[i] = 1
                        self.last_step_tick[i] = int(self.step_count[i])
                    elif cl > 0.5 and cr < 0.5 and self.last_single_foot[i] != 2 and (self.step_count[i] - self.last_step_tick[i]) >= 7:
                        if self.last_single_foot[i] == 1:
                            step_bonus = 0.45
                        self.last_single_foot[i] = 2
                        self.last_step_tick[i] = int(self.step_count[i])
                self.prev_rf_x[i] = rf_x
                self.prev_lf_x[i] = lf_x

                fwd_prog = max(-0.3, min(1.15, vx_loc / self.target_speed))
                vel_match = math.exp(-1.8 * (vx_loc - self.target_speed) ** 2)
                lat_pen = 0.10 * abs(float(d.qpos[1])) + 0.08 * abs(vy_loc)
                rew = (
                    (0.15 + 1.75 * fwd_prog + 1.25 * vel_match) * (0.25 + 0.75 * upright * height_ok)
                    + 0.65 * upright * heading_ok
                    + step_bonus
                    - lat_pen
                    - posture_pen
                    - energy_pen
                )

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
        q17 = d.qpos[self.qpos_ids]
        cr, cl = self._measure_foot_contacts(d)
        return {
            "pos": [round(float(d.qpos[0]), 4), round(float(d.qpos[2]), 4), round(-float(d.qpos[1]), 4)],
            "eul": [round(pitch, 4), round(yaw, 4), round(roll, 4)],
            "joints": [round(float(x), 4) for x in q17],
            "contacts": [int(cr > 0.5), int(cl > 0.5)],
            "speed": round(float(d.qvel[0]), 3),
            "step": int(self.step_count[0]),
        }
