"""GPU-first async PPO training manager with completely isolated Stand and Walk models."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from pathlib import Path
from typing import Awaitable, Callable

import torch

from .physics import BipedPhysicsEnv
from .ppo import PPOTrainer

MODES = ("stand", "walk")


def _new_mode_state(mode: str) -> dict:
    label = "Stand & Push Recovery" if mode == "stand" else "Walk Locomotion"
    return {
        "episode": 0,
        "updates": 0,
        "total_steps": 0,
        "best_reward": -1e9,
        "best_distance": 0.0,
        "avg_reward": 0.0,
        "avg_upright": 0.0,
        "avg_length": 0.0,
        "avg_distance": 0.0,
        "last_losses": {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0},
        "last_milestone": f"Isolated {label} mode ({mode}_ppo.pt)",
        "last_save_time": 0.0,
        "history": deque(maxlen=160),
    }


class LearningManager:
    """Runs multi-threaded MuJoCo physics + separate isolated Stand and Walk GPU PPO trainers."""

    def __init__(
        self,
        models_dir: Path | str = "models",
        num_envs: int = 24,
        rollout_steps: int = 64,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = BipedPhysicsEnv(num_envs=num_envs, device=self.device)
        self.trainers: dict[str, PPOTrainer] = {
            mode: PPOTrainer(
                obs_dim=self.env.OBS_DIM,
                act_dim=self.env.ACT_DIM,
                device=self.device,
                models_dir=models_dir,
                model_name=mode,
            )
            for mode in MODES
        }
        self.mode_states: dict[str, dict] = {mode: _new_mode_state(mode) for mode in MODES}
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps

        # Runtime controls
        self.paused = False
        self.speed_mult = 1.0
        self.active_mode = "stand"  # "stand" | "walk"
        self.play_mode = "train"  # "train" | "eval"

        self._subscribers: set[Callable[[dict], Awaitable[None]]] = set()
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._rollout_reset_req = False

        # Restore each isolated mode checkpoint independently if present on disk
        for mode in MODES:
            loaded = self.trainers[mode].load(prefer_best=False)
            if loaded:
                self._apply_loaded_stats(mode, loaded)
                ms = self.mode_states[mode]
                label = "Walk" if mode == "walk" else "Stand"
                ms["last_milestone"] = f"Loaded {label} model (Ep {ms['episode']} · {mode}_ppo.pt)"

        self.env.set_stage(0 if self.active_mode == "stand" else 1)
        self._sync_env_bests()
        self.env.reset()

        self._s_obs = torch.zeros(num_envs, self.env.OBS_DIM, device=self.device)
        self._s_act = torch.zeros(num_envs, self.env.ACT_DIM, device=self.device)
        self._s_logp = torch.zeros(num_envs, device=self.device)
        self._s_val = torch.zeros(num_envs, device=self.device)
        self._s_rew = torch.zeros(num_envs, device=self.device)
        self._s_done = torch.zeros(num_envs, device=self.device)
        self._s_next_obs = torch.zeros(num_envs, self.env.OBS_DIM, device=self.device)
        self._s_obs.copy_(self.env.get_obs())

    @property
    def active_trainer(self) -> PPOTrainer:
        return self.trainers[self.active_mode]

    @property
    def _ms(self) -> dict:
        return self.mode_states[self.active_mode]

    def _sync_env_bests(self) -> None:
        ms = self._ms
        self.env.best_reward = float(ms["best_reward"])
        self.env.best_distance = float(ms["best_distance"])
        self.env.best_rew_t.fill_(ms["best_reward"])
        self.env.best_dist_t.fill_(ms["best_distance"])

    def subscribe(self, cb: Callable[[dict], Awaitable[None]]) -> None:
        self._subscribers.add(cb)

    def unsubscribe(self, cb: Callable[[dict], Awaitable[None]]) -> None:
        self._subscribers.discard(cb)

    async def _broadcast(self, msg: dict) -> None:
        if not self._subscribers:
            return
        dead = []
        for cb in list(self._subscribers):
            try:
                await cb(msg)
            except Exception:
                dead.append(cb)
        for cb in dead:
            self._subscribers.discard(cb)

    def _stats_snapshot(self) -> dict:
        vram_mb = (
            round(torch.cuda.memory_allocated(self.device) / (1024 * 1024), 1)
            if self.device.type == "cuda"
            else 0.0
        )
        ms = self._ms
        return {
            "type": "telemetry",
            "device": torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU",
            "vram_mb": vram_mb,
            "paused": self.paused,
            "speed_mult": self.speed_mult,
            "active_mode": self.active_mode,
            "curriculum_mode": self.active_mode,
            "play_mode": self.play_mode,
            "stage": 1 if self.active_mode == "walk" else 0,
            "episode": ms["episode"],
            "updates": ms["updates"],
            "total_steps": ms["total_steps"],
            "avg_reward": round(ms["avg_reward"], 2),
            "best_reward": round(ms["best_reward"] if ms["best_reward"] > -1e8 else 0.0, 2),
            "best_distance": round(ms["best_distance"], 2),
            "avg_upright": round(ms["avg_upright"], 3),
            "avg_length": round(ms["avg_length"], 1),
            "avg_distance": round(ms["avg_distance"], 2),
            **ms["last_losses"],
            "milestone": ms["last_milestone"],
            "model_info": self.active_trainer.get_model_info(),
            "models": {m: self.trainers[m].get_model_info() for m in MODES},
            "history": list(ms["history"]),
        }

    def get_init_payload(self) -> dict:
        payload = self._stats_snapshot()
        payload["type"] = "init"
        payload["hero"] = self.env.get_hero_snapshot()
        return payload

    def _resolve_mode(self, mode: str | None) -> str:
        if mode in MODES:
            return mode
        return self.active_mode

    def _apply_loaded_stats(self, mode: str, loaded: dict) -> None:
        ms = self.mode_states[mode]
        ms["episode"] = int(loaded.get("episode", 0))
        ms["updates"] = int(loaded.get("updates", 0))
        ms["total_steps"] = int(loaded.get("total_steps", 0))
        ms["best_reward"] = float(loaded.get("best_reward", -1e9))
        ms["best_distance"] = float(loaded.get("best_distance", 0.0))
        ms["avg_reward"] = float(
            loaded.get("avg_reward", max(0.0, ms["best_reward"] * 0.8) if ms["best_reward"] > 0 else 0.0)
        )
        ms["avg_upright"] = float(loaded.get("avg_upright", 0.90 if ms["episode"] > 500 else 0.0))
        ms["avg_length"] = float(loaded.get("avg_length", 350.0 if ms["episode"] > 500 else 0.0))
        ms["avg_distance"] = float(loaded.get("avg_distance", 0.0))
        ms["history"].clear()
        if ms["updates"] > 0:
            ms["history"].append(
                {
                    "u": ms["updates"],
                    "ep": ms["episode"],
                    "r": round(ms["avg_reward"], 2),
                    "up": round(ms["avg_upright"], 3),
                    "d": round(ms["avg_distance"], 2),
                    "l": round(ms["avg_length"], 1),
                    "st": 1 if mode == "walk" else 0,
                }
            )

    def save_now(self, mode: str | None = None, export_onnx: bool = True, allow_best: bool = True) -> dict:
        target = self._resolve_mode(mode)
        ms = self.mode_states[target]
        stats = {
            "episode": ms["episode"],
            "updates": ms["updates"],
            "total_steps": ms["total_steps"],
            "stage": 1 if target == "walk" else 0,
            "best_reward": ms["best_reward"] if ms["best_reward"] > -1e8 else 0.0,
            "best_distance": ms["best_distance"],
            "avg_reward": ms["avg_reward"],
            "avg_upright": ms["avg_upright"],
            "avg_length": ms["avg_length"],
            "avg_distance": ms["avg_distance"],
        }
        meta = self.trainers[target].save(stats, export_onnx=export_onnx, allow_best=allow_best)
        ms["last_save_time"] = time.time()
        ms["last_milestone"] = f"Saved {target}_ppo.pt (Ep {ms['episode']}, {meta['pt_bytes'] // 1024} KB)"
        return meta

    async def load_saved(self, mode: str | None = None) -> dict | None:
        target = self._resolve_mode(mode)
        async with self._lock:
            loaded = self.trainers[target].load(prefer_best=True)
            if not loaded:
                return None
            self._apply_loaded_stats(target, loaded)
            ms = self.mode_states[target]
            if target == self.active_mode:
                self._sync_env_bests()
                self.env.reset()
                self.env.consume_gpu_stats()
                self._s_obs.copy_(self.env.get_obs())
                self._rollout_reset_req = True
            ms["last_save_time"] = time.time()
            src_label = "best saved model" if loaded.get("_loaded_source") == "best" else "saved model"
            label = "Walk" if target == "walk" else "Stand"
            ms["last_milestone"] = f"Restored {label} {src_label} (Ep {ms['episode']} · {ms['avg_upright']:.0%})"
        await self._broadcast(self._stats_snapshot())
        return self.trainers[target].get_model_info()

    async def reset_training(self, mode: str | None = None) -> dict:
        target = self._resolve_mode(mode)
        async with self._lock:
            self.trainers[target].reset_weights()
            self.mode_states[target] = _new_mode_state(target)
            ms = self.mode_states[target]
            ms["last_save_time"] = time.time()
            ms["last_milestone"] = f"Reset {target} weights from scratch ({target}_best.pt untouched)"
            if target == self.active_mode:
                self._sync_env_bests()
                self.env.reset()
                self.env.consume_gpu_stats()
                self._s_obs.copy_(self.env.get_obs())
                self._rollout_reset_req = True
        await self._broadcast(self._stats_snapshot())
        return self._stats_snapshot()

    async def apply_control(self, data: dict) -> dict:
        if "paused" in data:
            self.paused = bool(data["paused"])
        if "speed_mult" in data:
            self.speed_mult = float(max(0.25, min(6.0, float(data["speed_mult"]))))
        if "play_mode" in data and data["play_mode"] in ("train", "eval"):
            self.play_mode = data["play_mode"]

        req_mode = data.get("mode") or data.get("curriculum_mode")
        if req_mode in MODES and req_mode != self.active_mode:
            async with self._lock:
                self.active_mode = req_mode
                self.env.set_stage(1 if req_mode == "walk" else 0)
                self._sync_env_bests()
                self.env.reset()
                self.env.consume_gpu_stats()
                self._s_obs.copy_(self.env.get_obs())
                self._rollout_reset_req = True
        await self._broadcast(self._stats_snapshot())
        return self._stats_snapshot()

    async def nudge(self, direction: str = "random") -> dict:
        impulses = {
            "front": (-0.95, 0.0, "Front push — bending knees & stepping back cleanly!"),
            "back": (1.15, 0.0, "Back push — catching forward momentum cleanly!"),
            "left": (0.0, -0.95, "Left push — abducting hips/legs to balance!"),
            "right": (0.0, 0.95, "Right push — abducting hips/legs to balance!"),
            "random": (0.95, 0.85, "Applied directional nudge impulse!"),
        }
        dvx, dvz, note = impulses.get(direction, impulses["random"])
        async with self._lock:
            self.env.apply_impulse(dvx=dvx, dvz=dvz, env_idx=0)
            self._s_obs.copy_(self.env.get_obs())
            self._ms["last_milestone"] = note
        await self._broadcast(self._stats_snapshot())
        return {"ok": True, "dir": direction}

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def _step_once_gpu(self, is_eval: bool) -> None:
        model = self.active_trainer.model
        if not is_eval:
            model.obs_norm.update(self._s_obs)
        with torch.no_grad():
            act, logp, val = model.act(self._s_obs, deterministic=is_eval)
        next_obs, rew, done = self.env.step_gpu(act, dt_ctrl=0.02, substeps=4)
        self._s_act.copy_(act)
        self._s_logp.copy_(logp)
        self._s_val.copy_(val)
        self._s_rew.copy_(rew)
        self._s_done.copy_(done.float())
        self._s_next_obs.copy_(next_obs)

    async def _run_loop(self) -> None:
        T = self.rollout_steps
        N = self.num_envs
        obs_buf = torch.zeros(T, N, self.env.OBS_DIM, device=self.device)
        act_buf = torch.zeros(T, N, self.env.ACT_DIM, device=self.device)
        logp_buf = torch.zeros(T, N, device=self.device)
        rew_buf = torch.zeros(T, N, device=self.device)
        done_buf = torch.zeros(T, N, device=self.device)
        val_buf = torch.zeros(T, N, device=self.device)

        step_idx = 0

        while True:
            if self.paused:
                await asyncio.sleep(0.08)
                continue

            tick_start = time.perf_counter()
            ticks_this_frame = 2 if self.speed_mult >= 3.5 else 1
            base_interval = (1.0 / 28.0) / (self.speed_mult / ticks_this_frame)

            async with self._lock:
                if self._rollout_reset_req:
                    step_idx = 0
                    self._rollout_reset_req = False

                trainer = self.active_trainer
                ms = self._ms
                for _ in range(ticks_this_frame):
                    is_eval = self.play_mode == "eval"
                    self._step_once_gpu(is_eval)
                    ms["total_steps"] += N if not is_eval else 1

                    if not is_eval:
                        obs_buf[step_idx].copy_(self._s_obs)
                        act_buf[step_idx].copy_(self._s_act)
                        logp_buf[step_idx].copy_(self._s_logp)
                        rew_buf[step_idx].copy_(self._s_rew)
                        done_buf[step_idx].copy_(self._s_done)
                        val_buf[step_idx].copy_(self._s_val)
                        step_idx += 1

                    self._s_obs.copy_(self._s_next_obs)

                    if not is_eval and step_idx >= T:
                        with torch.no_grad():
                            _, _, last_val = trainer.model.act(self._s_obs)
                        ms["last_losses"] = trainer.update(
                            obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf, last_val
                        )
                        ms["updates"] += 1
                        step_idx = 0

                        gstats = self.env.consume_gpu_stats()
                        if gstats["count"] > 0:
                            ms["episode"] += gstats["count"]
                            if ms["avg_length"] <= 0.0 and gstats["count"] >= max(8, N // 4):
                                a = 1.0
                            else:
                                a = min(0.35, max(0.05, gstats["count"] / float(N)))
                            ms["avg_reward"] = (1 - a) * ms["avg_reward"] + a * gstats["avg_reward"]
                            ms["avg_upright"] = (1 - a) * ms["avg_upright"] + a * gstats["avg_upright"]
                            ms["avg_length"] = (1 - a) * ms["avg_length"] + a * gstats["avg_length"]
                            ms["avg_distance"] = (1 - a) * ms["avg_distance"] + a * gstats["avg_distance"]
                        ms["best_reward"] = max(ms["best_reward"], float(gstats["best_reward"]))
                        ms["best_distance"] = max(ms["best_distance"], float(gstats["best_distance"]))

                        ms["history"].append(
                            {
                                "u": ms["updates"],
                                "ep": ms["episode"],
                                "r": round(ms["avg_reward"], 2),
                                "up": round(ms["avg_upright"], 3),
                                "d": round(ms["avg_distance"], 2),
                                "l": round(ms["avg_length"], 1),
                                "st": 1 if self.active_mode == "walk" else 0,
                            }
                        )

                        if ms["updates"] >= 20 and (
                            ms["updates"] % 20 == 0 or (time.time() - ms["last_save_time"] > 18.0)
                        ):
                            allow_best = ms["avg_length"] >= 160.0
                            self.save_now(
                                mode=self.active_mode,
                                export_onnx=(ms["updates"] % 40 == 0),
                                allow_best=allow_best,
                            )

                        await self._broadcast(self._stats_snapshot())

                hero = self.env.get_hero_snapshot()

            hero["type"] = "frame"
            hero["stage"] = 1 if self.active_mode == "walk" else 0
            hero["active_mode"] = self.active_mode
            hero["best_distance"] = round(self._ms["best_distance"], 2)
            await self._broadcast(hero)

            elapsed = time.perf_counter() - tick_start
            sleep_for = max(0.004, base_interval - elapsed)
            await asyncio.sleep(sleep_for)
