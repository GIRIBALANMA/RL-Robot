"""Paced async GPU PPO training loop with curriculum promotion, telemetry, and model saving."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from pathlib import Path
from typing import Callable, Awaitable

import torch

from .physics import BipedPhysicsEnv
from .ppo import PPOTrainer


class LearningManager:
    """Runs paced GPU physics + PPO updates and broadcasts live frames & telemetry."""

    def __init__(
        self,
        models_dir: Path | str = "models",
        num_envs: int = 32,
        rollout_steps: int = 32,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = BipedPhysicsEnv(num_envs=num_envs, device=self.device)
        self.ppo = PPOTrainer(
            obs_dim=self.env.OBS_DIM,
            act_dim=self.env.ACT_DIM,
            device=self.device,
            models_dir=models_dir,
        )
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps

        # Runtime controls
        self.paused = False
        self.speed_mult = 1.0  # 0.5, 1.0, 2.0, 4.0
        self.curriculum_mode = "auto"  # "auto" | "stand" | "walk"
        self.play_mode = "train"  # "train" | "eval"

        # Metrics & history
        self.episode = 0
        self.total_steps = 0
        self.updates = 0
        self.best_reward = -1e9
        self.best_distance = 0.0
        self.last_losses = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        self.last_milestone = "Initializing GPU physics & neural policy..."
        self._last_save_time = 0.0

        self.recent_rewards: deque[float] = deque(maxlen=24)
        self.recent_uprights: deque[float] = deque(maxlen=24)
        self.recent_lengths: deque[int] = deque(maxlen=24)
        self.recent_distances: deque[float] = deque(maxlen=24)
        self.history: deque[dict] = deque(maxlen=160)

        self._subscribers: set[Callable[[dict], Awaitable[None]]] = set()
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

        # Restore prior checkpoint if already saved on disk
        loaded = self.ppo.load()
        if loaded:
            self.episode = int(loaded.get("episode", 0))
            self.total_steps = int(loaded.get("total_steps", 0))
            self.updates = int(loaded.get("updates", 0))
            self.best_reward = float(loaded.get("best_reward", -1e9))
            self.best_distance = float(loaded.get("best_distance", 0.0))
            stage = int(loaded.get("stage", 0))
            self.env.set_stage(stage)
            self.last_milestone = f"Loaded saved model (Ep {self.episode}, Stage {'Walk' if stage else 'Stand'})"

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
        avg_rew = sum(self.recent_rewards) / len(self.recent_rewards) if self.recent_rewards else 0.0
        avg_up = sum(self.recent_uprights) / len(self.recent_uprights) if self.recent_uprights else 0.0
        avg_len = sum(self.recent_lengths) / len(self.recent_lengths) if self.recent_lengths else 0.0
        avg_dist = sum(self.recent_distances) / len(self.recent_distances) if self.recent_distances else 0.0
        vram_mb = (
            round(torch.cuda.memory_allocated(self.device) / (1024 * 1024), 1)
            if self.device.type == "cuda"
            else 0.0
        )
        return {
            "type": "telemetry",
            "device": torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU",
            "vram_mb": vram_mb,
            "paused": self.paused,
            "speed_mult": self.speed_mult,
            "curriculum_mode": self.curriculum_mode,
            "play_mode": self.play_mode,
            "stage": self.env.stage,
            "episode": self.episode,
            "updates": self.updates,
            "total_steps": self.total_steps,
            "avg_reward": round(avg_rew, 2),
            "best_reward": round(self.best_reward if self.best_reward > -1e8 else 0.0, 2),
            "best_distance": round(self.best_distance, 2),
            "avg_upright": round(avg_up, 3),
            "avg_length": round(avg_len, 1),
            "avg_distance": round(avg_dist, 2),
            **self.last_losses,
            "milestone": self.last_milestone,
            "model_info": self.ppo.get_model_info(),
            "history": list(self.history),
        }

    def get_init_payload(self) -> dict:
        payload = self._stats_snapshot()
        payload["type"] = "init"
        payload["hero"] = self.env.get_hero_snapshot()
        return payload

    def save_now(self, export_onnx: bool = True) -> dict:
        stats = {
            "episode": self.episode,
            "updates": self.updates,
            "total_steps": self.total_steps,
            "stage": self.env.stage,
            "best_reward": self.best_reward if self.best_reward > -1e8 else 0.0,
            "best_distance": self.best_distance,
        }
        meta = self.ppo.save(stats, export_onnx=export_onnx)
        self._last_save_time = time.time()
        self.last_milestone = f"Saved model checkpoint (Ep {self.episode}, {meta['pt_bytes'] // 1024} KB)"
        return meta

    async def load_saved(self) -> dict | None:
        async with self._lock:
            loaded = self.ppo.load()
            if not loaded:
                return None
            self.episode = int(loaded.get("episode", 0))
            self.updates = int(loaded.get("updates", 0))
            self.total_steps = int(loaded.get("total_steps", 0))
            self.best_reward = float(loaded.get("best_reward", -1e9))
            self.best_distance = float(loaded.get("best_distance", 0.0))
            stage = int(loaded.get("stage", 0))
            self.env.set_stage(stage)
            self.env.reset()
            self.last_milestone = f"Restored saved model (Ep {self.episode}, Stage {'Walk' if stage else 'Stand'})"
        await self._broadcast(self._stats_snapshot())
        return self.ppo.get_model_info()

    async def reset_training(self) -> dict:
        async with self._lock:
            self.ppo.reset_weights()
            self.episode = 0
            self.updates = 0
            self.total_steps = 0
            self.best_reward = -1e9
            self.best_distance = 0.0
            self.recent_rewards.clear()
            self.recent_uprights.clear()
            self.recent_lengths.clear()
            self.recent_distances.clear()
            self.history.clear()
            self.last_losses = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
            if self.curriculum_mode == "walk":
                self.env.set_stage(1)
            else:
                self.env.set_stage(0)
            self.env.reset()
            self.last_milestone = "Reset policy weights — learning from scratch"
        await self._broadcast(self._stats_snapshot())
        return self._stats_snapshot()

    async def apply_control(self, data: dict) -> dict:
        if "paused" in data:
            self.paused = bool(data["paused"])
        if "speed_mult" in data:
            self.speed_mult = float(max(0.25, min(6.0, float(data["speed_mult"]))))
        if "play_mode" in data and data["play_mode"] in ("train", "eval"):
            self.play_mode = data["play_mode"]
        if "curriculum_mode" in data and data["curriculum_mode"] in ("auto", "stand", "walk"):
            self.curriculum_mode = data["curriculum_mode"]
            if self.curriculum_mode == "stand":
                self.env.set_stage(0)
                self.last_milestone = "Curriculum locked to Stage 1: Stand & Balance"
            elif self.curriculum_mode == "walk":
                self.env.set_stage(1)
                self.last_milestone = "Curriculum set to Stage 2: Stride & Walk"
            else:
                self.last_milestone = "Auto-Curriculum enabled (Stand -> Walk)"
        await self._broadcast(self._stats_snapshot())
        return self._stats_snapshot()

    async def nudge(self) -> dict:
        async with self._lock:
            self.env.apply_impulse(dvx=1.4, dvz=0.9, env_idx=0)
            self.last_milestone = "Applied lateral/forward nudge impulse!"
        await self._broadcast(self._stats_snapshot())
        return {"ok": True}

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

    async def _run_loop(self) -> None:
        T = self.rollout_steps
        N = self.num_envs
        obs_buf = torch.zeros(T, N, self.env.OBS_DIM, device=self.device)
        act_buf = torch.zeros(T, N, self.env.ACT_DIM, device=self.device)
        logp_buf = torch.zeros(T, N, device=self.device)
        rew_buf = torch.zeros(T, N, device=self.device)
        done_buf = torch.zeros(T, N, device=self.device)
        val_buf = torch.zeros(T, N, device=self.device)

        obs = self.env.get_obs()
        step_idx = 0

        while True:
            if self.paused:
                await asyncio.sleep(0.08)
                continue

            tick_start = time.perf_counter()

            # Determine sub-batch of visual ticks when speed_mult > 2, while staying gentle on GPU/CPU
            ticks_this_frame = 2 if self.speed_mult >= 3.5 else 1
            base_interval = (1.0 / 24.0) / (self.speed_mult / ticks_this_frame)

            async with self._lock:
                for _ in range(ticks_this_frame):
                    is_eval = self.play_mode == "eval"
                    if not is_eval:
                        self.ppo.model.obs_norm.update(obs)
                    with torch.no_grad():
                        act, logp, val = self.ppo.model.act(obs, deterministic=is_eval)

                    next_obs, rew, done, completed = self.env.step(act, dt_ctrl=1.0 / 32.0, substeps=4)
                    self.total_steps += N if not is_eval else 1

                    if not is_eval:
                        obs_buf[step_idx] = obs
                        act_buf[step_idx] = act
                        logp_buf[step_idx] = logp
                        rew_buf[step_idx] = rew
                        done_buf[step_idx] = done.float()
                        val_buf[step_idx] = val
                        step_idx += 1

                    obs = next_obs

                    for ep in completed:
                        self.episode += 1
                        self.recent_rewards.append(ep["reward"])
                        self.recent_uprights.append(ep["upright"])
                        self.recent_lengths.append(ep["steps"])
                        self.recent_distances.append(ep["distance"])
                        if ep["reward"] > self.best_reward:
                            self.best_reward = ep["reward"]
                        if self.env.stage == 1 and ep["distance"] > self.best_distance:
                            self.best_distance = ep["distance"]

                    # Perform GPU PPO update at end of rollout window
                    if not is_eval and step_idx >= T:
                        with torch.no_grad():
                            _, _, last_val = self.ppo.model.act(obs)
                        self.last_losses = self.ppo.update(
                            obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf, last_val
                        )
                        self.updates += 1
                        step_idx = 0

                        avg_rew = sum(self.recent_rewards) / max(1, len(self.recent_rewards))
                        avg_up = sum(self.recent_uprights) / max(1, len(self.recent_uprights))
                        avg_len = sum(self.recent_lengths) / max(1, len(self.recent_lengths))
                        avg_dist = sum(self.recent_distances) / max(1, len(self.recent_distances))

                        self.history.append(
                            {
                                "u": self.updates,
                                "ep": self.episode,
                                "r": round(avg_rew, 2),
                                "up": round(avg_up, 3),
                                "d": round(avg_dist, 2),
                                "l": round(avg_len, 1),
                                "st": self.env.stage,
                            }
                        )

                        # Auto-curriculum promotion from Stand (0) -> Walk (1)
                        if (
                            self.curriculum_mode == "auto"
                            and self.env.stage == 0
                            and self.updates >= 22
                            and avg_up >= 0.76
                            and avg_len >= 95.0
                        ):
                            self.env.set_stage(1)
                            self.save_now(export_onnx=True)
                            self.last_milestone = (
                                f"Milestone: Standing balanced ({avg_up:.0%}) — graduating to Walk!"
                            )
                        elif (
                            self.updates % 20 == 0
                            or (time.time() - self._last_save_time > 18.0 and self.updates > 0)
                        ):
                            self.save_now(export_onnx=(self.updates % 40 == 0))

                        await self._broadcast(self._stats_snapshot())

                hero = self.env.get_hero_snapshot()

            hero["type"] = "frame"
            hero["stage"] = self.env.stage
            hero["best_distance"] = round(self.best_distance, 2)
            await self._broadcast(hero)

            elapsed = time.perf_counter() - tick_start
            sleep_for = max(0.004, base_interval - elapsed)
            await asyncio.sleep(sleep_for)
