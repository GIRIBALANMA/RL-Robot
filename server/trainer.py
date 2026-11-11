"""GPU-first async PPO training loop using CUDA Graphs for minimal CPU overhead."""

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
    """Runs GPU-resident physics + PPO via CUDA Graphs and streams live frames."""

    def __init__(
        self,
        models_dir: Path | str = "models",
        num_envs: int = 64,
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
        self.speed_mult = 1.0
        self.curriculum_mode = "auto"  # "auto" | "stand" | "walk"
        self.play_mode = "train"  # "train" | "eval"

        # Metrics & history
        self.episode = 0
        self.total_steps = 0
        self.updates = 0
        self.best_reward = -1e9
        self.best_distance = 0.0
        self.avg_reward = 0.0
        self.avg_upright = 0.0
        self.avg_length = 0.0
        self.avg_distance = 0.0
        self.last_losses = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        self.last_milestone = "Initializing GPU CUDA-Graph physics..."
        self._last_save_time = 0.0
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
            self.env.best_rew_t.fill_(self.best_reward)
            self.env.best_dist_t.fill_(self.best_distance)
            stage = int(loaded.get("stage", 0))
            self.env.set_stage(stage)
            self.last_milestone = f"Loaded saved model (Ep {self.episode}, Stage {'Walk' if stage else 'Stand'})"

        # Pre-allocate static GPU buffers and capture CUDA Graphs for zero-CPU-loop stepping
        self._s_obs = torch.zeros(num_envs, self.env.OBS_DIM, device=self.device)
        self._s_act = torch.zeros(num_envs, self.env.ACT_DIM, device=self.device)
        self._s_logp = torch.zeros(num_envs, device=self.device)
        self._s_val = torch.zeros(num_envs, device=self.device)
        self._s_rew = torch.zeros(num_envs, device=self.device)
        self._s_done = torch.zeros(num_envs, device=self.device)
        self._s_next_obs = torch.zeros(num_envs, self.env.OBS_DIM, device=self.device)
        self._train_w = torch.ones(1, device=self.device)

        self._step_graph: torch.cuda.CUDAGraph | None = None
        if self.device.type == "cuda":
            self._capture_cuda_graph()
        else:
            self._s_obs.copy_(self.env.get_obs())

    def _capture_cuda_graph(self) -> None:
        """Capture one unified CUDA Graph for train/eval stepping (~2ms GPU, near-zero CPU)."""
        saved_mean = self.ppo.model.obs_norm.mean.clone()
        saved_var = self.ppo.model.obs_norm.var.clone()
        saved_count = self.ppo.model.obs_norm.count.clone()
        saved_stage = self.env.stage

        def unified_step():
            self.ppo.model.obs_norm.update(self._s_obs, active_w=self._train_w)
            with torch.no_grad():
                act, logp, val = self.ppo.model.act(self._s_obs, train_w=self._train_w)
            next_obs, rew, done = self.env.step_gpu(act, dt_ctrl=1.0 / 45.0, substeps=4)
            self._s_act.copy_(act)
            self._s_logp.copy_(logp)
            self._s_val.copy_(val)
            self._s_rew.copy_(rew)
            self._s_done.copy_(done.float())
            self._s_next_obs.copy_(next_obs)

        self._s_obs.copy_(self.env.get_obs())
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                unified_step()
                self._s_obs.copy_(self._s_next_obs)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        self._step_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._step_graph):
            unified_step()
        torch.cuda.synchronize()

        # Restore clean initial state & obs_norm after warmup/capture
        self.ppo.model.obs_norm.mean.copy_(saved_mean)
        self.ppo.model.obs_norm.var.copy_(saved_var)
        self.ppo.model.obs_norm.count.copy_(saved_count)
        self.env.set_stage(saved_stage)
        self.env.reset()
        self.env.consume_gpu_stats()
        self.env.best_rew_t.fill_(self.best_reward)
        self.env.best_dist_t.fill_(self.best_distance)
        self._s_obs.copy_(self.env.get_obs())

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
            "avg_reward": round(self.avg_reward, 2),
            "best_reward": round(self.best_reward if self.best_reward > -1e8 else 0.0, 2),
            "best_distance": round(self.best_distance, 2),
            "avg_upright": round(self.avg_upright, 3),
            "avg_length": round(self.avg_length, 1),
            "avg_distance": round(self.avg_distance, 2),
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
            self.env.best_rew_t.fill_(self.best_reward)
            self.env.best_dist_t.fill_(self.best_distance)
            stage = int(loaded.get("stage", 0))
            self.env.set_stage(stage)
            self.env.reset()
            self._s_obs.copy_(self.env.get_obs())
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
            self.avg_reward = 0.0
            self.avg_upright = 0.0
            self.avg_length = 0.0
            self.avg_distance = 0.0
            self.history.clear()
            self.last_losses = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
            self.env.set_stage(1 if self.curriculum_mode == "walk" else 0)
            self.env.reset()
            self.env.consume_gpu_stats()
            self.env.best_rew_t.fill_(-1e9)
            self.env.best_dist_t.fill_(0.0)
            self._s_obs.copy_(self.env.get_obs())
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
            self._train_w.fill_(0.0 if self.play_mode == "eval" else 1.0)
        if "curriculum_mode" in data and data["curriculum_mode"] in ("auto", "stand", "walk"):
            self.curriculum_mode = data["curriculum_mode"]
            if self.curriculum_mode == "stand":
                self.env.set_stage(0)
                self.last_milestone = "Curriculum locked to Stage 1: Stand & Arm Balance"
            elif self.curriculum_mode == "walk":
                self.env.set_stage(1)
                self.last_milestone = "Curriculum set to Stage 2: Walk & Arm Swing"
            else:
                self.last_milestone = "Auto-Curriculum enabled (Stand -> Walk)"
        await self._broadcast(self._stats_snapshot())
        return self._stats_snapshot()

    async def nudge(self) -> dict:
        async with self._lock:
            self.env.apply_impulse(dvx=1.4, dvz=0.9, env_idx=0)
            self._s_obs.copy_(self.env.get_obs())
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

    def _step_once_gpu(self, is_eval: bool) -> None:
        if self._step_graph is not None:
            self._step_graph.replay()
        else:
            if not is_eval:
                self.ppo.model.obs_norm.update(self._s_obs)
            with torch.no_grad():
                act, logp, val = self.ppo.model.act(self._s_obs, deterministic=is_eval)
            next_obs, rew, done = self.env.step_gpu(act, dt_ctrl=1.0 / 45.0, substeps=4)
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
                for _ in range(ticks_this_frame):
                    is_eval = self.play_mode == "eval"
                    self._step_once_gpu(is_eval)
                    self.total_steps += N if not is_eval else 1

                    if not is_eval:
                        obs_buf[step_idx].copy_(self._s_obs)
                        act_buf[step_idx].copy_(self._s_act)
                        logp_buf[step_idx].copy_(self._s_logp)
                        rew_buf[step_idx].copy_(self._s_rew)
                        done_buf[step_idx].copy_(self._s_done)
                        val_buf[step_idx].copy_(self._s_val)
                        step_idx += 1

                    self._s_obs.copy_(self._s_next_obs)

                    # Once per rollout window: PPO update on GPU + 1 batched GPU stats read
                    if not is_eval and step_idx >= T:
                        with torch.no_grad():
                            _, _, last_val = self.ppo.model.act(self._s_obs)
                        self.last_losses = self.ppo.update(
                            obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf, last_val
                        )
                        self.updates += 1
                        step_idx = 0

                        gstats = self.env.consume_gpu_stats()
                        if gstats["count"] > 0:
                            self.episode += gstats["count"]
                            a = 0.35 if self.updates > 1 else 1.0
                            self.avg_reward = (1 - a) * self.avg_reward + a * gstats["avg_reward"]
                            self.avg_upright = (1 - a) * self.avg_upright + a * gstats["avg_upright"]
                            self.avg_length = (1 - a) * self.avg_length + a * gstats["avg_length"]
                            self.avg_distance = (1 - a) * self.avg_distance + a * gstats["avg_distance"]
                        self.best_reward = max(self.best_reward, float(gstats["best_reward"]))
                        self.best_distance = max(self.best_distance, float(gstats["best_distance"]))

                        self.history.append(
                            {
                                "u": self.updates,
                                "ep": self.episode,
                                "r": round(self.avg_reward, 2),
                                "up": round(self.avg_upright, 3),
                                "d": round(self.avg_distance, 2),
                                "l": round(self.avg_length, 1),
                                "st": self.env.stage,
                            }
                        )

                        if (
                            self.curriculum_mode == "auto"
                            and self.env.stage == 0
                            and self.updates >= 26
                            and self.avg_upright >= 0.72
                            and self.avg_length >= 70.0
                        ):
                            self.env.set_stage(1)
                            self.save_now(export_onnx=True)
                            self.last_milestone = (
                                f"Milestone: Standing balanced ({self.avg_upright:.0%}) — graduating to Walk!"
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
