"""PyTorch CUDA Actor-Critic PPO learner with running observation normalization and .pt/.onnx saving."""

from __future__ import annotations

import copy
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Normal


def _atomic_torch_save(payload: dict, target: Path) -> None:
    """Atomically write a PyTorch checkpoint via a same-directory temporary file."""
    tmp = target.with_name(f"{target.stem}.{os.getpid()}.tmp{target.suffix}")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _atomic_write_text(text: str, target: Path) -> None:
    """Atomically write UTF-8 text via a same-directory temporary file."""
    tmp = target.with_name(f"{target.stem}.{os.getpid()}.tmp{target.suffix}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


class RunningNorm(nn.Module):
    """Online Welford mean/variance normalizer stored inside saved checkpoints."""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x: torch.Tensor, active_w: torch.Tensor | None = None) -> None:
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = float(x.shape[0])

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * (self.count * batch_count / tot_count)
        new_var = torch.clamp(M2 / tot_count, min=1e-4)

        if active_w is not None:
            w = active_w[0]
            self.mean.copy_(w * new_mean + (1.0 - w) * self.mean)
            self.var.copy_(w * new_var + (1.0 - w) * self.var)
            self.count.copy_(w * tot_count + (1.0 - w) * self.count)
        else:
            self.mean.copy_(new_mean)
            self.var.copy_(new_var)
            self.count.copy_(tot_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-6), -8.0, 8.0)


class ActorCritic(nn.Module):
    """GPU Actor-Critic MLP for 49-dim humanoid state -> 17-dim joint targets."""

    def __init__(self, obs_dim: int = 49, act_dim: int = 17):
        super().__init__()
        self.obs_norm = RunningNorm(obs_dim)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, act_dim),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        # Standard exploration std (~0.43) for pure RL from scratch
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.85))
        self.init_weights()

    @torch.no_grad()
    def init_weights(self) -> None:
        """Initialize or reset network weights cleanly near neutral posture."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-2].weight, gain=0.02)
        self.log_std.fill_(-0.85)
        self.obs_norm.mean.zero_()
        self.obs_norm.var.fill_(1.0)
        self.obs_norm.count.fill_(1e-4)

    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        train_w: torch.Tensor | None = None,
    ):
        x = self.obs_norm(obs)
        mean = self.actor(x)
        value = self.critic(x).squeeze(-1)
        if deterministic:
            return mean, torch.zeros(obs.shape[0], device=obs.device), value
        log_std = torch.clamp(self.log_std, -2.3, -0.35)
        std = torch.exp(log_std)
        raw_act = mean + std * torch.randn_like(mean)
        log_prob = -0.5 * (
            ((raw_act - mean) / std) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi)
        ).sum(dim=-1)
        if train_w is not None:
            w = train_w[0]
            raw_act = w * raw_act + (1.0 - w) * mean
        return raw_act, log_prob, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        x = self.obs_norm(obs)
        mean = self.actor(x)
        value = self.critic(x).squeeze(-1)
        std = torch.exp(torch.clamp(self.log_std, -2.3, -0.35))
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


class PolicyInferenceWrapper(nn.Module):
    """Exportable deterministic policy module for ONNX export."""

    def __init__(self, ac: ActorCritic):
        super().__init__()
        self.obs_norm = ac.obs_norm
        self.actor = ac.actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.obs_norm(obs))


class PPOTrainer:
    """Lightweight GPU PPO updater + isolated per-mode persistence in models/."""

    def __init__(
        self,
        obs_dim: int = 49,
        act_dim: int = 17,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        epochs: int = 5,
        minibatches: int = 6,
        device: torch.device | None = None,
        models_dir: Path | str = "models",
        model_name: str = "stand",
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.model_name = model_name
        self.model = ActorCritic(obs_dim, act_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.minibatches = minibatches
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.pt_path = self.models_dir / f"{model_name}_ppo.pt"
        self.best_pt_path = self.models_dir / f"{model_name}_best.pt"
        self.onnx_path = self.models_dir / f"{model_name}_policy.onnx"
        self.meta_path = self.models_dir / f"{model_name}_meta.json"
        self.update_count = 0
        self._best_score = self._read_best_score()

    def _score_stats(self, stats: dict | None) -> tuple[float, float, float]:
        if not stats or stats.get("arch") != "mujoco_pure_v1":
            return (-1.0, -1e9, -1.0)
        saved_mode = stats.get("model_name")
        if saved_mode is not None and saved_mode != self.model_name:
            return (-1.0, -1e9, -1.0)
        dist = float(stats.get("best_distance", 0.0))
        avg_rew = float(stats.get("avg_reward", 0.0))
        best_rew = float(stats.get("best_reward", 0.0))
        up = float(stats.get("avg_upright", 0.0))
        primary_rew = avg_rew if avg_rew > 0.0 else best_rew
        return (
            round(primary_rew, 2),
            round(up * 100.0 + (min(dist, 30.0) if self.model_name == "walk" else 0.0), 2),
            round(max(best_rew, 0.0), 2),
        )

    def _candidate_pt_paths(self, prefer_best: bool) -> list[Path]:
        """Return unique local checkpoint paths for this isolated mode."""
        ordered = [self.best_pt_path, self.pt_path] if prefer_best else [self.pt_path, self.best_pt_path]
        if prefer_best:
            ordered.extend(
                p for p in sorted(self.models_dir.glob(f"{self.model_name}*.pt"))
                if ".tmp" not in p.stem
            )
        seen: set[Path] = set()
        paths: list[Path] = []
        for path in ordered:
            if path not in seen and path.exists():
                seen.add(path)
                paths.append(path)
        return paths

    def _read_best_score(self) -> tuple[float, float, float]:
        best = (-1.0, -1e9, -1.0)
        for path in self._candidate_pt_paths(prefer_best=True):
            try:
                p = torch.load(path, map_location="cpu", weights_only=False)
                s = self._score_stats(p.get("stats", {}))
                if s > best:
                    best = s
            except Exception:
                pass
        return best

    def update(
        self,
        obs_buf: torch.Tensor,
        act_buf: torch.Tensor,
        logp_buf: torch.Tensor,
        rew_buf: torch.Tensor,
        done_buf: torch.Tensor,
        val_buf: torch.Tensor,
        last_val: torch.Tensor,
    ) -> dict[str, float]:
        """Run PPO update on rollout buffers [T, N, ...]."""
        T, N = rew_buf.shape
        adv_buf = torch.zeros_like(rew_buf)
        last_gae = torch.zeros(N, device=self.device)

        for t in reversed(range(T)):
            next_non_terminal = 1.0 - done_buf[t]
            next_values = last_val if t == T - 1 else val_buf[t + 1]
            delta = rew_buf[t] + self.gamma * next_values * next_non_terminal - val_buf[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            adv_buf[t] = last_gae

        ret_buf = adv_buf + val_buf

        b_obs = obs_buf.reshape(T * N, -1)
        b_act = act_buf.reshape(T * N, -1)
        b_logp = logp_buf.reshape(T * N)
        b_adv = adv_buf.reshape(T * N)
        b_ret = ret_buf.reshape(T * N)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        total = T * N
        batch_size = max(1, total // self.minibatches)
        pg_sum = torch.zeros(1, device=self.device)
        vf_sum = torch.zeros(1, device=self.device)
        ent_sum = torch.zeros(1, device=self.device)
        steps = 0

        actor_params = list(self.model.actor.parameters()) + [self.model.log_std]
        critic_params = list(self.model.critic.parameters())

        for _ in range(self.epochs):
            perm = torch.randperm(total, device=self.device)
            for start in range(0, total, batch_size):
                idx = perm[start : start + batch_size]
                new_logp, entropy, new_val = self.model.evaluate(b_obs[idx], b_act[idx])
                ratio = torch.exp(new_logp - b_logp[idx])
                surr1 = ratio * b_adv[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv[idx]
                pg_loss = -torch.min(surr1, surr2).mean()
                vf_loss = 0.5 * nn.functional.smooth_l1_loss(new_val, b_ret[idx], beta=10.0)
                ent = entropy.mean()
                loss = pg_loss + 0.5 * vf_loss - 0.004 * ent

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(actor_params, 0.5)
                nn.utils.clip_grad_norm_(critic_params, 1.0)
                self.optimizer.step()

                pg_sum = pg_sum + pg_loss.detach()
                vf_sum = vf_sum + vf_loss.detach()
                ent_sum = ent_sum + ent.detach()
                steps += 1

        vals = torch.cat([pg_sum, vf_sum, ent_sum]).tolist()
        self.update_count += 1
        # Smoothly anneal pure RL exploration std down from ~0.43 toward ~0.19 over 450 updates
        frac = min(1.0, self.update_count / 450.0)
        target_max = -0.85 - 0.80 * frac
        with torch.no_grad():
            cur_max = float(self.model.log_std.max().item())
            next_max = max(target_max, cur_max - 0.015)
            self.model.log_std.clamp_(-2.3, next_max)
        denom = max(1, steps)
        return {
            "policy_loss": round(vals[0] / denom, 4),
            "value_loss": round(vals[1] / denom, 4),
            "entropy": round(vals[2] / denom, 4),
        }

    def _build_meta(self, stats: dict, saved_at: str, onnx_ok: bool, source: str = "latest") -> dict:
        has_onnx = self.onnx_path.exists()
        return {
            "model_name": self.model_name,
            "pt_file": self.pt_path.name,
            "loaded_file": str(stats.get("_loaded_file", self.pt_path.name)),
            "pt_bytes": self.pt_path.stat().st_size if self.pt_path.exists() else 0,
            "onnx_file": self.onnx_path.name if has_onnx else None,
            "onnx_bytes": self.onnx_path.stat().st_size if has_onnx else 0,
            "saved_at": saved_at,
            "episode": stats.get("episode", 0),
            "updates": stats.get("updates", 0),
            "total_steps": stats.get("total_steps", 0),
            "stage": stats.get("stage", 0),
            "avg_reward": round(float(stats.get("avg_reward", 0.0)), 2),
            "avg_upright": round(float(stats.get("avg_upright", 0.0)), 3),
            "best_reward": round(float(stats.get("best_reward", 0.0)), 2),
            "best_distance": round(float(stats.get("best_distance", 0.0)), 2),
            "source": source,
            "has_best": self.best_pt_path.exists(),
            "device": str(self.device),
        }

    def _export_onnx_atomic(self) -> bool:
        """Export current policy + observation normalizer to ONNX atomically."""
        tmp = self.onnx_path.with_name(f"{self.onnx_path.stem}.{os.getpid()}.tmp.onnx")
        try:
            cpu_model = copy.deepcopy(self.model).eval().to("cpu")
            wrapper = PolicyInferenceWrapper(cpu_model)
            dummy = torch.zeros(1, self.obs_dim, dtype=torch.float32)
            torch.onnx.export(
                wrapper,
                dummy,
                str(tmp),
                input_names=["obs"],
                output_names=["joint_targets"],
                opset_version=14,
                dynamo=False,
            )
            os.replace(tmp, self.onnx_path)
            return True
        except Exception:
            tmp.unlink(missing_ok=True)
            return False

    def save(self, stats: dict, export_onnx: bool = True, allow_best: bool = True) -> dict:
        """Save isolated PyTorch .pt checkpoint, protected best .pt snapshot, and ONNX graph to models/."""
        clean_stats = {k: v for k, v in stats.items() if not str(k).startswith("_")}
        clean_stats["arch"] = "mujoco_pure_v1"
        clean_stats["model_name"] = self.model_name
        cpu_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        payload = {
            "model_state": cpu_state,
            "optimizer_state": self.optimizer.state_dict(),
            "stats": clean_stats,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _atomic_torch_save(payload, self.pt_path)

        score = self._score_stats(clean_stats)
        if not self.best_pt_path.exists():
            _atomic_torch_save(payload, self.best_pt_path)
            self._best_score = max(self._best_score, score)
        elif allow_best and score >= self._best_score:
            _atomic_torch_save(payload, self.best_pt_path)
            self._best_score = score

        onnx_ok = self._export_onnx_atomic() if export_onnx else self.onnx_path.exists()
        meta = self._build_meta(clean_stats, payload["saved_at"], onnx_ok)
        _atomic_write_text(json.dumps(meta, indent=2), self.meta_path)
        return meta

    def load(self, prefer_best: bool = False) -> dict | None:
        """Load saved PyTorch .pt checkpoint, choosing the highest-scoring best snapshot locally when requested."""
        candidates = []
        for path in self._candidate_pt_paths(prefer_best=prefer_best):
            try:
                payload = torch.load(path, map_location=self.device, weights_only=False)
                raw_stats = payload.get("stats", {})
                if raw_stats.get("arch") != "mujoco_pure_v1":
                    continue
                saved_mode = raw_stats.get("model_name")
                if saved_mode is not None and saved_mode != self.model_name:
                    continue
                score = self._score_stats(raw_stats)
                is_best_file = 1 if path == self.best_pt_path else 0
                updates = int(raw_stats.get("updates", 0))
                candidates.append((score, is_best_file, updates, path, payload))
            except Exception:
                continue

        if not candidates:
            return None

        if prefer_best:
            candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        chosen_score, _, _, chosen_path, payload = candidates[0]

        self.model.load_state_dict(payload["model_state"])
        if "optimizer_state" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state"])
        stats = dict(payload.get("stats", {}))
        self.update_count = int(stats.get("updates", 0))

        best_file_score = next(
            (item[0] for item in candidates if item[3] == self.best_pt_path),
            (-1.0, -1e9, -1.0),
        )
        if not self.best_pt_path.exists() or chosen_score > best_file_score:
            _atomic_torch_save(payload, self.best_pt_path)
            best_file_score = chosen_score
        self._best_score = max(self._best_score, best_file_score, chosen_score)

        is_best_choice = chosen_path == self.best_pt_path or (prefer_best and chosen_score >= self._best_score)
        stats["_loaded_file"] = chosen_path.name
        stats["_loaded_source"] = "best" if is_best_choice else "latest"

        if prefer_best:
            if chosen_path != self.pt_path:
                _atomic_torch_save(payload, self.pt_path)
            onnx_ok = self.onnx_path.exists() or self._export_onnx_atomic()
            meta = self._build_meta(
                stats,
                payload.get("saved_at", time.strftime("%Y-%m-%d %H:%M:%S")),
                onnx_ok,
                source=stats["_loaded_source"],
            )
            _atomic_write_text(json.dumps(meta, indent=2), self.meta_path)
        return stats

    def reset_weights(self) -> None:
        """Re-initialize fresh untrained neural network weights in-place."""
        self.model.init_weights()
        self.optimizer.state.clear()
        self.update_count = 0

    def get_model_info(self) -> dict | None:
        if self.meta_path.exists():
            try:
                return json.loads(self.meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        if self.pt_path.exists():
            return {
                "pt_file": self.pt_path.name,
                "pt_bytes": self.pt_path.stat().st_size,
                "onnx_file": self.onnx_path.name if self.onnx_path.exists() else None,
                "onnx_bytes": self.onnx_path.stat().st_size if self.onnx_path.exists() else 0,
            }
        return None
