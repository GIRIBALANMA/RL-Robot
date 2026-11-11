"""PyTorch CUDA Actor-Critic PPO learner with running observation normalization and .pt/.onnx saving."""

from __future__ import annotations

import copy
import json
import math
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Normal


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
    """Compact GPU Actor-Critic MLP for 45-dim biped state -> 15-dim joint targets."""

    def __init__(self, obs_dim: int = 49, act_dim: int = 17):
        super().__init__()
        self.obs_norm = RunningNorm(obs_dim)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, act_dim),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        # Start with moderate exploration std (~0.32) for smooth early probing
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.15))
        self.init_weights()

    @torch.no_grad()
    def init_weights(self) -> None:
        """Initialize or reset weights in-place so CUDA Graph tensor pointers stay fixed."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-2].weight, gain=0.05)
        self.log_std.fill_(-1.15)
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
        log_std = torch.clamp(self.log_std, -2.8, -1.05)
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
        std = torch.exp(torch.clamp(self.log_std, -2.8, -1.05))
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
    """Lightweight GPU PPO updater + model persistence in models/."""

    def __init__(
        self,
        obs_dim: int = 49,
        act_dim: int = 17,
        lr: float = 5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        epochs: int = 4,
        minibatches: int = 4,
        device: torch.device | None = None,
        models_dir: Path | str = "models",
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.model = ActorCritic(obs_dim, act_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.minibatches = minibatches
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.pt_path = self.models_dir / "steel_sentinel_ppo.pt"
        self.best_pt_path = self.models_dir / "steel_sentinel_best.pt"
        self.onnx_path = self.models_dir / "steel_sentinel_policy.onnx"
        self.meta_path = self.models_dir / "checkpoints.json"
        self.update_count = 0
        self._best_score = self._read_best_score()

    @staticmethod
    def _score_stats(stats: dict | None) -> tuple[float, float, float]:
        if not stats:
            return (-1.0, -1e9, -1.0)
        stage = int(stats.get("stage", 0))
        dist = float(stats.get("best_distance", 0.0))
        avg_rew = float(stats.get("avg_reward", 0.0))
        best_rew = float(stats.get("best_reward", 0.0))
        up = float(stats.get("avg_upright", 0.0))
        primary_rew = avg_rew if avg_rew > 0.0 else best_rew
        return (
            round(dist, 2) if stage == 1 and dist >= 3.0 else 0.0,
            round(primary_rew, 2),
            round(up * 100.0 + max(best_rew, 0.0) * 0.01, 2),
        )

    def _read_best_score(self) -> tuple[float, float, float]:
        best = (-1.0, -1e9, -1.0)
        for path in (self.best_pt_path, self.pt_path):
            if path.exists():
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
                loss = pg_loss + 0.5 * vf_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Clip actor and critic separately so large critic value scales never choke actor/log_std learning
                nn.utils.clip_grad_norm_(actor_params, 0.5)
                nn.utils.clip_grad_norm_(critic_params, 1.0)
                self.optimizer.step()

                pg_sum = pg_sum + pg_loss.detach()
                vf_sum = vf_sum + vf_loss.detach()
                ent_sum = ent_sum + ent.detach()
                steps += 1

        vals = torch.cat([pg_sum, vf_sum, ent_sum]).tolist()
        self.update_count += 1
        # Smoothly lower exploration ceiling toward log_std=-1.85 (std~0.16) without shocking existing weights
        frac = min(1.0, self.update_count / 180.0)
        target_max = -1.05 - 0.80 * frac
        with torch.no_grad():
            cur_max = float(self.model.log_std.max().item())
            next_max = max(target_max, cur_max - 0.015)
            self.model.log_std.clamp_(-2.8, next_max)
        denom = max(1, steps)
        return {
            "policy_loss": round(vals[0] / denom, 4),
            "value_loss": round(vals[1] / denom, 4),
            "entropy": round(vals[2] / denom, 4),
        }

    def _build_meta(self, stats: dict, saved_at: str, onnx_ok: bool, source: str = "latest") -> dict:
        return {
            "pt_file": self.pt_path.name,
            "pt_bytes": self.pt_path.stat().st_size if self.pt_path.exists() else 0,
            "onnx_file": self.onnx_path.name if onnx_ok else None,
            "onnx_bytes": self.onnx_path.stat().st_size if onnx_ok and self.onnx_path.exists() else 0,
            "saved_at": saved_at,
            "episode": stats.get("episode", 0),
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

    def save(self, stats: dict, export_onnx: bool = True, allow_best: bool = True) -> dict:
        """Save real PyTorch .pt checkpoint, protected best .pt snapshot, and ONNX graph to models/."""
        cpu_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        payload = {
            "model_state": cpu_state,
            "optimizer_state": self.optimizer.state_dict(),
            "stats": stats,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        torch.save(payload, self.pt_path)

        score = self._score_stats(stats)
        if not self.best_pt_path.exists() or (allow_best and score >= self._best_score):
            torch.save(payload, self.best_pt_path)
            if score > self._best_score:
                self._best_score = score

        onnx_ok = False
        if export_onnx:
            try:
                cpu_model = copy.deepcopy(self.model).eval().to("cpu")
                wrapper = PolicyInferenceWrapper(cpu_model)
                dummy = torch.zeros(1, self.obs_dim, dtype=torch.float32)
                torch.onnx.export(
                    wrapper,
                    dummy,
                    str(self.onnx_path),
                    input_names=["obs"],
                    output_names=["joint_targets"],
                    opset_version=14,
                    dynamo=False,
                )
                onnx_ok = True
            except Exception:
                onnx_ok = False

        meta = self._build_meta(stats, payload["saved_at"], onnx_ok)
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    def load(self, prefer_best: bool = False) -> dict | None:
        """Load saved PyTorch .pt checkpoint, choosing the highest-scoring best snapshot when requested."""
        candidates = []
        paths = (self.best_pt_path, self.pt_path) if prefer_best else (self.pt_path, self.best_pt_path)
        for path in paths:
            if not path.exists():
                continue
            try:
                payload = torch.load(path, map_location=self.device, weights_only=False)
                stats = payload.get("stats", {})
                candidates.append((self._score_stats(stats), path, payload))
            except Exception:
                continue

        if not candidates:
            return None

        if prefer_best:
            candidates.sort(key=lambda item: item[0], reverse=True)
        _, chosen_path, payload = candidates[0]

        self.model.load_state_dict(payload["model_state"])
        if "optimizer_state" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state"])
        stats = dict(payload.get("stats", {}))
        self.update_count = int(stats.get("updates", 0))
        stats["_loaded_source"] = "best" if chosen_path == self.best_pt_path else "latest"

        # Promote legacy checkpoint into best_pt_path on first run so Resets cannot lose it
        if not self.best_pt_path.exists():
            torch.save(payload, self.best_pt_path)
            self._best_score = self._score_stats(stats)

        if prefer_best:
            if chosen_path == self.best_pt_path:
                shutil.copy2(chosen_path, self.pt_path)
            meta = self._build_meta(
                stats,
                payload.get("saved_at", time.strftime("%Y-%m-%d %H:%M:%S")),
                self.onnx_path.exists(),
                source=stats["_loaded_source"],
            )
            self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
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
