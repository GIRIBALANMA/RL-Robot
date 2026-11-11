"""PyTorch CUDA Actor-Critic PPO learner with running observation normalization and .pt/.onnx saving."""

from __future__ import annotations

import json
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
    def update(self, x: torch.Tensor) -> None:
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = float(x.shape[0])

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * (self.count * batch_count / tot_count)

        self.mean.copy_(new_mean)
        self.var.copy_(torch.clamp(M2 / tot_count, min=1e-4))
        self.count.copy_(tot_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-6), -8.0, 8.0)


class ActorCritic(nn.Module):
    """Compact GPU Actor-Critic MLP for 45-dim biped state -> 15-dim joint targets."""

    def __init__(self, obs_dim: int = 45, act_dim: int = 15):
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

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-2].weight, gain=0.05)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        x = self.obs_norm(obs)
        mean = self.actor(x)
        value = self.critic(x).squeeze(-1)
        if deterministic:
            return mean, torch.zeros(obs.shape[0], device=obs.device), value
        std = torch.exp(torch.clamp(self.log_std, -2.5, -0.4))
        dist = Normal(mean, std)
        raw_act = dist.sample()
        log_prob = dist.log_prob(raw_act).sum(dim=-1)
        return raw_act, log_prob, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        x = self.obs_norm(obs)
        mean = self.actor(x)
        value = self.critic(x).squeeze(-1)
        std = torch.exp(torch.clamp(self.log_std, -2.5, -0.4))
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
        obs_dim: int = 45,
        act_dim: int = 15,
        lr: float = 4e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        epochs: int = 4,
        minibatches: int = 4,
        device: torch.device | None = None,
        models_dir: Path | str = "models",
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.onnx_path = self.models_dir / "steel_sentinel_policy.onnx"
        self.meta_path = self.models_dir / "checkpoints.json"

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
        pg_losses, vf_losses, entropies = [], [], []

        for _ in range(self.epochs):
            perm = torch.randperm(total, device=self.device)
            for start in range(0, total, batch_size):
                idx = perm[start : start + batch_size]
                new_logp, entropy, new_val = self.model.evaluate(b_obs[idx], b_act[idx])
                ratio = torch.exp(new_logp - b_logp[idx])
                surr1 = ratio * b_adv[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv[idx]
                pg_loss = -torch.min(surr1, surr2).mean()
                vf_loss = 0.5 * ((new_val - b_ret[idx]) ** 2).mean()
                ent = entropy.mean()
                loss = pg_loss + 0.5 * vf_loss - 0.008 * ent

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.7)
                self.optimizer.step()

                pg_losses.append(float(pg_loss.item()))
                vf_losses.append(float(vf_loss.item()))
                entropies.append(float(ent.item()))

        return {
            "policy_loss": round(sum(pg_losses) / max(1, len(pg_losses)), 4),
            "value_loss": round(sum(vf_losses) / max(1, len(vf_losses)), 4),
            "entropy": round(sum(entropies) / max(1, len(entropies)), 4),
        }

    def save(self, stats: dict, export_onnx: bool = True) -> dict:
        """Save real PyTorch .pt checkpoint and ONNX inference graph to models/."""
        payload = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "stats": stats,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        torch.save(payload, self.pt_path)

        onnx_ok = False
        if export_onnx:
            try:
                wrapper = PolicyInferenceWrapper(self.model).eval().to("cpu")
                dummy = torch.zeros(1, 45, dtype=torch.float32)
                torch.onnx.export(
                    wrapper,
                    dummy,
                    str(self.onnx_path),
                    input_names=["obs"],
                    output_names=["joint_targets"],
                    opset_version=14,
                    dynamo=False,
                )
                self.model.to(self.device)
                onnx_ok = True
            except Exception:
                self.model.to(self.device)

        meta = {
            "pt_file": self.pt_path.name,
            "pt_bytes": self.pt_path.stat().st_size if self.pt_path.exists() else 0,
            "onnx_file": self.onnx_path.name if onnx_ok else None,
            "onnx_bytes": self.onnx_path.stat().st_size if onnx_ok and self.onnx_path.exists() else 0,
            "saved_at": payload["saved_at"],
            "episode": stats.get("episode", 0),
            "total_steps": stats.get("total_steps", 0),
            "stage": stats.get("stage", 0),
            "best_reward": round(float(stats.get("best_reward", 0.0)), 2),
            "best_distance": round(float(stats.get("best_distance", 0.0)), 2),
            "device": str(self.device),
        }
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    def load(self) -> dict | None:
        """Load saved PyTorch .pt checkpoint if present."""
        if not self.pt_path.exists():
            return None
        payload = torch.load(self.pt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state"])
        if "optimizer_state" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state"])
        return payload.get("stats", {})

    def reset_weights(self) -> None:
        """Re-initialize fresh untrained neural network weights."""
        obs_dim = self.model.obs_norm.mean.shape[0]
        act_dim = self.model.log_std.shape[0]
        lr = self.optimizer.param_groups[0]["lr"]
        self.model = ActorCritic(obs_dim, act_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

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
