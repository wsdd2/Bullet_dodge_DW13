"""
train_ppo_pz.py - PettingZoo ParallelEnv + 独立 PPO-Clip（每个玩家一个策略网络）

依赖：
  pip install pettingzoo gymnasium torch numpy

用法：
  python train_ppo_pz.py --num_players 6 --steps 500000 --device cuda
  python train_ppo_pz.py --num_players 8 --steps 800000 --device cpu

输出：
  runs_pz/<timestamp>/player_i.pt  (每个玩家一份参数)
  runs_pz/<timestamp>/config.json

实现要点：
- 每个 agent 独立 PPO（有自己的 ActorCritic + Optimizer）
- 同步采样：每一步所有 agent 同时出动作，env 同步结算
- action_mask：对 logits 做 -1e9 屏蔽无效动作，采样/计算 logp 都使用 mask

注意：
- 多智能体对抗学习本质是非平稳问题。这个“独立 PPO”是一个能跑的基线；
  若想更稳、更强，下一步通常会做：
    * league / policy pool（对手来自历史快照）
    * centralized critic（共享价值网络或加对手特征）
"""

from __future__ import annotations

import os
import time
import json
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from bullet_dodge_pz import parallel_env


def masked_categorical_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, A]
    mask:   [B, A] {0,1}
    """
    # mask=0 -> very negative
    neg = torch.finfo(logits.dtype).min
    masked = torch.where(mask > 0.5, logits, torch.tensor(neg, device=logits.device, dtype=logits.dtype))
    return masked


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, depth: int = 2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU())
            d = hidden
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.body = MLP(obs_dim, hidden=hidden, depth=2)
        self.pi = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        logits = self.pi(h)
        value = self.v(h).squeeze(-1)
        return logits, value


@dataclass
class PPOCfg:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    epochs: int = 4
    minibatch: int = 256
    rollout_len: int = 512
    max_grad_norm: float = 0.5


def compute_gae(rews: np.ndarray, vals: np.ndarray, dones: np.ndarray, gamma: float, lam: float, last_val: float) -> (np.ndarray, np.ndarray):
    """
    rews: [T]
    vals: [T]
    dones:[T] (1 if episode ended at this step)
    """
    T = len(rews)
    adv = np.zeros((T,), dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_nonterminal = 1.0 - dones[t]
        next_val = last_val if t == T - 1 else vals[t + 1]
        delta = rews[t] + gamma * next_val * next_nonterminal - vals[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae
    ret = adv + vals
    return adv, ret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_players", type=int, default=6)
    ap.add_argument("--steps", type=int, default=500000)
    ap.add_argument("--max_rounds", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--rollout_len", type=int, default=512)
    args = ap.parse_args()

    cfg = PPOCfg(rollout_len=args.rollout_len)

    env = parallel_env(num_players=args.num_players, max_rounds=args.max_rounds, seed=args.seed)
    obs, infos = env.reset(seed=args.seed)

    agents = env.possible_agents
    obs_dim = next(iter(obs.values())).shape[0]
    act_dim = env.action_spaces[agents[0]].n

    device = torch.device(args.device)

    nets = {a: ActorCritic(obs_dim, act_dim, hidden=256).to(device) for a in agents}
    opts = {a: optim.Adam(nets[a].parameters(), lr=cfg.lr) for a in agents}

    run_dir = os.path.join("runs_pz", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "num_players": args.num_players,
            "steps": args.steps,
            "max_rounds": args.max_rounds,
            "seed": args.seed,
            "obs_dim": int(obs_dim),
            "act_dim": int(act_dim),
            "ppo": cfg.__dict__,
        }, f, ensure_ascii=False, indent=2)

    # rollout buffers per agent
    T = cfg.rollout_len
    obs_buf = {a: np.zeros((T, obs_dim), dtype=np.float32) for a in agents}
    act_buf = {a: np.zeros((T,), dtype=np.int64) for a in agents}
    logp_buf = {a: np.zeros((T,), dtype=np.float32) for a in agents}
    val_buf = {a: np.zeros((T,), dtype=np.float32) for a in agents}
    rew_buf = {a: np.zeros((T,), dtype=np.float32) for a in agents}
    done_buf = {a: np.zeros((T,), dtype=np.float32) for a in agents}
    mask_buf = {a: np.zeros((T, act_dim), dtype=np.float32) for a in agents}

    steps_done = 0
    start = time.time()
    ep = 0

    while steps_done < args.steps:
        # collect rollout
        for t in range(T):
            # sample actions for all agents
            actions: Dict[str, int] = {}
            for a in agents:
                o = obs[a].astype(np.float32)
                m = infos[a]["action_mask"].astype(np.float32)

                obs_buf[a][t] = o
                mask_buf[a][t] = m

                o_t = torch.tensor(o[None, :], dtype=torch.float32, device=device)
                m_t = torch.tensor(m[None, :], dtype=torch.float32, device=device)

                with torch.no_grad():
                    logits, v = nets[a](o_t)
                    logits = masked_categorical_logits(logits, m_t)
                    dist = torch.distributions.Categorical(logits=logits)
                    act = dist.sample()
                    logp = dist.log_prob(act)

                actions[a] = int(act.item())
                act_buf[a][t] = int(act.item())
                logp_buf[a][t] = float(logp.item())
                val_buf[a][t] = float(v.item())

            next_obs, rewards, terms, truncs, next_infos = env.step(actions)
            done_flag = any(terms.values()) or any(truncs.values())

            for a in agents:
                rew_buf[a][t] = float(rewards.get(a, 0.0))
                done_buf[a][t] = 1.0 if done_flag else 0.0

            obs, infos = next_obs, next_infos
            steps_done += len(agents)

            if done_flag:
                ep += 1
                obs, infos = env.reset(seed=args.seed + ep)

            if steps_done >= args.steps:
                break

        # PPO update for each agent
        for a in agents:
            # bootstrap value
            if done_buf[a][-1] > 0.5:
                last_val = 0.0
            else:
                o = obs[a].astype(np.float32)
                m = infos[a]["action_mask"].astype(np.float32)
                o_t = torch.tensor(o[None, :], dtype=torch.float32, device=device)
                with torch.no_grad():
                    _, v = nets[a](o_t)
                last_val = float(v.item())

            adv, ret = compute_gae(
                rews=rew_buf[a],
                vals=val_buf[a],
                dones=done_buf[a],
                gamma=cfg.gamma,
                lam=cfg.gae_lambda,
                last_val=last_val,
            )
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            b_obs = torch.tensor(obs_buf[a], dtype=torch.float32, device=device)
            b_act = torch.tensor(act_buf[a], dtype=torch.int64, device=device)
            b_logp_old = torch.tensor(logp_buf[a], dtype=torch.float32, device=device)
            b_adv = torch.tensor(adv, dtype=torch.float32, device=device)
            b_ret = torch.tensor(ret, dtype=torch.float32, device=device)
            b_mask = torch.tensor(mask_buf[a], dtype=torch.float32, device=device)

            idx = np.arange(T)
            for _ in range(cfg.epochs):
                np.random.shuffle(idx)
                for s in range(0, T, cfg.minibatch):
                    mb = idx[s:s + cfg.minibatch]
                    logits, value = nets[a](b_obs[mb])
                    logits = masked_categorical_logits(logits, b_mask[mb])

                    dist = torch.distributions.Categorical(logits=logits)
                    logp = dist.log_prob(b_act[mb])
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(logp - b_logp_old[mb])
                    surr1 = ratio * b_adv[mb]
                    surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * b_adv[mb]
                    pi_loss = -torch.min(surr1, surr2).mean()

                    v_loss = 0.5 * (b_ret[mb] - value).pow(2).mean()
                    loss = pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy

                    opts[a].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(nets[a].parameters(), cfg.max_grad_norm)
                    opts[a].step()

        if (steps_done // len(agents)) % (T * 10) < T:
            fps = int(steps_done / max(1e-6, (time.time() - start)))
            print(f"[PZ-PPO] env_steps={steps_done//len(agents)} sample_steps={steps_done} fps={fps}")

    # save
    for a in agents:
        torch.save(nets[a].state_dict(), os.path.join(run_dir, f"{a}.pt"))
    print("Saved to:", run_dir)


if __name__ == "__main__":
    main()
