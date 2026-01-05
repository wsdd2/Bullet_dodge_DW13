"""
play_pz.py - 用训练好的多策略在 PettingZoo 环境里对战（或混合随机）

用法：
  python play_pz.py --num_players 6 --ckpt_dir runs_pz/xxxx

默认：如果 ckpt_dir 存在 player_i.pt 就用该策略；否则该玩家随机合法动作。
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import torch
import torch.nn as nn

from bullet_dodge_pz import parallel_env
from game import ActionType


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


def masked_argmax(logits: np.ndarray, mask: np.ndarray) -> int:
    masked = logits.copy()
    masked[mask <= 0.5] = -1e9
    return int(np.argmax(masked))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_players", type=int, default=6)
    ap.add_argument("--ckpt_dir", type=str, default="")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    env = parallel_env(num_players=args.num_players, max_rounds=200, seed=0)
    obs, infos = env.reset(seed=0)

    agents = env.possible_agents
    obs_dim = next(iter(obs.values())).shape[0]
    act_dim = env.action_spaces[agents[0]].n
    device = torch.device(args.device)

    nets = {}
    for a in agents:
        net = ActorCritic(obs_dim, act_dim, hidden=256).to(device)
        ckpt = os.path.join(args.ckpt_dir, f"{a}.pt") if args.ckpt_dir else ""
        if ckpt and os.path.exists(ckpt):
            net.load_state_dict(torch.load(ckpt, map_location=device))
            net.eval()
            nets[a] = net
        else:
            nets[a] = None  # random

    for ep in range(args.episodes):
        obs, infos = env.reset(seed=ep)
        done = False
        while not done:
            actions = {}
            for a in agents:
                mask = infos[a]["action_mask"].astype(np.float32)
                if nets[a] is None:
                    legal = np.where(mask > 0.5)[0]
                    actions[a] = int(np.random.choice(legal)) if len(legal) else 0
                else:
                    o = torch.tensor(obs[a][None, :], dtype=torch.float32, device=device)
                    logits, _ = nets[a](o)
                    logits = logits[0].cpu().numpy()
                    actions[a] = masked_argmax(logits, mask)

            obs, rewards, terms, truncs, infos = env.step(actions)
            if args.render:
                env.render()
            done = any(terms.values()) or any(truncs.values())

        # winner info exists in infos
        any_info = next(iter(infos.values()))
        print(f"Episode {ep} ended. winner={any_info.get('winner')} round={any_info.get('round')} mode={any_info.get('mode')}")

    env.close()


if __name__ == "__main__":
    main()
