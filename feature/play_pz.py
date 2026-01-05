"""
play_pz.py

PettingZoo 环境试玩 / 自动对战 / 人类控制单个玩家

依赖：
  pip install pettingzoo gymnasium numpy torch

示例：
1) 人类控制 player_0，其余 bot 使用新策略（final）
  python play_pz.py --run_dir runs_pz/20260105_235959 --human_player 0 --bot-new-policy --render

2) 人类控制 player_2，其余 bot 使用旧策略（checkpoints 随机）
  python play_pz.py --run_dir runs_pz/20260105_235959 --human_player 2 --bot-old-policy --old_pick random

3) 自动对战：player_0 用 best，其余 bot 用旧策略
  python play_pz.py --run_dir runs_pz/20260105_235959 --auto-play --episodes 5 --render
"""

from __future__ import annotations

import os
import sys
sys.path.append(r"F:/round_godness/PettingZooV2")
import glob
import random
import argparse
from typing import Dict, Optional, List, Tuple
import time
import numpy as np
import torch
import torch.nn as nn

from bullet_dodge_pz import parallel_env
from game import ActionType


def masked_categorical_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    neg = torch.finfo(logits.dtype).min
    return torch.where(mask > 0.5, logits, torch.tensor(neg, device=logits.device, dtype=logits.dtype))


def ansi_highlight(s: str) -> str:
    return f"\033[93m{s}\033[0m"


def action_to_str(a_type: int, a_target: int, none_target: int) -> str:
    try:
        at = ActionType(a_type).name
    except Exception:
        at = f"T{a_type}"
    if a_target == none_target:
        return at
    return f"{at}->{a_target}"


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


def find_final(run_dir: str, agent: str) -> Optional[str]:
    cands = [
        os.path.join(run_dir, f"final_{agent}.pt"),
        os.path.join(run_dir, f"{agent}.pt"),
        os.path.join(run_dir, f"player_{agent.split('_')[1]}.pt"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    return None


def find_best(run_dir: str) -> Optional[str]:
    p = os.path.join(run_dir, "best.pt")
    return p if os.path.exists(p) else None


def find_old(run_dir: str, agent: str, pick: str) -> Optional[str]:
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    patt = os.path.join(ckpt_dir, f"upd_*_{agent}.pt")
    files = sorted(glob.glob(patt))
    if not files:
        # training-only-p0 case uses upd_*_player_0.pt only; fallback to final
        return None
    if pick == "latest":
        return files[-1]
    return random.choice(files)


def decode_action(env, action_id: int) -> Tuple[int, int]:
    # env.action_id_to_tuple exists in wrapper
    a_type, a_tgt = env.action_id_to_tuple(action_id)
    return int(a_type), int(a_tgt)


def print_action_help(env, agent: str, action_mask: np.ndarray):
    none_target = env.num_players
    valid = np.where(action_mask > 0.5)[0].tolist()
    print(f"\nValid actions for {agent}:")
    for aid in valid:
        at, tgt = decode_action(env, int(aid))
        print(f"  {aid:>3d}: {action_to_str(at, tgt, none_target)}")


@torch.no_grad()
def bot_policy_action(net: ActorCritic, obs: np.ndarray, mask: np.ndarray, device: torch.device, sample: bool) -> int:
    o = torch.tensor(obs[None, :], dtype=torch.float32, device=device)
    m = torch.tensor(mask[None, :], dtype=torch.float32, device=device)
    logits, _ = net(o)
    logits = masked_categorical_logits(logits, m)
    if sample:
        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())
    return int(torch.argmax(logits, dim=1).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="runs_pz/... 目录")
    ap.add_argument("--num_players", type=int, default=6)
    ap.add_argument("--max_rounds", type=int, default=200)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--bot_sample", action="store_true", help="bot 采样动作；默认 argmax")

    # control mode
    ap.add_argument("--human_player", type=int, default=-1, help=">=0 表示人类控制该玩家")
    ap.add_argument("--auto-play", action="store_true", help="player_0 使用 best；其余 bot 使用旧策略")

    # bot policy selection
    ap.add_argument("--bot-new-policy", dest="bot_new", action="store_true", help="bot 使用 final")
    ap.add_argument("--bot-old-policy", dest="bot_old", action="store_true", help="bot 使用 checkpoints")
    ap.add_argument("--old_pick", choices=["random", "latest"], default="random")

    args = ap.parse_args()

    device = torch.device(args.device)
    env = parallel_env(num_players=args.num_players, max_rounds=args.max_rounds, seed=args.seed)
    obs, infos = env.reset(seed=args.seed)

    agents = env.possible_agents
    obs_dim = next(iter(obs.values())).shape[0]
    act_dim = env.action_spaces[agents[0]].n

    # build nets
    nets: Dict[str, ActorCritic] = {a: ActorCritic(obs_dim, act_dim).to(device) for a in agents}
    for a in agents:
        nets[a].eval()

    # decide policy files
    def load_agent(agent: str, path: Optional[str]):
        if path is None:
            return
        sd = torch.load(path, map_location=device)
        nets[agent].load_state_dict(sd)

    if args.auto_play:
        # player_0 uses best
        best = find_best(args.run_dir)
        if best is None:
            raise FileNotFoundError("best.pt not found under run_dir")
        load_agent("player_0", best)
        # others old if possible else final
        for a in agents:
            if a == "player_0":
                continue
            p = find_old(args.run_dir, a, args.old_pick)
            if p is None:
                p = find_final(args.run_dir, a)
            load_agent(a, p)
    else:
        # player_0 uses final if exists
        p0_final = find_final(args.run_dir, "player_0")
        if p0_final is not None:
            load_agent("player_0", p0_final)

        # bot policy kind
        if args.bot_old and args.bot_new:
            raise ValueError("bot-new-policy and bot-old-policy are mutually exclusive")
        if not args.bot_old and not args.bot_new:
            args.bot_new = True

        for a in agents:
            if a == f"player_{args.human_player}" and args.human_player >= 0:
                # human player net unused
                continue
            if a == "player_0" and args.human_player == 0:
                continue

            if args.bot_old:
                p = find_old(args.run_dir, a, args.old_pick)
                if p is None:
                    p = find_final(args.run_dir, a)
            else:
                p = find_final(args.run_dir, a)
            load_agent(a, p)

    none_target = env.num_players

    def log_round(infos, rewards):
        r = int(env.game.round_idx)
        mode_name = env.game.mode.name
        died = infos["player_0"]["died_this_round"]
        ra = infos["player_0"]["round_actions"]
        print(f"\n[Round {r}] Mode={mode_name}")
        for i in range(env.num_players):
            p = env.game.players[i]
            a_type, a_tgt = ra[i]
            act_s = action_to_str(a_type, a_tgt, none_target)
            line = (
                f"  P{i}: act={act_s:<12} "
                f"alive={int(p.alive)} hp={p.hp} b={p.bullets} d={p.dodges} "
                f"{'DIED' if died[i] else ''} "
                f"r={rewards.get(f'player_{i}', 0.0):+.3f}"
            )
            if i == args.human_player or (args.auto_play and i == 0):
                line = ansi_highlight(">> " + line)
            print(line)

        winner = infos["player_0"].get("winner", None)
        if winner is not None:
            print("")
            print(ansi_highlight(f"==> WINNER: player_{winner}"))

    # episodes loop
    for ep in range(args.episodes):
        obs, infos = env.reset(seed=args.seed + ep)
        done = False
        if args.render:
            print("\n=== Episode", ep, "===")
        while not done:
            actions: Dict[str, int] = {}
            for a in agents:
                am = infos[a]["action_mask"].astype(np.float32)

                if args.human_player >= 0 and a == f"player_{args.human_player}":
                    print_action_help(env, a, am)
                    while True:
                        raw = input(f"Select action id for {a}: ").strip()
                        if raw.isdigit():
                            aid = int(raw)
                            if 0 <= aid < act_dim and am[aid] > 0.5:
                                actions[a] = aid
                                break
                        print("Invalid action id.")
                else:
                    actions[a] = bot_policy_action(nets[a], obs[a], am, device, sample=args.bot_sample)

            obs, rewards, terms, truncs, infos = env.step(actions)

            if args.render:
                log_round(infos, rewards)

            done = any(terms.values()) or any(truncs.values())
            time.sleep(0.5)

    env.close()


if __name__ == "__main__":
    main()
