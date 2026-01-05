"""
play_pz.py

PettingZoo 环境试玩 / 人类控制（支持多个真人）/ bot 难度（easy/normal/hard/mixed）

bot 策略来源：
- 默认从 run_dir/checkpoints 中选取（不同难度对应不同阶段的 checkpoint）
- 同一个 checkpoint 会被多个 bot 共享加载（同一策略适配不同人数）

人类输入（NORMAL）：
- RELOAD / R
- DODGE / D
- ATTACK <id> / A <id>

人类输入（SUDDEN_DEATH，且启用决斗）：
- DODGE / D
- DUEL <id> / DU <id>

示例：
1) 6 人局，其中 2 个真人（player_0, player_1），其余 bot=normal
  python play_pz.py --run_dir runs_pz/20260105_235959 --num_players 6 --human-player-num 2 --normal --render

2) 12 人局，无决斗，1 个真人（player_0），bot=mixed
  python play_pz.py --run_dir runs_pz/20260105_235959 --num_players 12 --no-duel --human-player-num 1 --mixed --render

3) 全 bot 对战，hard
  python play_pz.py --run_dir runs_pz/20260105_235959 --num_players 10 --hard --episodes 5 --render
"""

from __future__ import annotations

import os
import glob
import random
import argparse
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from bullet_dodge_pz import parallel_env
from game import ActionType, Mode


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


def parse_upd_num(path: str) -> int:
    # .../upd_00123_player_0.pt
    base = os.path.basename(path)
    try:
        p = base.split("_")[1]
        return int(p)
    except Exception:
        return 0


def choose_ckpt(files: List[str], difficulty: str, rng: random.Random) -> Optional[str]:
    if not files:
        return None
    files = sorted(files, key=parse_upd_num)
    n = len(files)

    def pick_quantile(q: float, jitter: float = 0.1) -> str:
        idx = int(round((n - 1) * q))
        span = max(1, int(round(n * jitter)))
        lo = max(0, idx - span)
        hi = min(n - 1, idx + span)
        return files[rng.randint(lo, hi)]

    if difficulty == "easy":
        return pick_quantile(0.15, jitter=0.15)
    if difficulty == "normal":
        return pick_quantile(0.50, jitter=0.10)
    if difficulty == "hard":
        return pick_quantile(0.90, jitter=0.08)

    # mixed
    return choose_ckpt(files, rng.choice(["easy", "normal", "hard"]), rng)


@torch.no_grad()
def bot_action(net: ActorCritic, obs: np.ndarray, mask: np.ndarray, device: torch.device, sample: bool) -> int:
    o = torch.tensor(obs[None, :], dtype=torch.float32, device=device)
    m = torch.tensor(mask[None, :], dtype=torch.float32, device=device)
    logits, _ = net(o)
    logits = masked_categorical_logits(logits, m)
    if sample:
        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())
    return int(torch.argmax(logits, dim=1).item())


def print_round(env, infos, rewards, highlight_ids: List[int], player_tags: Dict[int, str]):
    r = int(env.game.round_idx)
    mode_name = env.game.mode.name
    none_target = env.max_players
    died = infos["player_0"]["died_this_round"]
    ra = infos["player_0"]["round_actions"]

    print(f"\n[Round {r}] Mode={mode_name}")
    for i in range(env.num_players):
        p = env.game.players[i]
        a_type, a_tgt = ra[i]
        act_s = action_to_str(a_type, a_tgt, none_target)
        tag = player_tags.get(i, "normal")
        line = (
            f"  P{i}({tag}): act={act_s:<14} "
            f"alive={int(p.alive)} hp={p.hp} b={p.bullets} d={p.dodges} "
            f"{'DIED' if died[i] else ''} "
            f"r={rewards.get(f'player_{i}', 0.0):+.3f}"
        )
        if i in highlight_ids:
            line = ansi_highlight(">> " + line)
        print(line)

    winner = infos["player_0"].get("winner", None)
    ended = bool(infos["player_0"].get("ended", False))
    if ended:
        print("")
        if winner is None:
            print(ansi_highlight("==> RESULT: DRAW"))
        else:
            print(ansi_highlight(f"==> WINNER: player_{winner}"))


def parse_human_cmd(raw: str, mode: Mode) -> Tuple[ActionType, int, bool]:
    """
    return: (action_type, target, has_target)
    """
    s = raw.strip().upper()
    if not s:
        return ActionType.NOOP, -1, False

    parts = s.replace(",", " ").split()
    head = parts[0]

    # normalize abbreviations
    if head in ["R", "RELOAD"]:
        return ActionType.RELOAD, -1, False
    if head in ["D", "DODGE"]:
        return ActionType.DODGE, -1, False

    if mode == Mode.NORMAL:
        if head in ["A", "ATTACK"]:
            if len(parts) >= 2 and parts[1].lstrip("-").isdigit():
                return ActionType.ATTACK, int(parts[1]), True
            return ActionType.ATTACK, -1, True
        return ActionType.NOOP, -1, False

    # sudden death
    if head in ["DU", "DUEL"]:
        if len(parts) >= 2 and parts[1].lstrip("-").isdigit():
            return ActionType.DUEL, int(parts[1]), True
        return ActionType.DUEL, -1, True

    return ActionType.NOOP, -1, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="runs_pz/... 目录")

    ap.add_argument("--num_players", type=int, default=6)
    ap.add_argument("--max_players", type=int, default=32)
    ap.add_argument("--max_rounds", type=int, default=200)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--bot_sample", action="store_true", help="bot 采样动作；默认 argmax")
    ap.add_argument("--auto-play", action="store_true",
                    help="player_0 uses best.pt; other bots pick checkpoints by difficulty")


    ap.add_argument("--duel", dest="duel_enabled", action="store_true")
    ap.add_argument("--no-duel", dest="duel_enabled", action="store_false")
    ap.set_defaults(duel_enabled=True)

    ap.add_argument("--init_hp", type=int, default=2)
    ap.add_argument("--init_bullets", type=int, default=1)
    ap.add_argument("--init_dodges", type=int, default=2)

    ap.add_argument("--human-player-num", dest="human_num", type=int, default=0, help="真人玩家数量（从 player_0 开始）")
    ap.add_argument("--interval_time", type=float, default=0.5, help='每轮间隔时间')

    # difficulty flags
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--easy", action="store_true")
    g.add_argument("--normal", action="store_true")
    g.add_argument("--hard", action="store_true")
    g.add_argument("--mixed", action="store_true")

    args = ap.parse_args()

    if args.num_players < 4:
        raise ValueError("num_players 必须 >= 4")
    if args.num_players > args.max_players:
        raise ValueError("num_players 不能超过 max_players")
    if args.human_num < 0 or args.human_num > args.num_players:
        raise ValueError("human-player-num 必须在 [0, num_players]")
    if args.auto_play and args.human_num != 0:
        raise ValueError("--auto-play requires --human-player-num=0")


    if args.easy:
        difficulty = "easy"
    elif args.hard:
        difficulty = "hard"
    elif args.mixed:
        difficulty = "mixed"
    else:
        difficulty = "normal"

    rng = random.Random(args.seed)

    env = parallel_env(
        num_players=args.num_players,
        max_players=args.max_players,
        duel_enabled=args.duel_enabled,
        init_hp=args.init_hp,
        init_bullets=args.init_bullets,
        init_dodges=args.init_dodges,
        max_rounds=args.max_rounds,
        seed=args.seed,
    )
    obs, infos = env.reset(seed=args.seed)

    agents = env.possible_agents
    obs_dim = next(iter(obs.values())).shape[0]
    act_dim = env.action_spaces[agents[0]].n
    device = torch.device(args.device)

    # policy file selection from checkpoints
    ckpt_files = sorted(glob.glob(os.path.join(args.run_dir, "checkpoints", "upd_*_player_0.pt")))
    if not ckpt_files:
        raise FileNotFoundError("No checkpoints found under run_dir/checkpoints (expect upd_*_player_0.pt)")

    # nets
    nets: Dict[str, ActorCritic] = {a: ActorCritic(obs_dim, act_dim).to(device) for a in agents}
    for a in agents:
        nets[a].eval()

    # human players: player_0..player_{human_num-1}
    human_ids = list(range(args.human_num))
    bot_ids = [i for i in range(env.num_players) if i not in human_ids]
    prompt_human_ids = human_ids[:]   # 本局仍需要键盘输入的真人玩家（每回合会更新）

    player_tags = {i: "human" for i in human_ids}

    highlight_ids = human_ids[:]

    if args.auto_play or args.human_num == 0:
        best_path = os.path.join(args.run_dir, "best.pt")
        if not os.path.exists(best_path):
            raise FileNotFoundError("best.pt not found under run_dir")
        nets["player_0"].load_state_dict(torch.load(best_path, map_location=device))

        player_tags[0] = "best"

        if 0 in bot_ids:
            bot_ids.remove(0)

        highlight_ids = [0] + highlight_ids


    # load bot policies from checkpoint (shared policy is allowed)
    def load_path_into_all(path: str, ids: List[int]):
        sd = torch.load(path, map_location=device)
        for i in ids:
            nets[f"player_{i}"].load_state_dict(sd)

    if difficulty != "mixed":
        p = choose_ckpt(ckpt_files, difficulty, rng)
        load_path_into_all(p, bot_ids)
        bot_ckpt_map = {i: os.path.basename(p) for i in bot_ids}
        for i in bot_ids:
            player_tags[i] = difficulty
    else:
        bot_ckpt_map = {}
        path_to_ids: Dict[str, List[int]] = {}

        for i in bot_ids:
            d = rng.choices(["easy", "normal", "hard"], weights=[1, 1, 1], k=1)[0]
            p = choose_ckpt(ckpt_files, d, rng)
            path_to_ids.setdefault(p, []).append(i)
            #bot_ckpt_map[i] = os.path.basename(p)
            sd = torch.load(p, map_location=device)
            nets[f"player_{i}"].load_state_dict(sd)
            bot_ckpt_map[i] = os.path.basename(p)
            player_tags[i] = d

        for p, ids in path_to_ids.items():
            load_path_into_all(p, ids)


    if args.render:
        print("Human players:", human_ids)
        print("Bot difficulty:", difficulty)
        if bot_ckpt_map:
            sample_show = dict(list(bot_ckpt_map.items())[: min(6, len(bot_ckpt_map))])
            print("Bot ckpt sample:", sample_show)

    # episode loop
    for ep in range(args.episodes):
        obs, infos = env.reset(seed=args.seed + ep)
        done = False
        if args.render:
            print("\n=== Episode", ep, "===")

        prompt_human_ids = human_ids[:]

        while not done:
            # 更新：死亡真人不再要求输入
            if prompt_human_ids:
                prompt_human_ids = [hid for hid in prompt_human_ids if env.game.players[hid].alive]


            actions: Dict[str, int] = {}
            mode = env.game.mode

            for i in range(env.num_players):
                agent = f"player_{i}"
                am = infos[agent]["action_mask"].astype(np.float32)

                if i in prompt_human_ids:
                    print("")
                    print(ansi_highlight(f"[Human] player_{i} | mode={mode.name} | hp={env.game.players[i].hp} b={env.game.players[i].bullets} d={env.game.players[i].dodges}"))
                    if mode == Mode.NORMAL:
                        print("Commands: R/RELOAD, D/DODGE, A/ATTACK <id>")
                    else:
                        if env.duel_enabled:
                            print("Commands: D/DODGE, DU/DUEL <id>")
                        else:
                            print("Commands: D/DODGE")

                    while True:
                        raw = input(f"player_{i}> ").strip()
                        at, tgt, has_tgt = parse_human_cmd(raw, mode)
                        if at == ActionType.NOOP:
                            print("Invalid command.")
                            continue

                        if at in [ActionType.RELOAD, ActionType.DODGE]:
                            aid = env.tuple_to_action_id(int(at), env.max_players)
                        else:
                            if not has_tgt or tgt < 0:
                                print("Missing target id.")
                                continue
                            aid = env.tuple_to_action_id(int(at), int(tgt))

                        if 0 <= aid < act_dim and am[aid] > 0.5:
                            actions[agent] = aid
                            break
                        print("Action not allowed now (check ammo/dodge/target).")
                else:
                    actions[agent] = bot_action(nets[agent], obs[agent], am, device, sample=args.bot_sample)

            obs, rewards, terms, truncs, infos = env.step(actions)

            if args.render:
                print_round(env, infos, rewards, highlight_ids=highlight_ids, player_tags=player_tags)

            done = any(terms.values()) or any(truncs.values())
            import time
            time.sleep(args.interval_time)

    env.close()


if __name__ == "__main__":
    main()
