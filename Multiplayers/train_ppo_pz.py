"""
train_ppo_pz.py

PettingZoo ParallelEnv + PPO-Clip（独立策略）

PPO-Clip（要点）：
- policy ratio r = pi_new(a|s) / pi_old(a|s)
- 用 clip(r, 1-eps, 1+eps) 限制每次更新的幅度，避免策略大跳跃导致训练不稳定
- 同时训练 value function（状态价值）与 policy（策略），并加 entropy 奖励维持探索

特性：
- action_mask：对无效动作做 logits 屏蔽
- 训练日志：rolling 胜率 / 平均回报 / 平均局长 / 决斗触发比例 / PPO loss 指标
- checkpoint：定期保存策略快照
- best.pt：按指标保存 player_0 的最佳策略
- 固定对手模式：加载外部 .pt 并冻结（只训练 player_0）

依赖：
  pip install pettingzoo gymnasium numpy torch

示例：
1) 全员同时学习（默认）
  python train_ppo_pz.py --device cuda

2) 固定对手，仅训练 player_0（对手都加载同一个策略文件）
  python train_ppo_pz.py --device cuda --opponent_policy_path runs_pz/xxx/checkpoints/upd_00100_player_0.pt
"""

from __future__ import annotations

import os
import time
import json
import argparse
import glob
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from bullet_dodge_pz import parallel_env
from game import ActionType


# ---------------- utils ----------------

def masked_categorical_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """logits: [B, A], mask: [B, A] (0/1)"""
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


# ---------------- model ----------------

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
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    lr: float = 3e-4
    epochs: int = 4
    minibatch: int = 256
    rollout_len: int = 512
    max_grad_norm: float = 0.5


def compute_gae(rews: np.ndarray, vals: np.ndarray, dones: np.ndarray, gamma: float, lam: float, last_val: float) -> Tuple[np.ndarray, np.ndarray]:
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


def ppo_update(
    net: ActorCritic,
    opt: optim.Optimizer,
    b_obs: torch.Tensor,
    b_act: torch.Tensor,
    b_logp_old: torch.Tensor,
    b_adv: torch.Tensor,
    b_ret: torch.Tensor,
    b_mask: torch.Tensor,
    cfg: PPOCfg,
) -> Dict[str, float]:
    idx = np.arange(b_obs.shape[0])
    pi_losses, v_losses, entropies, clip_fracs = [], [], [], []

    for _ in range(cfg.epochs):
        np.random.shuffle(idx)
        for s in range(0, len(idx), cfg.minibatch):
            mb = idx[s:s + cfg.minibatch]
            logits, value = net(b_obs[mb])
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

            clip_frac = (torch.abs(ratio - 1.0) > cfg.clip_eps).float().mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            opt.step()

            pi_losses.append(float(pi_loss.item()))
            v_losses.append(float(v_loss.item()))
            entropies.append(float(entropy.item()))
            clip_fracs.append(float(clip_frac.item()))

    return {
        "pi_loss": float(np.mean(pi_losses)) if pi_losses else 0.0,
        "v_loss": float(np.mean(v_losses)) if v_losses else 0.0,
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
        "clip_frac": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
    }


def find_ckpts(run_dir: str, agent: str) -> List[str]:
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    patt = os.path.join(ckpt_dir, f"upd_*_{agent}.pt")
    return sorted(glob.glob(patt))


@torch.no_grad()
def write_eval_logs(env, obs_dim: int, act_dim: int, device: torch.device, run_dir: str, episodes: int):
    """
    player_0 使用 best.pt
    其余玩家从 run_dir/checkpoints 中随机抽取旧策略（若不足则使用 best）
    """
    eval_dir = os.path.join(run_dir, "eval_logs")
    os.makedirs(eval_dir, exist_ok=True)

    agents = env.possible_agents
    nets: Dict[str, ActorCritic] = {}

    best_path = os.path.join(run_dir, "best.pt")
    if not os.path.exists(best_path):
        return

    net0 = ActorCritic(obs_dim, act_dim).to(device)
    net0.load_state_dict(torch.load(best_path, map_location=device))
    net0.eval()
    nets["player_0"] = net0

    for a in agents:
        if a == "player_0":
            continue
        net = ActorCritic(obs_dim, act_dim).to(device)
        files = find_ckpts(run_dir, "player_0")
        p = random.choice(files) if files else best_path
        net.load_state_dict(torch.load(p, map_location=device))
        net.eval()
        nets[a] = net

    none_target = env.max_players

    for ep in range(episodes):
        obs, infos = env.reset(seed=10000 + ep)
        done = False
        lines: List[str] = []
        lines.append(f"=== Eval Episode {ep} ===\n")

        while not done:
            actions: Dict[str, int] = {}
            for a in agents:
                o = torch.tensor(obs[a][None, :], dtype=torch.float32, device=device)
                m = torch.tensor(infos[a]["action_mask"][None, :], dtype=torch.float32, device=device)
                logits, _ = nets[a](o)
                logits = masked_categorical_logits(logits, m)
                act = int(torch.argmax(logits, dim=1).item())
                actions[a] = act

            obs, rewards, terms, truncs, infos = env.step(actions)
            r = int(env.game.round_idx)
            mode_name = env.game.mode.name

            lines.append(f"[Round {r}] Mode={mode_name}")
            died = infos["player_0"]["died_this_round"]
            ra = infos["player_0"]["round_actions"]

            for i in range(env.num_players):
                p = env.game.players[i]
                a_type, a_tgt = ra[i]
                act_s = action_to_str(a_type, a_tgt, none_target)
                line = (
                    f"  P{i}: act={act_s:<14} "
                    f"alive={int(p.alive)} hp={p.hp} b={p.bullets} d={p.dodges} "
                    f"{'DIED' if died[i] else ''} "
                    f"r={rewards.get(f'player_{i}', 0.0):+.3f}"
                )
                if i == 0:
                    line = ansi_highlight(">> " + line)
                lines.append(line)

            winner = infos["player_0"].get("winner", None)
            ended = infos["player_0"].get("ended", False)
            if ended:
                lines.append("")
                if winner is None:
                    lines.append(ansi_highlight("==> RESULT: DRAW"))
                else:
                    lines.append(ansi_highlight(f"==> WINNER: player_{winner}"))
            lines.append("")

            done = any(terms.values()) or any(truncs.values())

        with open(os.path.join(eval_dir, f"eval_{ep}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()

    # env config
    ap.add_argument("--num_players", type=int, default=10)
    ap.add_argument("--max_players", type=int, default=32)
    ap.add_argument("--max_rounds", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--duel", dest="duel_enabled", action="store_true")
    ap.add_argument("--no-duel", dest="duel_enabled", action="store_false")
    ap.set_defaults(duel_enabled=True)

    ap.add_argument("--init_hp", type=int, default=2)
    ap.add_argument("--init_bullets", type=int, default=1)
    ap.add_argument("--init_dodges", type=int, default=2)

    # training
    ap.add_argument("--steps", type=int, default=800000, help="sample_steps = env_steps * num_players")
    ap.add_argument("--rollout_len", type=int, default=512)
    ap.add_argument("--stats_window", type=int, default=100)
    ap.add_argument("--print_every_updates", type=int, default=10)
    ap.add_argument("--save_every_updates", type=int, default=50)
    ap.add_argument("--eval_episodes", type=int, default=3)

    ap.add_argument("--best_metric", choices=["p0_winrate", "p0_reward"], default="p0_winrate")

    # fixed opponents: all opponents load one .pt and freeze
    ap.add_argument("--opponent_policy_path", type=str, default="", help="固定对手策略文件（只训练 player_0）")
    ap.add_argument("--train_all", action="store_true", help="全员同时学习")
    ap.add_argument("--render_info", action="store_true", help="是否输出决斗结果")
    args = ap.parse_args()

    cfg = PPOCfg(rollout_len=args.rollout_len)

    env = parallel_env(
        num_players=args.num_players,
        max_players=args.max_players,
        duel_enabled=args.duel_enabled,
        init_hp=args.init_hp,
        init_bullets=args.init_bullets,
        init_dodges=args.init_dodges,
        max_rounds=args.max_rounds,
        seed=args.seed,
        render_info=args.render_info,
    )
    obs, infos = env.reset(seed=args.seed)

    agents = env.possible_agents
    obs_dim = next(iter(obs.values())).shape[0]
    act_dim = env.action_spaces[agents[0]].n
    device = torch.device(args.device)

    # policy nets
    nets: Dict[str, ActorCritic] = {a: ActorCritic(obs_dim, act_dim).to(device) for a in agents}
    for a in agents:
        nets[a].train()

    fixed_opponents = (not args.train_all) and bool(args.opponent_policy_path)
    if fixed_opponents:
        p = args.opponent_policy_path
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        sd = torch.load(p, map_location=device)
        for a in agents:
            if a == "player_0":
                continue
            nets[a].load_state_dict(sd)
            nets[a].eval()  # freeze

    if fixed_opponents:
        opts = {"player_0": optim.Adam(nets["player_0"].parameters(), lr=cfg.lr)}
    else:
        opts = {a: optim.Adam(nets[a].parameters(), lr=cfg.lr) for a in agents}

    run_dir = os.path.join("runs_pz", time.strftime("%Y%m%d_%H%M%S"))
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "env": {
                "num_players": args.num_players,
                "max_players": args.max_players,
                "max_rounds": args.max_rounds,
                "seed": args.seed,
                "duel_enabled": args.duel_enabled,
                "init_hp": args.init_hp,
                "init_bullets": args.init_bullets,
                "init_dodges": args.init_dodges,
                "render_info": args.render_info,
            },
            "train": {
                "steps": args.steps,
                "rollout_len": cfg.rollout_len,
                "fixed_opponents": fixed_opponents,
                "opponent_policy_path": args.opponent_policy_path,
            },
            "model": {"obs_dim": int(obs_dim), "act_dim": int(act_dim)},
            "ppo": cfg.__dict__,
        }, f, ensure_ascii=False, indent=2)

    T = cfg.rollout_len
    obs_buf = {a: np.zeros((T, obs_dim), dtype=np.float32) for a in agents}
    act_buf = {a: np.zeros((T,), dtype=np.int64) for a in agents}
    logp_buf = {a: np.zeros((T,), dtype=np.float32) for a in agents}
    val_buf = {a: np.zeros((T,), dtype=np.float32) for a in agents}
    rew_buf = {a: np.zeros((T,), dtype=np.float32) for a in agents}
    done_buf = {a: np.zeros((T,), dtype=np.float32) for a in agents}
    mask_buf = {a: np.zeros((T, act_dim), dtype=np.float32) for a in agents}

    win_hist = deque(maxlen=args.stats_window)
    sd_hist = deque(maxlen=args.stats_window)
    draw_hist = deque(maxlen=args.stats_window)
    ep_len_hist = deque(maxlen=args.stats_window)
    ep_ret_hist_p0 = deque(maxlen=args.stats_window)

    cur_ep_ret = {a: 0.0 for a in agents}
    cur_ep_len = 0
    cur_ep_sd = False

    best_score = -1e9
    best_path = os.path.join(run_dir, "best.pt")

    steps_done = 0
    env_steps_done = 0
    updates_done = 0
    start = time.time()

    while steps_done < args.steps:
        # rollout
        for t in range(T):
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
            env_steps_done += 1
            steps_done += len(agents)

            done_flag = any(terms.values()) or any(truncs.values())
            sd_started = bool(next_infos["player_0"].get("sudden_death_started", False))
            cur_ep_sd = cur_ep_sd or sd_started

            for a in agents:
                r = float(rewards.get(a, 0.0))
                rew_buf[a][t] = r
                done_buf[a][t] = 1.0 if done_flag else 0.0
                cur_ep_ret[a] += r

            cur_ep_len += 1
            obs, infos = next_obs, next_infos

            if done_flag:
                winner = next_infos["player_0"].get("winner", None)
                ended = bool(next_infos["player_0"].get("ended", False))
                is_draw = ended and (winner is None)

                win_hist.append(1 if (winner is not None and int(winner) == 0) else 0)
                draw_hist.append(1 if is_draw else 0)
                sd_hist.append(1 if cur_ep_sd else 0)
                ep_len_hist.append(cur_ep_len)
                ep_ret_hist_p0.append(cur_ep_ret["player_0"])

                cur_ep_ret = {a: 0.0 for a in agents}
                cur_ep_len = 0
                cur_ep_sd = False

                obs, infos = env.reset(seed=args.seed + env_steps_done)

            if steps_done >= args.steps:
                break

        updates_done += 1

        # update
        update_agents = ["player_0"] if fixed_opponents else agents
        update_metrics: Dict[str, Dict[str, float]] = {}

        for a in update_agents:
            if done_buf[a][-1] > 0.5:
                last_val = 0.0
            else:
                o = obs[a].astype(np.float32)
                o_t = torch.tensor(o[None, :], dtype=torch.float32, device=device)
                with torch.no_grad():
                    _, v = nets[a](o_t)
                last_val = float(v.item())

            adv, ret = compute_gae(rew_buf[a], val_buf[a], done_buf[a], cfg.gamma, cfg.gae_lambda, last_val)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            b_obs = torch.tensor(obs_buf[a], dtype=torch.float32, device=device)
            b_act = torch.tensor(act_buf[a], dtype=torch.int64, device=device)
            b_logp_old = torch.tensor(logp_buf[a], dtype=torch.float32, device=device)
            b_adv = torch.tensor(adv, dtype=torch.float32, device=device)
            b_ret = torch.tensor(ret, dtype=torch.float32, device=device)
            b_mask = torch.tensor(mask_buf[a], dtype=torch.float32, device=device)

            update_metrics[a] = ppo_update(nets[a], opts[a], b_obs, b_act, b_logp_old, b_adv, b_ret, b_mask, cfg)

        # checkpoints
        if updates_done % args.save_every_updates == 0:
            torch.save(nets["player_0"].state_dict(), os.path.join(ckpt_dir, f"upd_{updates_done:05d}_player_0.pt"))
            if not fixed_opponents:
                for a in agents:
                    if a == "player_0":
                        continue
                    torch.save(nets[a].state_dict(), os.path.join(ckpt_dir, f"upd_{updates_done:05d}_{a}.pt"))

        # best
        if len(ep_len_hist) > 0:
            p0_winrate = float(np.mean(win_hist)) if len(win_hist) else 0.0
            p0_rew = float(np.mean(ep_ret_hist_p0)) if len(ep_ret_hist_p0) else 0.0
            score = p0_winrate if args.best_metric == "p0_winrate" else p0_rew
            if score > best_score:
                best_score = score
                torch.save(nets["player_0"].state_dict(), best_path)

        # print
        if updates_done % args.print_every_updates == 0:
            fps = int(steps_done / max(1e-6, (time.time() - start)))
            p0_winrate = float(np.mean(win_hist)) if len(win_hist) else 0.0
            p0_rew = float(np.mean(ep_ret_hist_p0)) if len(ep_ret_hist_p0) else 0.0
            avg_len = float(np.mean(ep_len_hist)) if len(ep_len_hist) else 0.0
            sd_rate = float(np.mean(sd_hist)) if len(sd_hist) else 0.0
            draw_rate = float(np.mean(draw_hist)) if len(draw_hist) else 0.0

            mean_pi = float(np.mean([update_metrics[a]["pi_loss"] for a in update_agents]))
            mean_v = float(np.mean([update_metrics[a]["v_loss"] for a in update_agents]))
            mean_ent = float(np.mean([update_metrics[a]["entropy"] for a in update_agents]))
            mean_clip = float(np.mean([update_metrics[a]["clip_frac"] for a in update_agents]))

            print(
                f"[PZ-PPO] upd={updates_done:05d} env_steps={env_steps_done} sample_steps={steps_done} fps={fps} | "
                f"p0_winrate@{args.stats_window}={p0_winrate:.3f} p0_avgR@{args.stats_window}={p0_rew:.3f} "
                f"avg_len@{args.stats_window}={avg_len:.1f} duelRate={sd_rate:.3f} drawRate={draw_rate:.3f} | "
                f"loss: pi={mean_pi:.3f} v={mean_v:.3f} ent={mean_ent:.3f} clip={mean_clip:.3f} | "
                f"best_score={best_score:.3f} fixed_opponents={int(fixed_opponents)}"
            )

    torch.save(nets["player_0"].state_dict(), os.path.join(run_dir, "final_player_0.pt"))
    if not fixed_opponents:
        for a in agents:
            if a == "player_0":
                continue
            torch.save(nets[a].state_dict(), os.path.join(run_dir, f"final_{a}.pt"))

    if args.eval_episodes > 0:
        write_eval_logs(env, obs_dim, act_dim, device, run_dir, args.eval_episodes)

    print("Saved to:", run_dir)
    print("Best saved to:", best_path)
    if args.eval_episodes > 0:
        print("Eval logs:", os.path.join(run_dir, "eval_logs"))


if __name__ == "__main__":
    main()
