"""
bullet_dodge_pz.py - PettingZoo ParallelEnv 封装（同步同时行动）

特点：
- 同步 simultaneous move：每回合所有玩家提交动作后，同时结算
- fully observable：每个玩家看到所有玩家（hp/bullets/dodges/上一回合动作）+ 全局 mode/round
- action mask：infos[agent]["action_mask"] 给出合法动作（强烈建议 PPO 采样时使用）

依赖：
  pip install pettingzoo gymnasium numpy

说明：
- 这里采用“固定 agents 集合，不在中途移除死亡玩家”的方式，避免突然死亡复活带来的 agent 进出问题。
  死亡玩家在 game 内 alive=False，会导致其动作无效（NOOP），观察中也会体现 alive=0。
- episode 终止：game.winner != None 或 round >= max_rounds（truncation）

奖励塑形（每个 agent 单独计算）：
- sudden_death_started：参与者每人 +sudden_death_bonus
<<<<<<< HEAD
- duel_kill（突然死亡阶段你 DUEL 指定的目标当回合死亡且你没死） +duel_win_bonus
=======
- duel_kill（突然死亡阶段玩家 DUEL 指定的目标当回合死亡且自己没死） +duel_win_bonus
>>>>>>> 49efced (merge remote and local)
- terminal：冠军 +win_bonus；第二名（按 death_round 推断） +top2_bonus（第一名不额外给 top2）
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import gymnasium as gym

from pettingzoo.utils.env import ParallelEnv

from game import BulletDodgeGame, Mode, ActionType


class BulletDodgeParallelEnv(ParallelEnv):
    metadata = {"name": "bullet_dodge_v0"}

    def __init__(
        self,
        num_players: int = 6,
        init_hp: int = 3,
        max_rounds: int = 200,
        max_bullets: int = 10,
        max_dodges: int = 10,
        seed: Optional[int] = None,
        # reward shaping
        win_bonus: float = 5.0,
        top2_bonus: float = 1.0,
        sudden_death_bonus: float = 0.5,
        duel_win_bonus: float = 2.0,
    ):
        if not (4 <= num_players <= 8):
            raise ValueError("num_players must be in [4, 8]")
        self.num_players = int(num_players)
        self.init_hp = int(init_hp)
        self.max_rounds = int(max_rounds)
        self.max_bullets = int(max_bullets)
        self.max_dodges = int(max_dodges)

        self.win_bonus = float(win_bonus)
        self.top2_bonus = float(top2_bonus)
        self.sudden_death_bonus = float(sudden_death_bonus)
        self.duel_win_bonus = float(duel_win_bonus)

        self.possible_agents = [f"player_{i}" for i in range(self.num_players)]
        self.agents = self.possible_agents[:]  # active agents (we keep constant until done)

        # Discrete action: 0 LOAD, 1 DODGE, 2 TRADE, 3.. target (ATTACK or DUEL)
        self.action_n = 3 + (self.num_players - 1)
        self.action_spaces = {a: gym.spaces.Discrete(self.action_n) for a in self.possible_agents}

        # Observation: normalized float vector in [0,1]
        # raw features: num_players*6 + 2 (mode, round) + 1 (self_id_norm)
        self.obs_dim = self.num_players * 6 + 2 + 1
        self.observation_spaces = {
            a: gym.spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
            for a in self.possible_agents
        }

        self._seed = seed
        self.game = BulletDodgeGame(
            num_players=self.num_players,
            init_hp=self.init_hp,
            init_bullets=1,
            init_dodges=1,
            max_bullets=self.max_bullets,
            max_dodges=self.max_dodges,
            seed=seed,
        )

        # normalization denominators (match env high bounds idea)
        # per-player: [alive(1), hp(init_hp), bullets(max_bullets), dodges(max_dodges), last_action_type(5), target(num_players)]
        per_hi = np.array([1, max(self.init_hp, 1), self.max_bullets, self.max_dodges, int(ActionType.NOOP), self.num_players], dtype=np.float32)
        self._hi = np.concatenate(
            [np.tile(per_hi, self.num_players), np.array([int(Mode.SUDDEN_DEATH), self.max_rounds + 5, 1.0], dtype=np.float32)],
            axis=0,
        )
        self._hi = np.maximum(self._hi, 1.0)

        # ranking helper
        self._death_round: List[Optional[int]] = [None for _ in range(self.num_players)]
        self._round: int = 0
        self._terminated: bool = False
        self._truncated: bool = False

    def seed(self, seed: Optional[int] = None):
        self._seed = seed

    def _agent_index(self, agent: str) -> int:
        return int(agent.split("_")[1])

    def _build_raw_obs(self) -> np.ndarray:
        per = self.game.get_obs().astype(np.float32).reshape(-1)  # num_players*6
        glob = np.array([float(self.game.mode), float(self.game.round_idx)], dtype=np.float32)
        return np.concatenate([per, glob], axis=0)

    def _obs_for_agent(self, agent: str) -> np.ndarray:
        raw = self._build_raw_obs()
        idx = self._agent_index(agent)
        self_id = 0.0 if self.num_players == 1 else float(idx) / float(self.num_players - 1)
        raw = np.concatenate([raw, np.array([self_id], dtype=np.float32)], axis=0)
        obs = np.clip(raw / self._hi, 0.0, 1.0).astype(np.float32)
        return obs

    def _action_mask_for_agent(self, agent: str) -> np.ndarray:
        i = self._agent_index(agent)
        mask = np.zeros((self.action_n,), dtype=np.int8)
        for a in range(self.action_n):
            act = self.game.decode_action(a, i)
            if self.game.is_action_valid(act, i):
                mask[a] = 1
        return mask

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is None:
            seed = self._seed
        self.game.reset(seed=seed)
        self.agents = self.possible_agents[:]
        self._death_round = [None for _ in range(self.num_players)]
        self._round = 0
        self._terminated = False
        self._truncated = False

        obs = {a: self._obs_for_agent(a) for a in self.agents}
        infos = {a: {"action_mask": self._action_mask_for_agent(a)} for a in self.agents}
        return obs, infos

    def _compute_second_place_set(self, winner: int) -> List[int]:
        # based on recorded death rounds: larger = died later = better
        dr = []
        for i in range(self.num_players):
            if i == winner:
                continue
            v = self._death_round[i]
            # None means never recorded (could happen if truncation); treat as very small
            dr.append((i, -1 if v is None else int(v)))
        if not dr:
            return []
        max_r = max(v for _, v in dr)
        return [i for i, v in dr if v == max_r]

    def step(self, actions: Dict[str, int]):
        if self._terminated or self._truncated:
            # pettingzoo convention: after done, call reset
            obs = {a: self._obs_for_agent(a) for a in self.agents}
            rewards = {a: 0.0 for a in self.agents}
            terms = {a: True for a in self.agents}
            truncs = {a: False for a in self.agents}
            infos = {a: {"action_mask": self._action_mask_for_agent(a)} for a in self.agents}
            return obs, rewards, terms, truncs, infos

        # ensure all agents have an action
        action_ids = [0 for _ in range(self.num_players)]
        for a in self.agents:
            i = self._agent_index(a)
            act = int(actions.get(a, 0))
            action_ids[i] = act

        result = self.game.step(action_ids)
        self._round = int(self.game.round_idx)

        # update death rounds (within current life-stage)
        for i, d in enumerate(result.died_this_round):
            if d and self._death_round[i] is None:
                self._death_round[i] = self._round

        # sudden death resets ranking trace (everyone revived)
        if result.sudden_death_started:
            self._death_round = [None for _ in range(self.num_players)]

        rewards = {a: 0.0 for a in self.agents}

        # sudden death started reward for participants: those alive at start of that round
        if result.sudden_death_started:
            # participants in our game impl are "alive at start of round", which were those not yet dead previously.
            # In this wrapper, everyone was dead after resolution, then revived; easiest: reward all agents that are alive now.
            for a in self.agents:
                i = self._agent_index(a)
                if self.game.players[i].alive:
                    rewards[a] += self.sudden_death_bonus

        # duel win bonus: sudden death mode after step, your action is DUEL and your target died this round while you survived
        if result.mode_after == Mode.SUDDEN_DEATH:
            for a in self.agents:
                i = self._agent_index(a)
                ra = result.round_actions[i]
                if int(ra.type) == int(ActionType.DUEL):
                    t = int(ra.target)
                    if 0 <= t < self.num_players:
                        if result.died_this_round[t] and (not result.died_this_round[i]):
                            rewards[a] += self.duel_win_bonus

        winner = result.winner
        self._terminated = winner is not None
        self._truncated = (not self._terminated) and (self._round >= self.max_rounds)

        terms = {a: self._terminated for a in self.agents}
        truncs = {a: self._truncated for a in self.agents}

        # terminal ranking rewards
        if self._terminated and winner is not None:
            w = int(winner)
            # winner big bonus
            rewards[f"player_{w}"] += self.win_bonus
            # second place bonus
            seconds = self._compute_second_place_set(w)
            for i in seconds:
                rewards[f"player_{i}"] += self.top2_bonus

        obs = {a: self._obs_for_agent(a) for a in self.agents}
        infos = {a: {"action_mask": self._action_mask_for_agent(a),
                    "winner": winner,
                    "round": self._round,
                    "mode": int(self.game.mode),
                    "round_actions": [(int(x.type), int(x.target)) for x in result.round_actions],
                    "died_this_round": result.died_this_round,
                    "sudden_death_started": bool(result.sudden_death_started)} for a in self.agents}

        if self._terminated or self._truncated:
            # keep agents list until next reset; some libraries expect agents cleared, but ParallelEnv allows keeping.
            pass

        return obs, rewards, terms, truncs, infos

    def render(self):
        lines = []
        lines.append(f"Round {self.game.round_idx} | Mode={self.game.mode.name}")
        for i, p in enumerate(self.game.players):
            la = self.game.last_actions[i]
            lines.append(
                f"P{i} alive={int(p.alive)} hp={p.hp} b={p.bullets} d={p.dodges} "
                f"last=({ActionType(la.type).name},{la.target if la.target!=self.num_players else '-'})"
            )
        print("\n".join(lines))

    def close(self):
        pass


def parallel_env(**kwargs):
    return BulletDodgeParallelEnv(**kwargs)
