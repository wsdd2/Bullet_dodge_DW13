"""
bullet_dodge_pz.py - PettingZoo ParallelEnv 封装（同步同时行动）

特点：
- 同步同时行动：所有玩家提交动作后统一结算
- 固定动作空间与固定观测维度：支持同一策略适配不同玩家人数
- action_mask：无效动作自动屏蔽（可直接喂给 PPO）

动作空间（Discrete(action_n)）与游戏内部一致：
  0  RELOAD
  1  DODGE
  2..(2+max_players-1)                 ATTACK->t
  (2+max_players)..(2+2*max_players-1) DUEL->t
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
try:
    from gymnasium import spaces
except ImportError:  # fallback
    import gym
    spaces = gym.spaces
from pettingzoo.utils.env import ParallelEnv

from game import BulletDodgeGame, Mode, ActionType, RoundAction


class BulletDodgePZEnv(ParallelEnv):
    metadata = {"name": "bullet_dodge_pz_v4"}

    def __init__(
        self,
        *,
        num_players: int = 10,
        max_players: int = 32,
        duel_enabled: bool = True,
        init_hp: int = 2,
        init_bullets: int = 1,
        init_dodges: int = 2,
        max_rounds: int = 200,
        seed: int = 0,
        # reward shaping
        win_bonus: float = 10.0,
        top2_bonus: float = 3.0,
        sudden_death_bonus: float = 0.5,
        duel_win_bonus: float = 0.5,
        step_penalty: float = -0.01,
        death_penalty: float = -1.0,
    ):
        if num_players < 4:
            raise ValueError("num_players 必须 >= 4")
        if num_players > max_players:
            raise ValueError("num_players 不能超过 max_players")

        self.num_players = int(num_players)
        self.max_players = int(max_players)
        self.duel_enabled = bool(duel_enabled)

        self.max_rounds = int(max_rounds)
        self._seed = int(seed)

        self.win_bonus = float(win_bonus)
        self.top2_bonus = float(top2_bonus)
        self.sudden_death_bonus = float(sudden_death_bonus)
        self.duel_win_bonus = float(duel_win_bonus)
        self.step_penalty = float(step_penalty)
        self.death_penalty = float(death_penalty)

        self.game = BulletDodgeGame(
            num_players=self.num_players,
            max_players=self.max_players,
            duel_enabled=self.duel_enabled,
            init_hp=int(init_hp),
            init_bullets=int(init_bullets),
            init_dodges=int(init_dodges),
            seed=self._seed,
        )

        self.possible_agents = [f"player_{i}" for i in range(self.num_players)]
        self.agents = self.possible_agents[:]

        # fixed action/obs space
        self.action_n = int(self.game.action_n)
        self.action_spaces = {a: spaces.Discrete(self.action_n) for a in self.possible_agents}

        # observation: flatten padded per-player obs + globals + self_id
        # per-player features = 7; padded to max_players
        self.per_dim = 7
        self.glob_dim = 4  # mode, round_idx, duel_enabled, num_players
        self.obs_dim = self.max_players * self.per_dim + self.glob_dim + 1  # + self_id
        self.observation_spaces = {
            a: spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
            for a in self.possible_agents
        }

        self._hi = self._build_hi(
            init_hp=int(init_hp),
            max_bullets=int(self.game.max_bullets),
            max_dodges=int(self.game.max_dodges),
        )

        # tracking
        self._round: int = 0
        self._terminated: bool = False
        self._truncated: bool = False
        self._death_round: List[Optional[int]] = [None for _ in range(self.num_players)]

    # --------- exposed helpers for play ---------

    def action_id_to_tuple(self, action_id: int) -> Tuple[int, int]:
        return self.game.action_id_to_tuple(int(action_id))

    def tuple_to_action_id(self, a_type: int, target: int) -> int:
        return int(self.game.tuple_to_action_id(int(a_type), int(target)))

    # --------- spaces ---------

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    # --------- core ---------

    def _build_hi(self, init_hp: int, max_bullets: int, max_dodges: int) -> np.ndarray:
        """
        raw obs 由整数构成，这里提供逐元素归一化上界。
        per-player:
          active(1), alive(1), hp(init_hp), bullets(max_bullets), dodges(max_dodges),
          last_action_type(max(ActionType)), last_action_target(max_players)
        globals:
          mode(1), round_idx(max_rounds), duel_enabled(1), num_players(max_players)
        self_id(1)
        """
        per_hi = []
        per_hi.extend([1, 1, max(1, int(init_hp)), max(1, int(max_bullets)), max(1, int(max_dodges)),
                      int(max(ActionType)), int(self.max_players)])
        per_hi = np.array(per_hi, dtype=np.float32)  # len=7

        hi = np.concatenate([np.tile(per_hi, (self.max_players,)),  # max_players*7
                             np.array([1, max(1, self.max_rounds), 1, self.max_players], dtype=np.float32),
                             np.array([1], dtype=np.float32)], axis=0)
        return hi

    def _agent_index(self, agent: str) -> int:
        return int(agent.split("_")[1])

    def _build_raw_obs(self) -> np.ndarray:
        per = self.game.get_obs().astype(np.float32).reshape(-1)  # max_players*7
        glob = np.array([
            float(self.game.mode),
            float(self.game.round_idx),
            1.0 if self.duel_enabled else 0.0,
            float(self.num_players),
        ], dtype=np.float32)
        return np.concatenate([per, glob], axis=0)

    def _obs_for_agent(self, agent: str) -> np.ndarray:
        raw = self._build_raw_obs()
        idx = self._agent_index(agent)
        self_id = 0.0 if self.max_players <= 1 else float(idx) / float(self.max_players - 1)
        raw = np.concatenate([raw, np.array([self_id], dtype=np.float32)], axis=0)
        obs = np.clip(raw / self._hi, 0.0, 1.0).astype(np.float32)
        return obs

    def _action_mask_for_agent(self, agent: str) -> np.ndarray:
        i = self._agent_index(agent)
        mask = np.zeros((self.action_n,), dtype=np.int8)
        for aid in range(self.action_n):
            act = self.game.decode_action(aid)
            if self.game.is_action_valid(act, i):
                # duel disabled: still mask DUEL actions
                if (not self.duel_enabled) and (act.type == ActionType.DUEL):
                    continue
                mask[aid] = 1
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
        """
        second place = 最后阵亡的那一批（不含 winner）
        """
        death = self._death_round[:]
        # winner not dead => death_round None; assign +inf for sorting
        max_r = self._round + 1
        for i in range(self.num_players):
            if i == winner:
                death[i] = max_r
            elif death[i] is None:
                death[i] = max_r

        # second max among non-winner
        vals = [death[i] for i in range(self.num_players) if i != winner]
        if not vals:
            return []
        second_val = max(vals)
        return [i for i in range(self.num_players) if i != winner and death[i] == second_val]

    def step(self, actions: Dict[str, int]):
        if self._terminated or self._truncated:
            obs = {a: self._obs_for_agent(a) for a in self.agents}
            rewards = {a: 0.0 for a in self.agents}
            terms = {a: True for a in self.agents}
            truncs = {a: False for a in self.agents}
            infos = {a: {"action_mask": self._action_mask_for_agent(a)} for a in self.agents}
            return obs, rewards, terms, truncs, infos

        action_ids = [0 for _ in range(self.num_players)]
        for a in self.agents:
            i = self._agent_index(a)
            action_ids[i] = int(actions.get(a, 0))

        result = self.game.step(action_ids)
        self._round = int(self.game.round_idx)

        for i, d in enumerate(result.died_this_round):
            if d and self._death_round[i] is None:
                self._death_round[i] = self._round

        if result.sudden_death_started:
            self._death_round = [None for _ in range(self.num_players)]

        rewards = {a: self.step_penalty for a in self.agents}

        # death penalty
        for i, d in enumerate(result.died_this_round):
            if d:
                rewards[f"player_{i}"] += self.death_penalty

        # sudden death started bonus
        if result.sudden_death_started:
            for a in self.agents:
                i = self._agent_index(a)
                if self.game.players[i].alive:
                    rewards[a] += self.sudden_death_bonus

        # duel win bonus (sudden death mode after step)
        if result.mode_after == Mode.SUDDEN_DEATH:
            for i in range(self.num_players):
                ra = result.round_actions[i]
                if int(ra.type) == int(ActionType.DUEL):
                    t = int(ra.target)
                    if 0 <= t < self.num_players:
                        if result.died_this_round[t] and (not result.died_this_round[i]):
                            rewards[f"player_{i}"] += self.duel_win_bonus

        winner = result.winner
        self._terminated = bool(result.ended)
        self._truncated = (not self._terminated) and (self._round >= self.max_rounds)

        terms = {a: self._terminated for a in self.agents}
        truncs = {a: self._truncated for a in self.agents}

        # terminal ranking rewards: only when winner exists
        if self._terminated and (winner is not None) and (winner >= 0):
            rewards[f"player_{winner}"] += self.win_bonus
            seconds = self._compute_second_place_set(int(winner))
            for i in seconds:
                rewards[f"player_{i}"] += self.top2_bonus

        obs = {a: self._obs_for_agent(a) for a in self.agents}
        infos = {
            a: {
                "action_mask": self._action_mask_for_agent(a),
                "winner": winner,
                "ended": bool(result.ended),
                "round": self._round,
                "mode": int(self.game.mode),
                "round_actions": [(int(x.type), int(x.target)) for x in result.round_actions],
                "died_this_round": result.died_this_round,
                "sudden_death_started": bool(result.sudden_death_started),
            }
            for a in self.agents
        }
        return obs, rewards, terms, truncs, infos

    def close(self):
        return


def parallel_env(
    *,
    num_players: int = 10,
    max_players: int = 32,
    duel_enabled: bool = True,
    init_hp: int = 2,
    init_bullets: int = 1,
    init_dodges: int = 2,
    max_rounds: int = 200,
    seed: int = 0,
    **kwargs,
) -> BulletDodgePZEnv:
    return BulletDodgePZEnv(
        num_players=num_players,
        max_players=max_players,
        duel_enabled=duel_enabled,
        init_hp=init_hp,
        init_bullets=init_bullets,
        init_dodges=init_dodges,
        max_rounds=max_rounds,
        seed=seed,
        **kwargs,
    )
