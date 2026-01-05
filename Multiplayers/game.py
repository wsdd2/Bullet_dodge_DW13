"""
game.py - 核心游戏逻辑（与 Gymnasium / PettingZoo 解耦）

规则（摘要）：
- N 名玩家（N>=4），每名玩家有生命值 HP、子弹 bullets、闪避次数 dodges
- NORMAL 回合动作（三选一）：
  - RELOAD：装填 1 发子弹
  - DODGE：消耗 1 次闪避，本回合免疫所有伤害
  - ATTACK->target：消耗 1 发子弹，指定攻击某个存活玩家
- 多个玩家可以同时攻击同一个目标；本回合行动先收集后同时结算
- 存活到最后的玩家获胜
- 若本回合结束后所有玩家都阵亡：
  - 若 duel_enabled=True：进入 SUDDEN_DEATH（突然死亡/决斗）
  - 若 duel_enabled=False：本局平局结束

SUDDEN_DEATH（若启用）：
- 参与者：本回合开始时仍存活的玩家（但本回合全灭）
- 进入阶段时参与者状态重置为：HP=1, bullets=0, dodges=1
- 每回合动作：
  - DODGE：消耗 1 次闪避；本回合不能被指定为 DUEL 目标
  - DUEL->target：指定一个未闪避的存活玩家比大小（掷骰 1-6，点数相同重掷，点小者扣 1 血）
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional
import numpy as np


class Mode(IntEnum):
    NORMAL = 0
    SUDDEN_DEATH = 1


class ActionType(IntEnum):
    RELOAD = 0
    DODGE = 1
    ATTACK = 2
    DUEL = 3
    NOOP = 4  # 兜底


@dataclass
class PlayerState:
    alive: bool
    hp: int
    bullets: int
    dodges: int


@dataclass
class RoundAction:
    type: ActionType
    target: int  # 目标玩家 id；无目标时为 none_target (= max_players)


@dataclass
class StepResult:
    mode_before: Mode
    mode_after: Mode
    damage_taken: List[int]
    died_this_round: List[bool]
    winner: Optional[int]          # 赢家 id；平局为 None
    ended: bool                   # 是否结束（赢家或平局）
    sudden_death_started: bool
    round_actions: List[RoundAction]


class BulletDodgeGame:
    """
    只负责状态转移；不负责奖励、不负责对手策略。

    关键点：
    - num_players：本局参与玩家数（真实人数）
    - max_players：动作空间与观测的上限（用于支持同一策略适配不同人数）
    """

    def __init__(
        self,
        num_players: int,
        *,
        max_players: int = 32,
        duel_enabled: bool = True,
        init_hp: int = 2,
        init_bullets: int = 1,
        init_dodges: int = 2,
        max_bullets: int = 10,
        max_dodges: int = 10,
        seed: Optional[int] = None,
    ):
        if num_players < 4:
            raise ValueError("num_players 必须 >= 4")
        if num_players > max_players:
            raise ValueError("num_players 不能超过 max_players")
        self.num_players = int(num_players)
        self.max_players = int(max_players)
        self.duel_enabled = bool(duel_enabled)

        self.init_hp = int(init_hp)
        self.init_bullets = int(init_bullets)
        self.init_dodges = int(init_dodges)
        self.max_bullets = int(max_bullets)
        self.max_dodges = int(max_dodges)

        self.rng = np.random.default_rng(seed)
        self.mode: Mode = Mode.NORMAL
        self.round_idx: int = 0

        self.players: List[PlayerState] = []
        self.last_actions: List[RoundAction] = []
        self.reset(seed=seed)

    # ---------- action encoding ----------

    @property
    def action_n(self) -> int:
        """
        固定动作空间（与 num_players 无关）：
          0  RELOAD
          1  DODGE
          2..(2+max_players-1)                 ATTACK->t
          (2+max_players)..(2+2*max_players-1) DUEL->t
        """
        return 2 + 2 * self.max_players

    @property
    def none_target(self) -> int:
        return self.max_players

    def action_id_to_tuple(self, action_id: int) -> tuple[int, int]:
        if action_id == 0:
            return int(ActionType.RELOAD), self.none_target
        if action_id == 1:
            return int(ActionType.DODGE), self.none_target
        base_attack = 2
        base_duel = 2 + self.max_players
        if base_attack <= action_id < base_attack + self.max_players:
            return int(ActionType.ATTACK), int(action_id - base_attack)
        if base_duel <= action_id < base_duel + self.max_players:
            return int(ActionType.DUEL), int(action_id - base_duel)
        return int(ActionType.NOOP), self.none_target

    def tuple_to_action_id(self, a_type: int, target: int) -> int:
        at = ActionType(int(a_type))
        if at == ActionType.RELOAD:
            return 0
        if at == ActionType.DODGE:
            return 1
        if at == ActionType.ATTACK:
            return 2 + int(target)
        if at == ActionType.DUEL:
            return 2 + self.max_players + int(target)
        return 0

    def decode_action(self, action_id: int) -> RoundAction:
        t, tgt = self.action_id_to_tuple(int(action_id))
        return RoundAction(ActionType(t), int(tgt))

    # ---------- reset / step ----------

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.mode = Mode.NORMAL
        self.round_idx = 0
        self.players = [
            PlayerState(True, self.init_hp, self.init_bullets, self.init_dodges)
            for _ in range(self.num_players)
        ]
        self.last_actions = [RoundAction(ActionType.NOOP, self.none_target) for _ in range(self.num_players)]

    def alive_indices(self) -> List[int]:
        return [i for i, p in enumerate(self.players) if p.alive]

    def is_action_valid(self, act: RoundAction, actor: int, dodging_flags: Optional[List[bool]] = None) -> bool:
        """
        dodging_flags：仅在 SUDDEN_DEATH 下用于判断目标是否闪避（闪避者不可被 DUEL 指定）
        """
        if actor < 0 or actor >= self.num_players:
            return False
        p = self.players[actor]
        if not p.alive:
            return False

        if self.mode == Mode.NORMAL:
            if act.type == ActionType.RELOAD:
                return True
            if act.type == ActionType.DODGE:
                return p.dodges > 0
            if act.type == ActionType.ATTACK:
                t = act.target
                if p.bullets <= 0:
                    return False
                if t < 0 or t >= self.num_players:
                    return False
                if t == actor:
                    return False
                return self.players[t].alive
            return False

        # SUDDEN_DEATH
        if act.type == ActionType.DODGE:
            return p.dodges > 0
        if act.type == ActionType.DUEL:
            t = act.target
            if t < 0 or t >= self.num_players:
                return False
            if t == actor:
                return False
            if not self.players[t].alive:
                return False
            if dodging_flags is not None and dodging_flags[t]:
                return False
            return True
        return False

    def step(self, action_ids: List[int]) -> StepResult:
        if len(action_ids) != self.num_players:
            raise ValueError("action_ids 长度必须等于 num_players")

        mode_before = self.mode
        self.round_idx += 1

        alive_at_start = [p.alive for p in self.players]

        # decode & sanitize
        actions: List[RoundAction] = []
        for i in range(self.num_players):
            act = self.decode_action(int(action_ids[i]))
            if not self.is_action_valid(act, i):
                act = RoundAction(ActionType.NOOP, self.none_target)
            actions.append(act)

        damage_taken = [0 for _ in range(self.num_players)]
        died = [False for _ in range(self.num_players)]
        winner: Optional[int] = None
        ended = False
        sudden_death_started = False

        if self.mode == Mode.NORMAL:
            # apply DODGE first
            dodging = [False] * self.num_players
            for i, act in enumerate(actions):
                if act.type == ActionType.DODGE and self.players[i].alive and self.players[i].dodges > 0:
                    self.players[i].dodges -= 1
                    self.players[i].dodges = max(0, min(self.players[i].dodges, self.max_dodges))
                    dodging[i] = True

            # apply RELOAD
            for i, act in enumerate(actions):
                if act.type == ActionType.RELOAD and self.players[i].alive:
                    self.players[i].bullets += 1
                    self.players[i].bullets = max(0, min(self.players[i].bullets, self.max_bullets))

            # collect ATTACK events
            attacks: List[tuple[int, int]] = []
            for i, act in enumerate(actions):
                if act.type == ActionType.ATTACK and self.players[i].alive:
                    t = act.target
                    if 0 <= t < self.num_players and t != i and self.players[i].bullets > 0:
                        self.players[i].bullets -= 1
                        self.players[i].bullets = max(0, min(self.players[i].bullets, self.max_bullets))
                        attacks.append((i, t))

            # apply damage simultaneously (count hits on each target)
            hit_count = [0] * self.num_players
            for _, t in attacks:
                hit_count[t] += 1

            for t in range(self.num_players):
                if not self.players[t].alive:
                    continue
                if dodging[t]:
                    continue
                if hit_count[t] > 0:
                    damage_taken[t] += hit_count[t]
                    self.players[t].hp -= hit_count[t]
                    if self.players[t].hp <= 0:
                        self.players[t].alive = False
                        died[t] = True

            alive_after = self.alive_indices()
            if len(alive_after) == 1:
                winner = alive_after[0]
                ended = True
            elif len(alive_after) == 0:
                # 全灭：按开关进入 SUDDEN_DEATH 或平局结束
                if self.duel_enabled:
                    participants = [i for i, a in enumerate(alive_at_start) if a]
                    for i in participants:
                        self.players[i].alive = True
                        self.players[i].hp = 1
                        self.players[i].bullets = 0
                        self.players[i].dodges = 1
                    self.mode = Mode.SUDDEN_DEATH
                    sudden_death_started = True
                else:
                    winner = None
                    ended = True

        else:
            # SUDDEN_DEATH
            alive = self.alive_indices()
            if len(alive) <= 1:
                winner = alive[0] if len(alive) == 1 else None
                ended = True
            else:
                # process DODGE
                dodging = [False] * self.num_players
                for i, act in enumerate(actions):
                    if act.type == ActionType.DODGE and self.players[i].alive and self.players[i].dodges > 0:
                        self.players[i].dodges -= 1
                        self.players[i].dodges = max(0, min(self.players[i].dodges, self.max_dodges))
                        dodging[i] = True

                # challengers (alive + not dodging + DUEL)
                challengers = [i for i in alive if (not dodging[i] and actions[i].type == ActionType.DUEL)]
                self.rng.shuffle(challengers)

                paired = set()
                for i in challengers:
                    if i in paired or not self.players[i].alive or dodging[i]:
                        continue
                    j = actions[i].target
                    if j in paired or j == self.none_target:
                        continue
                    if (j < 0 or j >= self.num_players) or (not self.players[j].alive) or dodging[j]:
                        continue
                    if j == i:
                        continue

                    paired.add(i)
                    paired.add(j)

                    # roll until different
                    while True:
                        di = int(self.rng.integers(1, 7))
                        dj = int(self.rng.integers(1, 7))
                        if di != dj:
                            break
                        else:
                            time.sleep(0.5)
                            print(f"点数相同，都是{di}，继续投掷骰子")
                    loser = i if di < dj else j
                    print(f"玩家P{i} 点数为{di}, 玩家P{j} 点数为{dj}")
                    import time
                    if loser == i:
                        print(f"玩家P{j} 决斗获胜")
                        time.sleep(2)
                    else:
                        print(f"玩家P{i} 决斗获胜")
                        time.sleep(2)
                    self.players[loser].hp -= 1
                    damage_taken[loser] += 1
                    if self.players[loser].hp <= 0:
                        self.players[loser].alive = False
                        died[loser] = True

                alive_after = self.alive_indices()
                if len(alive_after) == 1:
                    winner = alive_after[0]
                    ended = True
                elif len(alive_after) == 0:
                    winner = None
                    ended = True

        self.last_actions = actions
        return StepResult(
            mode_before=mode_before,
            mode_after=self.mode,
            damage_taken=damage_taken,
            died_this_round=died,
            winner=winner,
            ended=ended,
            sudden_death_started=sudden_death_started,
            round_actions=actions,
        )

    # ---------- observation encoding ----------

    def get_obs(self) -> np.ndarray:
        """
        返回 shape=(max_players, 7) 的观测（固定维度）：
          per-player:
            [active(0/1), alive(0/1), hp, bullets, dodges, last_action_type, last_action_target_or_none]
        inactive 的玩家（i>=num_players）：
            active=0, alive=0，其余为 0 / NOOP
        last_action_target_or_none 无目标时为 max_players
        """
        obs = np.zeros((self.max_players, 7), dtype=np.int32)
        for i in range(self.max_players):
            if i < self.num_players:
                p = self.players[i]
                obs[i, 0] = 1
                obs[i, 1] = 1 if p.alive else 0
                obs[i, 2] = int(p.hp)
                obs[i, 3] = int(p.bullets)
                obs[i, 4] = int(p.dodges)
                if i < len(self.last_actions):
                    obs[i, 5] = int(self.last_actions[i].type)
                    obs[i, 6] = int(self.last_actions[i].target)
                else:
                    obs[i, 5] = int(ActionType.NOOP)
                    obs[i, 6] = int(self.none_target)
            else:
                obs[i, 0] = 0
                obs[i, 1] = 0
                obs[i, 2] = 0
                obs[i, 3] = 0
                obs[i, 4] = 0
                obs[i, 5] = int(ActionType.NOOP)
                obs[i, 6] = int(self.none_target)
        return obs
