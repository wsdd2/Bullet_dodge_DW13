"""
game.py - 核心游戏逻辑（与 Gymnasium 解耦）

游戏（工作定义，方便 RL 先跑起来）：
- N 名玩家（4~8），默认初始 hp=3, bullets=1, dodges=1
- 正常模式每回合动作（同时结算）：
    0 LOAD   : 装填 +1 子弹
    1 DODGE  : 本回合闪避（消耗 1 次闪避，本回合免疫所有伤害）
    2 TRADE  : 2 发子弹换 1 次闪避（bullets -=2, dodges +=1）
    3..      : ATTACK 指定攻击某玩家（消耗 1 发子弹，目标若未闪避则 -1 hp；可被多人同时攻击叠加）
- 终局：
    - 若回合结算后只剩 1 人存活 => 胜者
    - 若回合结算后存活人数变为 0（通常是最后几名互射同归于尽）=> 进入突然死亡模式：
        * 参与者：该回合开始时存活的那些人
        * 每人重置为 hp=1, dodges=1, bullets=0
        * 每回合可 DODGE 或 DUEL（比大小）
        * 本实现中 DUEL 的配对规则：按随机顺序处理挑战 i->j，
          若 i 和 j 都未闪避且还未被配对，则配成一对进行一次掷骰决斗；
          点数相同继续掷，较小者扣 1 hp（= 死亡）
        * 若一回合后仍多人存活，继续突然死亡
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Dict, Tuple
import numpy as np


class Mode(IntEnum):
    NORMAL = 0
    SUDDEN_DEATH = 1


class ActionType(IntEnum):
    LOAD = 0
    DODGE = 1
    TRADE = 2
    ATTACK = 3
    DUEL = 4
    NOOP = 5  # 用于无效动作兜底


@dataclass
class PlayerState:
    hp: int
    bullets: int
    dodges: int
    alive: bool = True


@dataclass
class RoundAction:
    type: ActionType
    target: int  # 0..num_players-1 或者 num_players 表示无目标


@dataclass
class StepResult:
    mode_before: Mode
    mode_after: Mode
    damage_taken: List[int]
    died_this_round: List[bool]
    winner: Optional[int]  # None 表示尚未结束
    sudden_death_started: bool
    round_actions: List[RoundAction]


class BulletDodgeGame:
    """
    只负责状态转移；不负责奖励、不负责对手策略。
    """

    def __init__(
        self,
        num_players: int,
        init_hp: int = 3,
        init_bullets: int = 1,
        init_dodges: int = 1,
        max_bullets: int = 10,
        max_dodges: int = 10,
        seed: Optional[int] = None,
    ):
        if not (4 <= num_players <= 8):
            raise ValueError("num_players 必须在 [4, 8]")
        self.num_players = num_players
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

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.mode = Mode.NORMAL
        self.round_idx = 0
        self.players = [
            PlayerState(
                hp=self.init_hp,
                bullets=self.init_bullets,
                dodges=self.init_dodges,
                alive=True,
            )
            for _ in range(self.num_players)
        ]
        self.last_actions = [RoundAction(ActionType.NOOP, self.num_players) for _ in range(self.num_players)]

    def alive_indices(self) -> List[int]:
        return [i for i, p in enumerate(self.players) if p.alive]

    def is_terminal(self) -> bool:
        return len(self.alive_indices()) <= 1

    def winner(self) -> Optional[int]:
        alive = self.alive_indices()
        if len(alive) == 1:
            return alive[0]
        return None

    # ---------- action decode / validation ----------

    def decode_action(self, action_id: int, actor: int) -> RoundAction:
        """
        Gym 侧用统一 Discrete(action_n)；这里把 action_id 映射成带 target 的结构化动作。
        action_id 设计（对每个 actor 都一致）：
          0 LOAD
          1 DODGE
          2 TRADE
          3..(3+num_players-2)  ATTACK 或 DUEL（取决于 mode），目标是“除了自己以外的第 k 个玩家”
        """
        none_t = self.num_players
        if action_id == 0:
            return RoundAction(ActionType.LOAD, none_t)
        if action_id == 1:
            return RoundAction(ActionType.DODGE, none_t)
        if action_id == 2:
            return RoundAction(ActionType.TRADE, none_t)

        k = action_id - 3
        targets = [i for i in range(self.num_players) if i != actor]
        if 0 <= k < len(targets):
            t = targets[k]
            if self.mode == Mode.NORMAL:
                return RoundAction(ActionType.ATTACK, t)
            else:
                return RoundAction(ActionType.DUEL, t)

        return RoundAction(ActionType.NOOP, none_t)

    def is_action_valid(self, act: RoundAction, actor: int) -> bool:
        p = self.players[actor]
        if not p.alive:
            return False

        if self.mode == Mode.NORMAL:
            if act.type == ActionType.LOAD:
                return True
            if act.type == ActionType.DODGE:
                return p.dodges > 0
            if act.type == ActionType.TRADE:
                return p.bullets >= 2
            if act.type == ActionType.ATTACK:
                if p.bullets <= 0:
                    return False
                if act.target == actor:
                    return False
                return self.players[act.target].alive
            return False
        else:
            # Sudden death: 只能 DODGE 或 DUEL
            if act.type == ActionType.DODGE:
                return p.dodges > 0
            if act.type == ActionType.DUEL:
                if act.target == actor:
                    return False
                return self.players[act.target].alive
            return False

    # ---------- step transition ----------

    def step(self, action_ids: List[int]) -> StepResult:
        """
        输入：每个玩家一个 action_id（长度=num_players）
        输出：本回合结算结果（包含是否进入突然死亡、谁赢等）
        """
        if len(action_ids) != self.num_players:
            raise ValueError("action_ids 长度必须等于 num_players")

        self.round_idx += 1
        mode_before = self.mode

        actions = [self.decode_action(aid, i) for i, aid in enumerate(action_ids)]
        # 无效动作 -> NOOP（并在结算中自然无效）
        for i, act in enumerate(actions):
            if not self.is_action_valid(act, i):
                actions[i] = RoundAction(ActionType.NOOP, self.num_players)

        damage_taken = [0 for _ in range(self.num_players)]
        died = [False for _ in range(self.num_players)]
        sudden_death_started = False
        winner: Optional[int] = None

        if self.mode == Mode.NORMAL:
            # 先处理 DODGE 标记
            dodging = [False] * self.num_players
            for i, act in enumerate(actions):
                if act.type == ActionType.DODGE and self.players[i].alive:
                    # 该动作已验证合法
                    self.players[i].dodges -= 1
                    dodging[i] = True
                    self.players[i].dodges = max(0, min(self.players[i].dodges, self.max_dodges))

            # 处理 LOAD / TRADE 资源变化（攻击消耗在后面做）
            for i, act in enumerate(actions):
                if not self.players[i].alive:
                    continue
                if act.type == ActionType.LOAD:
                    self.players[i].bullets = min(self.max_bullets, self.players[i].bullets + 1)
                elif act.type == ActionType.TRADE:
                    self.players[i].bullets -= 2
                    self.players[i].dodges = min(self.max_dodges, self.players[i].dodges + 1)

            # 处理 ATTACK
            for i, act in enumerate(actions):
                if act.type != ActionType.ATTACK:
                    continue
                if not self.players[i].alive:
                    continue
                # act 已验证 target alive 且 bullets>0
                self.players[i].bullets -= 1
                self.players[i].bullets = max(0, min(self.players[i].bullets, self.max_bullets))

                t = act.target
                if t < 0 or t >= self.num_players:
                    continue
                if not self.players[t].alive:
                    continue
                if dodging[t]:
                    continue
                damage_taken[t] += 1

            # 应用伤害
            alive_at_start = [p.alive for p in self.players]
            for i, dmg in enumerate(damage_taken):
                if not self.players[i].alive:
                    continue
                if dmg > 0:
                    self.players[i].hp -= dmg
                    if self.players[i].hp <= 0:
                        self.players[i].alive = False
                        died[i] = True

            alive_after = self.alive_indices()
            if len(alive_after) == 1:
                winner = alive_after[0]
            elif len(alive_after) == 0:
                # 进入突然死亡：参与者=本回合开始时还活着的人
                participants = [i for i, a in enumerate(alive_at_start) if a]
                # 重置他们
                for i in participants:
                    self.players[i].alive = True
                    self.players[i].hp = 1
                    self.players[i].bullets = 0
                    self.players[i].dodges = 1
                self.mode = Mode.SUDDEN_DEATH
                sudden_death_started = True

        else:
            # Sudden death
            alive = self.alive_indices()
            if len(alive) <= 1:
                winner = alive[0] if len(alive) == 1 else None
            else:
                dodging = [False] * self.num_players
                for i, act in enumerate(actions):
                    if act.type == ActionType.DODGE and self.players[i].alive:
                        self.players[i].dodges -= 1
                        dodging[i] = True
                        self.players[i].dodges = max(0, min(self.players[i].dodges, self.max_dodges))

                # 收集挑战（只考虑活着且未闪避的人）
                challengers = [i for i in alive if (not dodging[i] and actions[i].type == ActionType.DUEL)]
                self.rng.shuffle(challengers)

                paired = set()
                # 记录哪些 pair 决斗了
                for i in challengers:
                    if i in paired or not self.players[i].alive or dodging[i]:
                        continue
                    j = actions[i].target
                    if j in paired or j == self.num_players:
                        continue
                    if (j < 0 or j >= self.num_players) or (not self.players[j].alive) or dodging[j]:
                        continue
                    if j == i:
                        continue
                    # 配对决斗一次
                    paired.add(i)
                    paired.add(j)

                    # 掷骰直到不同
                    while True:
                        di = int(self.rng.integers(1, 7))
                        dj = int(self.rng.integers(1, 7))
                        if di != dj:
                            break
                    loser = i if di < dj else j
                    self.players[loser].hp -= 1
                    if self.players[loser].hp <= 0:
                        self.players[loser].alive = False
                        died[loser] = True

                alive_after = self.alive_indices()
                if len(alive_after) == 1:
                    winner = alive_after[0]

        self.last_actions = actions
        mode_after = self.mode
        return StepResult(
            mode_before=mode_before,
            mode_after=mode_after,
            damage_taken=damage_taken,
            died_this_round=died,
            winner=winner,
            sudden_death_started=sudden_death_started,
            round_actions=actions,
        )

    # ---------- observation encoding ----------

    def get_obs(self) -> np.ndarray:
        """
        返回 shape=(num_players, 6) 的观测，再由 env 决定是否 flatten。
        每个玩家特征：
          [alive(0/1), hp, bullets, dodges, last_action_type, last_action_target_or_none]
        其中 last_action_target_or_none 在无目标时为 num_players
        """
        obs = np.zeros((self.num_players, 6), dtype=np.int32)
        none_t = self.num_players
        for i, p in enumerate(self.players):
            obs[i, 0] = 1 if p.alive else 0
            obs[i, 1] = int(p.hp)
            obs[i, 2] = int(p.bullets)
            obs[i, 3] = int(p.dodges)
            obs[i, 4] = int(self.last_actions[i].type) if i < len(self.last_actions) else int(ActionType.NOOP)
            tgt = self.last_actions[i].target if i < len(self.last_actions) else none_t
            obs[i, 5] = int(tgt)
        return obs
