# Bullet Dodge (PettingZoo + PPO-Clip)

一个多人同步行动的“装弹 / 闪避 / 指定攻击”小游戏环境，用于多智能体强化学习实验（PettingZoo ParallelEnv）。

---

## 游戏规则

一局比赛有 `num_players` 名玩家（默认训练 10 人，游玩可变但至少 4 人）。

每名玩家有三个资源：

- **生命值 HP**
- **子弹 bullets**
- **闪避次数 dodges**

### NORMAL 阶段（常规回合）

每回合每名存活玩家选择一个动作（同步执行）：

- **RELOAD**：装填 1 发子弹
- **DODGE**：消耗 1 次闪避，本回合免疫所有伤害
- **ATTACK <target>**：消耗 1 发子弹，指定攻击某位存活玩家

结算规则：

- 同回合可以出现多人同时攻击同一目标
- 若目标本回合选择 **DODGE**，则免疫所有伤害
- 否则，每被命中一次扣 1 点血
- 仅剩 1 名存活玩家时获胜

### 决斗阶段（可选：SUDDEN_DEATH）

若某回合结束后 **所有玩家同时阵亡**：

- `duel_enabled=True`：进入 **SUDDEN_DEATH**
- `duel_enabled=False`：本局 **平局结束**（RESULT: DRAW）

SUDDEN_DEATH 规则：

- 参与者：该回合开始时仍存活的玩家（但该回合全灭）
- 状态重置：`HP=1, bullets=0, dodges=1`
- 每回合动作：
  - **DODGE**：消耗 1 次闪避；本回合不可被指定为 DUEL 目标
  - **DUEL <target>**：与目标比大小（掷骰 1-6，点数相同重掷，点小者扣 1 血）

---

## 动作空间与观测（固定维度）

为了支持“训练用 10 人，但游玩可少于/多于 10 人”的需求，环境使用 **固定上限 `max_players`（默认 32）** 来定义动作空间与观测维度。

动作空间（`Discrete(action_n)`）：

- `0`：RELOAD
- `1`：DODGE
- `2 .. 2+max_players-1`：ATTACK->target
- `2+max_players .. 2+2*max_players-1`：DUEL->target

当 `target >= num_players` 或目标不合法时，对应动作会被 `action_mask` 屏蔽。

观测为 `Box(0..1)` 的扁平向量，包含：

- `max_players` 个玩家的 padded 状态（active/alive/hp/bullets/dodges/last_action）
- 全局信息（mode、round_idx、duel_enabled、num_players）
- self_id（按 `max_players` 归一化）

---

## PPO-Clip 简介（PPO 的“稳定器”）

PPO-Clip 的核心思想是：  
在更新策略时限制“新策略相对于旧策略”的变化幅度，避免策略一次跳太远导致训练不稳定。

- 计算比率 `r = pi_new(a|s) / pi_old(a|s)`
- 目标函数对 `r` 做 `clip(r, 1-eps, 1+eps)`
- 同时训练 value function，并加入 entropy 项维持探索

本项目实现中还使用了：

- `action_mask` 对无效动作做 logits 屏蔽（避免学到“开局打空气”）

---

## 训练（PPO-Clip）

依赖：

```bash
pip install pettingzoo gymnasium numpy torch
```

默认：训练 10 人局，并启用决斗（SUDDEN_DEATH）。

```bash
python train_ppo_pz.py --device cuda
```

调整初始资源（默认：HP=2, bullets=1, dodges=2）：

```bash
python train_ppo_pz.py --device cuda --init_hp 2 --init_bullets 1 --init_dodges 2
```

固定对手，仅训练 `player_0`（对手都加载同一个策略文件并冻结）：

```bash
python train_ppo_pz.py --device cuda --opponent_policy_path runs_pz/xxx/checkpoints/upd_00100_player_0.pt
```

输出目录：`runs_pz/YYYYMMDD_HHMMSS/`

- `checkpoints/`：训练快照（例如 `upd_00100_player_0.pt`）
- `best.pt`：按指标保存的 player_0 最佳策略
- `eval_logs/`：训练结束后自动评测的逐回合日志（高亮 player_0）

---

## 游玩（多真人 + bot 难度）

bot 默认从 `run_dir/checkpoints` 选取策略：

- `--easy`：早期 checkpoint
- `--normal`：中期 checkpoint（默认）
- `--hard`：后期 checkpoint（但不是 best）
- `--mixed`：混合抽取

6 人局，其中 2 个真人：

```bash
python play_pz.py --run_dir runs_pz/xxx --num_players 6 --human-player-num 2 --normal --render
```

12 人局，关闭决斗，1 个真人：

```bash
python play_pz.py --run_dir runs_pz/xxx --num_players 12 --no-duel --human-player-num 1 --mixed --render
```

---

## 文件结构

- `game.py`：纯游戏状态转移（不依赖 RL 框架）
- `bullet_dodge_pz.py`：PettingZoo ParallelEnv 封装 + action_mask
- `train_ppo_pz.py`：PPO-Clip 训练脚本
- `play_pz.py`：试玩脚本（多真人 + bot 难度）
