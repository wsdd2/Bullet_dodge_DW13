# Bullet-Dodge (PettingZoo + PPO)

## 安装
```bash
pip install pettingzoo gymnasium numpy torch
```

## 文件
- `game.py`：核心规则
- `bullet_dodge_pz.py`：PettingZoo ParallelEnv（同步同时行动 + action_mask + 奖励塑形）
- `train_ppo_pz.py`：PPO-Clip 训练脚本（支持固定对手，仅训练 player_0）
- `play_pz.py`：试玩脚本（支持人类控制 / 自动对战 / bot 策略选择）

## 训练

### 全员同时学习（默认）
```bash
python train_ppo_pz.py --num_players 6 --steps 800000 --device cuda
```

### 固定对手，仅训练 player_0
对手策略从外部 run 目录加载（final 或 checkpoints）。
```bash
python train_ppo_pz.py --num_players 6 --steps 800000 --device cuda \
  --opponent_run_dir runs_pz/20260105_235959 --opponent_source final
```

训练输出：
- `p0_winrate@100`：最近 100 局 player_0 胜率
- `p0_avgR@100`：最近 100 局 player_0 平均回报
- `avg_len@100`：最近 100 局平均回合数
- `suddenDeathRate`：最近 100 局进入突然死亡比例
- `loss`：PPO 指标（pi/v/ent/clip）

训练结束自动生成：
- `best.pt`：按指标保存的 player_0 最佳策略
- `checkpoints/`：策略快照
- `eval_logs/`：best vs old 的逐回合日志（可通过 --eval_episodes 控制）

## 试玩 / 自动对战

### 人类控制某个玩家，其余为 bot
bot 使用 final（新策略）：
```bash
python play_pz.py --run_dir runs_pz/20260105_235959 --human_player 0 --bot-new-policy --render
```

bot 使用 checkpoints（旧策略）：
```bash
python play_pz.py --run_dir runs_pz/20260105_235959 --human_player 2 --bot-old-policy --old_pick random --render
```

### 自动对战（player_0 使用 best，其余 bot 使用旧策略）
```bash
python play_pz.py --run_dir runs_pz/20260105_235959 --auto-play --episodes 5 --render
```
