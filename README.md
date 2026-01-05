# Bullet-Dodge (PettingZoo + PPO)
<<<<<<< HEAD
基于Gymnasium + PettingZoo构建的基于PPO-Clip的强化学习小游戏代码，小游戏原型是DollWeekend 13上的“天选之娃”游戏，当时就觉得好像是个强化学习任务→_→
=======

>>>>>>> 49efced (merge remote and local)
## 安装
```bash
pip install pettingzoo gymnasium numpy torch
```

## 文件
- `game.py`：核心规则
- `bullet_dodge_pz.py`：PettingZoo ParallelEnv（同步同时行动 + action_mask + 奖励塑形）
<<<<<<< HEAD
- `train_ppo_pz.py`：独立 PPO-Clip（每个 player 一个网络）
- `play_pz.py`：加载 checkpoint 对战/试玩

## 训练
```bash
python train_ppo_pz.py --num_players 6 --steps 500000 --device cpu
# 或 cuda
python train_ppo_pz.py --num_players 6 --steps 500000 --device cuda
```

## 试玩
```bash
python play_pz.py --num_players 6 --ckpt_dir runs_pz/20260105_123456 --episodes 3 --render
```

## 备注：为什么“每个玩家一个网络”
对抗博弈里如果所有玩家共享同一个网络并且共同优化，很容易变成“合作最大化总回报”，不符合期望中的各自为战的策略。
独立 PPO 是一个可跑基线；下一步考虑做 league/policy-pool（对手混合历史快照）。
=======
- `train_ppo_pz.py`：独立 PPO-Clip（每个 player 一个网络；带训练指标/ckpt/best/eval 日志）
- `play_pz.py`：加载 checkpoint 对战/试玩（简单版）

## 训练（更丰富日志 + best.pt + checkpoints + eval logs）
```bash
python train_ppo_pz.py --num_players 6 --steps 600000 --device cpu
# CUDA:
python train_ppo_pz.py --num_players 6 --steps 600000 --device cuda
```

输出目录（示例）：
```
runs_pz/20260105_235959/
  config.json
  best.pt
  final_player_0.pt ...
  checkpoints/
    upd_00050_player_3.pt ...
  eval_logs/
    eval_0.txt ...
```

训练打印字段说明：
- `p0_winrate@100`：player_0 最近 100 局胜率
- `p0_avgR@100`：player_0 最近 100 局平均总奖励
- `avg_len@100`：最近 100 局平均回合数
- `suddenDeathRate`：最近 100 局进入突然死亡比例
- `loss(mean)`：所有玩家 PPO 的平均 pi_loss / v_loss / entropy / clip_frac

训练结束会自动生成 `eval_logs/eval_*.txt`，逐回合记录所有玩家的行动与资源变化，并高亮 `player_0`。
>>>>>>> 49efced (merge remote and local)
