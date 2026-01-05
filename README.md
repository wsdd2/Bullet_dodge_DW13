# Bullet_dodge
使用PettingZoo + Gymnasium构建的PPO-Clip算法的强化学习小游戏，小游戏原型是Doll Weekend 13上的“天选之娃”游戏
# Bullet-Dodge (PettingZoo + PPO)

## 安装
```bash
pip install pettingzoo gymnasium numpy torch
```

## 文件
- `game.py`：核心规则（与你前面那份一致）
- `bullet_dodge_pz.py`：PettingZoo ParallelEnv（同步同时行动 + action_mask + 奖励塑形）
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
对抗博弈里如果所有玩家共享同一个网络并且共同优化，很容易变成“合作最大化总回报”，不符合期望中各自为战的策略。
独立 PPO 是一个可跑基线；下一步考虑做 league/policy-pool（对手混合历史快照）。
