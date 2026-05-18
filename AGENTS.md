# DouZero 项目文档 — AGENTS.md

**DouZero: 基于自博弈深度强化学习的四人斗地主 AI（双副牌）**
原始论文: ICML 2021 · [arxiv.org/abs/2106.06135](https://arxiv.org/abs/2106.06135)

---

## 1. 项目全景

DouZero 用深度强化学习（Deep Monte Carlo，DMC）训练四人斗地主 AI（双副牌，108张）。

```
DouZero/
├── train.py                    # 训练入口
├── evaluate.py                 # 评估入口（4 个位置参数）
├── generate_eval_data.py       # 生成固定评估数据集（双副牌发牌）
├── requirements.txt
├── baselines/
└── douzero/
    ├── env/                    # 游戏引擎
    │   ├── game.py             # GameEnv、InfoSet（4人）
    │   ├── env.py              # Env（gym 风格）、观测编码（108维）
    │   ├── move_detector.py    # 牌型识别（四人规则）
    │   ├── move_generator.py   # 合法动作生成（四人规则）
    │   ├── move_selector.py    # 筛选能压过对手的动作
    │   └── utils.py            # 常量（含新炸弹类型 TYPE_16-19）
    ├── dmc/                    # 训练系统
    │   ├── dmc.py              # Actor-Learner 主训练循环（4位置）
    │   ├── models.py           # 神经网络（Landlord 861维 / Farmer 977维）
    │   ├── utils.py            # Actor 进程、Buffer（4位置）
    │   ├── env_utils.py        # 训练用环境包装
    │   ├── arguments.py        # 超参数 CLI
    │   └── file_writer.py      # 日志 & 检查点
    └── evaluation/
        ├── deep_agent.py
        ├── random_agent.py
        ├── rlcard_agent.py
        └── simulation.py       # 4人多进程评估
```

---

## 2. 四人斗地主规则（双副牌）

### 2.1 发牌

- 使用两副牌，共 **108** 张
- 每位玩家初始发 **25** 张；地主获得 8 张底牌，共 **33** 张
- 三位农民各持 **25** 张

### 2.2 出牌顺序

```
地主 → 地主下家 → 地主对面 → 地主上家 → 地主 → …（循环）
```

### 2.3 炸弹层级（由弱到强）

| 类型 | 规则 | 代码 ID |
|------|------|---------|
| 枪 | 4张相同 | TYPE_4_BOMB |
| 炮 | 5张相同 | TYPE_16_PENTA_BOMB |
| 火箭/六喜 | 6张相同 | TYPE_17_HEXA_BOMB |
| 导弹/七巧 | 7张相同 | TYPE_18_HEPTA_BOMB |
| 天炸 | 8张相同 | TYPE_19_OCTA_BOMB |
| 天尊 | 4张王 [20,20,30,30] | TYPE_5_KING_BOMB |

除比自己大的炸弹外，炸弹可打任意牌型。

### 2.4 合法牌型

| 类型 | 说明 | 代码 ID |
|------|------|---------|
| PASS | 过牌 | TYPE_0_PASS |
| 单牌 | 单张 | TYPE_1_SINGLE |
| 对牌 | 两张相同 | TYPE_2_PAIR |
| 三张 | 三张相同 | TYPE_3_TRIPLE |
| 三带一对 | 3张 + 1对（**只能带对子**，不能带单张）| TYPE_7_3_2 |
| 双顺 | 3对或以上连续对牌，最高到 AA，**不能跨越 A-2**（2不在顺序中）| TYPE_9_SERIAL_PAIR |
| 三顺/飞机 | 2组或以上连续三张 | TYPE_10_SERIAL_TRIPLE |
| 飞机带翅膀 | 三顺 + **同等数量的连续对子**（翅膀也必须是连顺）| TYPE_12_SERIAL_3_2 |

**注意（与三人版的关键区别）：**
- **无顺子（单联）**：单牌序列（3,4,5,6,7…）在四人规则中**不是**合法牌型
- **三带只能带对**：三带一单（TYPE_6_3_1）无效
- **飞机只能带连对**：飞机带单张（TYPE_11_SERIAL_3_1）无效，且对子翅膀必须也连续
- **无四带**：四带两单/两对（TYPE_13/14）无效，4张相同直接为枪（炸弹）
- **2不入序**：2（牌值17）不能出现在双顺、三顺、飞机任何顺序组合中；A(14) 和 2(17) 之间差值为 3，天然不连续

### 2.5 胜负与奖励

| 结果 | 地主 | 农民（各） |
|------|------|---------|
| 地主先出完 | +3 × 2^bomb_num | -1 × 2^bomb_num |
| 农民先出完 | -3 × 2^bomb_num | +1 × 2^bomb_num |

---

## 3. 牌面编码（双副牌，108维）

| 结构 | 维度 |
|------|------|
| 8×13 矩阵（3-A 各 8 张）列主序展开 | 104 |
| 小王计数位（0-2张） | 2 |
| 大王计数位（0-2张） | 2 |
| **合计** | **108** |

`Card2Column`: 3→0, 4→1, …, A→11, 2→12  
`NumOnes2Array`: 支持 0-8 张（2 副牌最多 8 张同点数）

---

## 4. 观测编码（env/env.py）

### 历史序列 z

20 步（5 轮 × 4人）× 108维 → reshape (5, 432)，送入 LSTM

### 地主观测 x_no_action = 753

| 特征 | 维度 |
|------|------|
| 我的手牌 | 108 |
| 其他三人手牌（联合） | 108 |
| 上一次出牌 | 108 |
| 地主下家已出牌 | 108 |
| 地主对面已出牌 | 108 |
| 地主上家已出牌 | 108 |
| 下家剩余张数（one-hot, max 25）| 25 |
| 对面剩余张数（one-hot, max 25）| 25 |
| 上家剩余张数（one-hot, max 25）| 25 |
| 炸弹数（one-hot, max 30）| 30 |

x_batch = 753 + 108（动作）= **861**

### 农民观测 x_no_action = 869（三个农民共用同一架构）

| 特征 | 维度 |
|------|------|
| 我的手牌 | 108 |
| 其他三人手牌（联合） | 108 |
| 地主已出牌 | 108 |
| 队友1 已出牌 | 108 |
| 队友2 已出牌 | 108 |
| 上一次出牌 | 108 |
| 地主上一次出牌 | 108 |
| 地主剩余张数（one-hot, max 33）| 33 |
| 队友1 剩余张数（one-hot, max 25）| 25 |
| 队友2 剩余张数（one-hot, max 25）| 25 |
| 炸弹数（one-hot, max 30）| 30 |

x_batch = 869 + 108（动作）= **977**

---

## 5. 神经网络（dmc/models.py）

### LandlordLstmModel（地主）

```
z: (batch, 5, 432) → LSTM(432, 128) → 取最后隐状态 (batch, 128)
拼接 x (861 维) → (batch, 989)
→ Linear(989, 512) → ReLU × 5 → Linear(512, 1)
```

### FarmerLstmModel（农民，三个位置共享同一架构）

```
z: (batch, 5, 432) → LSTM(432, 128) → (batch, 128)
拼接 x (977 维) → (batch, 1105)
→ Linear(1105, 512) → ReLU × 5 → Linear(512, 1)
```

### Model 包装类

4 个模型：`landlord`、`landlord_down`、`landlord_across`、`landlord_up`

---

## 6. 训练系统（dmc/）

### Buffer 规格

| 字段 | 形状 |
|------|------|
| obs_x_no_action | (T, 753) 地主 / (T, 869) 农民 |
| obs_action | (T, 108) |
| obs_z | (T, 5, 432) |
| done / episode_return / target | (T,) |

### 位置列表

```python
_POSITIONS = ['landlord', 'landlord_down', 'landlord_across', 'landlord_up']
```

---

## 7. 评估系统

```bash
# 生成固定评估数据（双副牌，4人）
python generate_eval_data.py --output eval_data.pkl --num_games 10000

# 评估
python evaluate.py \
  --landlord baselines/landlord.ckpt \
  --landlord_down baselines/landlord_down.ckpt \
  --landlord_across baselines/landlord_across.ckpt \
  --landlord_up baselines/landlord_up.ckpt
```

---

## 8. 快速上手

```bash
# GPU 训练
python train.py --gpu_devices 0 --num_actors 5 --training_device 0

# CPU 训练
python train.py --actor_device_cpu --training_device cpu

# 从检查点继续
python train.py --load_model --xpid my_experiment
```
