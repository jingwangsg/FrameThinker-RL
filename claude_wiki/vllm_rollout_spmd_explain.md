# vLLM Rollout SPMD 多轮迭代机制完整解析

## 目录

1. [概述](#1-概述)
2. [主入口：generate_sequences() 方法](#2-主入口generate_sequences-方法)
3. [批次扩展机制](#3-批次扩展机制)
4. [多轮迭代控制](#4-多轮迭代控制)
5. [状态管理](#5-状态管理)
6. [Agent 模式 vs 标准模式](#6-agent-模式-vs-标准模式)
7. [并行执行机制](#7-并行执行机制)
8. [完整数据流](#8-完整数据流)
9. [关键设计洞察](#9-关键设计洞察)
10. [实际执行示例](#10-实际执行示例)
11. [代码走查](#11-代码走查)

---

## 1. 概述

### 1.1 文件作用

**文件**: `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`

**核心功能**:
- 使用 vLLM 引擎进行高效的序列生成（rollout）
- 支持标准单次生成和 Agent 多轮交互模式
- 实现 SPMD (Single Program, Multiple Data) 并行模式
- 管理多模态输入（图像、视频）

### 1.2 关键类

```python
class vLLMRollout:
    """vLLM-based rollout worker for RL training"""

    def __init__(self, config):
        # 初始化 vLLM 引擎和采样参数
        self.inference_engine = vllm.LLM(...)
        self.sampling_params = SamplingParams(...)

    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # 主要生成方法
        pass
```

### 1.3 两种工作模式

| 模式 | 触发条件 | 特点 |
|------|---------|------|
| **Agent 模式** | `config.agent.activate_agent=True` | 多轮交互、工具调用、动态观察 |
| **标准模式** | `config.agent.activate_agent=False` | 单次生成、无工具调用 |

---

## 2. 主入口：generate_sequences() 方法

### 2.1 方法签名

**位置**: `vllm_rollout_spmd.py:224-421`

```python
@torch.no_grad()
def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
    """
    Generate sequences using vLLM engine.

    Args:
        prompts: DataProto containing prompts and multi-modal inputs
        **kwargs: Additional arguments (e.g., temperature override)

    Returns:
        DataProto with generated sequences and metadata
    """
```

### 2.2 完整执行流程

```
┌─────────────────────────────────────────────────────────────────────┐
│ INPUT: DataProto(prompts)                                            │
│   ├── batch["input_ids"]: (batch_size, prompt_len)                  │
│   ├── batch["attention_mask"]: (batch_size, prompt_len)             │
│   ├── batch["position_ids"]: (batch_size, [4,] prompt_len)          │
│   └── non_tensor_batch: Dict with multi-modal data                  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: 预处理 (Lines 238-287)                                      │
│   ├── 提取 tensor 和 non-tensor 数据                                 │
│   ├── 准备 vLLM 输入格式                                             │
│   └── 处理多模态数据（图像、视频）                                   │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 2: 模式分支 (Line 311)                                         │
│                                                                      │
│   if config.agent.activate_agent:                                   │
│       ├─→ Agent 模式 (Lines 312-319)                                │
│       │   └── agent_rollout_loop()                                  │
│       │                                                              │
│   else:                                                              │
│       └─→ 标准模式 (Lines 321-338)                                  │
│           └── inference_engine.generate()                           │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3: 批次扩展 (Lines 340-355) [仅标准模式]                       │
│   └── 如果 sampling_params.n > 1，扩展响应                          │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 4: 拼接序列 (Line 357)                                         │
│   └── sequences = cat([prompts, responses], dim=-1)                 │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 5: 计算掩码和位置编码 (Lines 359-378)                          │
│   ├── attention_mask: 拼接 prompt 和 response 掩码                  │
│   ├── position_ids: 处理视觉模型的特殊位置编码                       │
│   └── action_mask: [Agent 模式] 从 agent_proto 获取                 │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 6: 构建返回 DataProto (Lines 380-421)                          │
│   ├── 标准字段: input_ids, attention_mask, position_ids             │
│   └── Agent 字段: action_mask, env_reward, tool_cnt                 │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ OUTPUT: DataProto(complete sequences)                                │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 关键代码片段

```python
# Agent 模式入口
if self.config.agent.activate_agent:
    agent_proto = agent_rollout_loop(
        config=self.config,
        vllm_engine=self.inference_engine,
        vllm_inputs=vllm_inputs,
        prompts=prompts,
        multi_modal_inputs=non_tensor_batch.get("multi_modal_inputs", None),
        sampling_params=self.sampling_params,
    )
    response = agent_proto.batch.pop("response")
    # 获取 agent 特定字段
    action_mask = agent_proto.batch.pop("action_mask")
    env_reward = agent_proto.batch.pop("env_reward")
    tool_cnt = agent_proto.batch.pop("tool_cnt")

# 标准模式入口
else:
    outputs = self.inference_engine.generate(
        prompts=vllm_inputs,
        sampling_params=self.sampling_params,
        use_tqdm=False,
    )
    # 提取生成的 token IDs
    response = [output.outputs[sample_id].token_ids
                for output in outputs
                for sample_id in range(len(output.outputs))]
```

---

## 3. 批次扩展机制

### 3.1 扩展时机对比

| 模式 | 扩展位置 | 扩展方式 |
|------|---------|---------|
| **Agent 模式** | `agent_rollout_loop` 内部 | 交错扩展（初始化时） |
| **标准模式** | `generate_sequences` 后处理 | 重复扩展（生成后） |

### 3.2 Agent 模式的批次扩展

**位置**: `parallel_env.py:188-200`

**核心逻辑**: 交错复制（Interleaved Duplication）

```python
# 输入: batch_size 个样本，每个生成 n 条轨迹
# 输出: batch_size × n 个样本（交错排列）

batch_size = len(vllm_inputs)
vllm_input_list = []
running_states = []
running_action_masks = []
running_attn_masks = []
reward_tensor_list = []
active_mask = []
mm_input_list = []
tool_call_cnt_list = []

# 交错扩展
for i in range(batch_size):
    for _ in range(sampling_params.n):  # n 条轨迹
        # 复制输入
        vllm_input_list.append(deepcopy(vllm_inputs[i]))

        # 复制初始状态
        prompt_ids = prompts.batch["input_ids"][i, :].clone()
        running_states.append(prompt_ids)

        prompt_mask = prompts.batch["attention_mask"][i, :].clone()
        running_action_masks.append(prompt_mask)  # 初始全 0（prompt 不是 action）
        running_attn_masks.append(prompt_mask)    # 初始为 prompt mask

        # 初始化奖励
        reward_tensor = torch.zeros_like(prompt_ids, dtype=torch.float)
        reward_tensor_list.append(reward_tensor)

        # 初始状态：活跃
        active_mask.append(True)

        # 复制多模态输入
        mm_input_list.append(deepcopy(multi_modal_inputs[i]))

        # 工具调用计数
        tool_call_cnt_list.append(0)
```

### 3.3 交错扩展示例

```
输入: batch_size=3, sampling_params.n=2

原始批次:
┌──────────┐
│ Sample 0 │
├──────────┤
│ Sample 1 │
├──────────┤
│ Sample 2 │
└──────────┘

交错扩展后: (总大小 = 3×2 = 6)
┌────────────────┐
│ Sample 0, R1   │ ← 索引 0
├────────────────┤
│ Sample 0, R2   │ ← 索引 1
├────────────────┤
│ Sample 1, R1   │ ← 索引 2
├────────────────┤
│ Sample 1, R2   │ ← 索引 3
├────────────────┤
│ Sample 2, R1   │ ← 索引 4
├────────────────┤
│ Sample 2, R2   │ ← 索引 5
└────────────────┘

active_mask 初始: [True, True, True, True, True, True]
```

### 3.4 为什么使用交错扩展？

**优势**:
1. **索引管理简单**: 同一样本的不同 rollout 相邻，便于追踪
2. **内存局部性好**: 相关数据在内存中相邻
3. **调试友好**: 便于观察同一样本的不同轨迹

**对比分组扩展**:
```python
# 方案 A: 交错扩展（实际使用）
[S0-R1, S0-R2, S1-R1, S1-R2, S2-R1, S2-R2]

# 方案 B: 分组扩展
[S0-R1, S0-R2], [S1-R1, S1-R2], [S2-R1, S2-R2]
```

### 3.5 标准模式的批次扩展

**位置**: `vllm_rollout_spmd.py:340-355`

**核心逻辑**: 后处理重复

```python
if self.config.rollout.n > 1:
    # 标准模式在生成后扩展
    # response: (batch_size, response_len)
    # 扩展为: (batch_size * n, response_len)

    expanded_response = []
    for i in range(len(response)):
        for _ in range(self.config.rollout.n):
            expanded_response.append(response[i])
    response = torch.stack(expanded_response)
```

---

## 4. 多轮迭代控制

### 4.1 主循环结构

**位置**: `parallel_env.py:204-358`

```python
# 最大轮数
max_turns = config.agent.max_turns  # 例如 5

for step in range(max_turns):
    print(f" [DEBUG 000] {step=}, total={batch_size}, n={sampling_params.n}, "
          f"num_active={sum(active_mask)}")

    # 1. 检查是否还有活跃样本
    if sum(active_mask) == 0:
        break

    # 2. 过滤活跃样本
    active_indices = [idx for idx, is_active in enumerate(active_mask) if is_active]
    active_vllm_inputs = [vinput for vinput, is_active
                          in zip(vllm_input_list, active_mask) if is_active]

    # 3. 生成 actions (只对活跃样本)
    actions = vllm_engine.generate(
        prompts=active_vllm_inputs,
        sampling_params=agent_sampling_params,  # n=1
        use_tqdm=False,
    )

    # 4. 执行工具调用（只在 rank 0）
    if pg.is_first_rank:
        obs_results = env.step(active_indices, actions)
    else:
        obs_results = None

    # 5. 广播结果到所有 rank
    obs_results = pg.broadcast_object(obs_results)
    observations, rewards, dones, info = obs_results

    # 6. 更新每个样本的状态
    for idx, obs, act, rew, done in zip(active_indices, observations,
                                         actions, rewards, dones):
        # 6.1 追加 action tokens
        response_token_ids = torch.tensor(act.outputs[0].token_ids)
        running_states[idx] = torch.cat([running_states[idx], response_token_ids])

        # 6.2 更新 vllm 输入
        vllm_input_list[idx]["prompt_token_ids"] = _concat_vllm_input(
            vllm_input_list[idx]["prompt_token_ids"],
            response_token_ids,
            tokenizer=tokenizer,
        )

        # 6.3 分配奖励到最后一个 action token
        action_reward = torch.zeros_like(response_token_ids, dtype=torch.float)
        reward_tensor_list[idx] = torch.cat([reward_tensor_list[idx], action_reward])
        reward_tensor_list[idx][-1] += rew  # 奖励分配到最后一个 token

        # 6.4 更新 action mask (action tokens = 1)
        action_mask = torch.ones_like(response_token_ids, dtype=torch.int64)
        running_action_masks[idx] = torch.cat([running_action_masks[idx], action_mask])
        running_attn_masks[idx] = torch.cat([running_attn_masks[idx], action_mask])

        # 6.5 检查长度限制
        if running_states[idx].shape[-1] >= max_total_length:
            active_mask[idx] = False
            continue

        # 6.6 检查完成状态
        if done or step == config.agent.max_turns - 1:
            active_mask[idx] = False
            continue

        tool_call_cnt_list[idx] += 1

        # 6.7 追加 observation tokens
        if "prompt_token_ids_vllm" in obs.keys() and "prompt_token_ids_model" in obs.keys():
            obs_token_ids_vllm = obs["prompt_token_ids_vllm"]
            obs_token_ids_model = obs["prompt_token_ids_model"]

            # 检查长度
            if (len(vllm_input_list[idx]["prompt_token_ids"]) + len(obs_token_ids_vllm)
                >= max_total_length):
                active_mask[idx] = False
                continue

            # 更新 vllm 输入
            vllm_input_list[idx]["prompt_token_ids"] = _concat_vllm_input(
                vllm_input_list[idx]["prompt_token_ids"],
                obs_token_ids_vllm,
                tokenizer=tokenizer,
            )

            # 更新模型状态
            running_states[idx] = torch.cat([running_states[idx], obs_token_ids_model])

            # 观察 tokens 的奖励为 0
            obs_reward = torch.zeros(len(obs_token_ids_model), dtype=torch.float)
            reward_tensor_list[idx] = torch.cat([reward_tensor_list[idx], obs_reward])

            # 观察 tokens: action_mask=0, attn_mask=1
            obs_mask = torch.zeros(len(obs_token_ids_model), dtype=torch.int64)
            running_action_masks[idx] = torch.cat([running_action_masks[idx], obs_mask])

            attn_mask = torch.ones(len(obs_token_ids_model), dtype=torch.int64)
            running_attn_masks[idx] = torch.cat([running_attn_masks[idx], attn_mask])

            # 更新多模态数据
            mm_data = obs.get("multi_modal_data", {})
            if "image" in mm_data.keys():
                if "multi_modal_data" not in vllm_input_list[idx].keys():
                    vllm_input_list[idx]["multi_modal_data"] = {"image": []}
                vllm_input_list[idx]["multi_modal_data"]["image"] += mm_data["image"]

            mm_input = obs.get("multi_modal_inputs", {})
            if mm_input:
                mm_input_list[idx] = _merge_multi_modal_inputs(
                    mm_input_list[idx], mm_input
                )

        # 6.8 最终长度检查
        if running_states[idx].shape[-1] >= max_total_length:
            active_mask[idx] = False
```

### 4.2 迭代控制要点

#### ① 动态活跃跟踪

```python
# 每轮只处理活跃样本
active_indices = [idx for idx, is_active in enumerate(active_mask) if is_active]

# 示例：
# Turn 0: active_mask = [T, T, T, T, T, T]  → 6 个活跃
# Turn 1: active_mask = [T, F, T, T, F, T]  → 4 个活跃 (2 个完成)
# Turn 2: active_mask = [T, F, F, T, F, T]  → 3 个活跃 (1 个完成)
# Turn 3: active_mask = [F, F, F, F, F, F]  → 0 个活跃 (全部完成)
```

#### ② 提前终止条件

样本在以下情况下标记为非活跃：

```python
# 条件 1: 达到最大长度
if running_states[idx].shape[-1] >= max_total_length:
    active_mask[idx] = False

# 条件 2: 工具返回 done=True
if done:
    active_mask[idx] = False

# 条件 3: 达到最大轮数
if step == config.agent.max_turns - 1:
    active_mask[idx] = False
```

#### ③ 每轮生成参数

```python
# agent_sampling_params 的关键设置
agent_sampling_params.n = 1                    # 每轮只生成一次
agent_sampling_params.detokenize = True         # 返回文本
agent_sampling_params.skip_special_tokens = False  # 保留特殊 token
agent_sampling_params.include_stop_str_in_output = True  # 包含停止符
agent_sampling_params.max_tokens = min(
    config.agent.single_response_max_tokens,
    config.response_length
)
```

### 4.3 循环流程可视化

```
┌──────────────────────────────────────────────────────────────────┐
│                         TURN 0 (初始)                             │
│                                                                   │
│  6 个样本: [S0-R1, S0-R2, S1-R1, S1-R2, S2-R1, S2-R2]           │
│  active_mask: [T, T, T, T, T, T]                                │
│  状态: [PROMPT] × 6                                              │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                         TURN 1                                    │
│                                                                   │
│  ┌────────────────────┐                                          │
│  │ 1. 生成 actions    │  vllm_engine.generate()                 │
│  │    (6 个样本)      │  agent_sampling_params.n = 1            │
│  └────────────────────┘                                          │
│           ↓                                                       │
│  ┌────────────────────┐                                          │
│  │ 2. 执行工具        │  env.step(active_indices, actions)      │
│  │    (并行执行)      │  ThreadPoolExecutor                      │
│  └────────────────────┘                                          │
│           ↓                                                       │
│  ┌────────────────────┐                                          │
│  │ 3. 更新状态        │  追加 action + observation tokens       │
│  │                    │  更新 masks 和 rewards                   │
│  └────────────────────┘                                          │
│           ↓                                                       │
│  状态: [PROMPT][A1][O1] × 6                                      │
│  完成: S0-R2, S2-R1  (done=True)                                │
│  active_mask: [T, F, T, T, F, T]  → 4 个活跃                    │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                         TURN 2                                    │
│                                                                   │
│  活跃样本: 4 个 [S0-R1, S1-R1, S1-R2, S2-R2]                   │
│                                                                   │
│  ┌────────────────────┐                                          │
│  │ 1. 生成 actions    │  只对 4 个活跃样本生成                  │
│  └────────────────────┘                                          │
│           ↓                                                       │
│  ┌────────────────────┐                                          │
│  │ 2. 执行工具        │                                          │
│  └────────────────────┘                                          │
│           ↓                                                       │
│  ┌────────────────────┐                                          │
│  │ 3. 更新状态        │                                          │
│  └────────────────────┘                                          │
│           ↓                                                       │
│  状态: [PROMPT][A1][O1][A2][O2] × 4                             │
│  完成: S1-R1  (done=True)                                       │
│  active_mask: [T, F, F, T, F, T]  → 3 个活跃                    │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                         TURN 3                                    │
│                                                                   │
│  活跃样本: 3 个 [S0-R1, S1-R2, S2-R2]                          │
│  ... (继续直到全部完成或达到 max_turns)                          │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                         最终化                                    │
│                                                                   │
│  active_mask: [F, F, F, F, F, F]  → 全部完成                    │
│                                                                   │
│  最终状态长度:                                                    │
│    S0-R1: 3 turns → [PROMPT][A1][O1][A2][O2][A3][O3]           │
│    S0-R2: 1 turn  → [PROMPT][A1][O1]                           │
│    S1-R1: 2 turns → [PROMPT][A1][O1][A2][O2]                   │
│    S1-R2: 3 turns → [PROMPT][A1][O1][A2][O2][A3][O3]           │
│    S2-R1: 1 turn  → [PROMPT][A1][O1]                           │
│    S2-R2: 3 turns → [PROMPT][A1][O1][A2][O2][A3][O3]           │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. 状态管理

### 5.1 状态张量列表

每个样本维护以下状态：

```python
# 1. running_states: 完整的 token 序列
running_states: List[Tensor]  # 长度 = batch_size × n
# 每个元素: (seq_len,) tensor，逐轮增长

# 2. running_action_masks: 区分 action 和 observation
running_action_masks: List[Tensor]
# 1 = action token (模型生成，需要梯度)
# 0 = observation token (环境返回，不需要梯度) 或 prompt token

# 3. running_attn_masks: 注意力掩码
running_attn_masks: List[Tensor]
# 1 = 有效 token
# 0 = padding token

# 4. reward_tensor_list: 奖励分配
reward_tensor_list: List[Tensor]
# 每个 token 的奖励值，通常只有 action 的最后一个 token 有非零奖励

# 5. active_mask: 活跃状态
active_mask: List[bool]
# True = 样本仍在迭代
# False = 样本已完成

# 6. mm_input_list: 多模态输入
mm_input_list: List[Dict]
# 包含 image_grid_thw, video_grid_thw 等视觉信息

# 7. tool_call_cnt_list: 工具调用次数
tool_call_cnt_list: List[int]
# 记录每个样本调用工具的次数
```

### 5.2 状态演化示例

以单个样本为例，展示状态如何逐轮增长：

```python
# ===== TURN 0: 初始状态 =====
running_states[i]:
  [101, 102, 103, 104, 105]  # PROMPT (5 tokens)

running_action_masks[i]:
  [0, 0, 0, 0, 0]  # Prompt 不是 action

running_attn_masks[i]:
  [1, 1, 1, 1, 1]  # 全部有效

reward_tensor_list[i]:
  [0.0, 0.0, 0.0, 0.0, 0.0]  # 无奖励

active_mask[i]: True


# ===== TURN 1: 生成 action =====
action_token_ids = [201, 202, 203]  # 3 tokens

running_states[i]:
  [101, 102, 103, 104, 105, 201, 202, 203]  # PROMPT + ACTION

running_action_masks[i]:
  [0, 0, 0, 0, 0, 1, 1, 1]  # Action tokens = 1

running_attn_masks[i]:
  [1, 1, 1, 1, 1, 1, 1, 1]  # 全部有效

reward_tensor_list[i]:
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]  # 奖励在最后一个 action token

active_mask[i]: True


# ===== TURN 1: 追加 observation =====
obs_token_ids = [301, 302, 303, 304]  # 4 tokens

running_states[i]:
  [101, 102, 103, 104, 105, 201, 202, 203, 301, 302, 303, 304]
  # PROMPT + ACTION_1 + OBS_1

running_action_masks[i]:
  [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]  # Obs tokens = 0

running_attn_masks[i]:
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 全部有效

reward_tensor_list[i]:
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]

active_mask[i]: True (如果 done=False)


# ===== TURN 2: 生成 action =====
action_token_ids = [401, 402]  # 2 tokens

running_states[i]:
  [101, 102, 103, 104, 105, 201, 202, 203, 301, 302, 303, 304, 401, 402]
  # PROMPT + ACTION_1 + OBS_1 + ACTION_2

running_action_masks[i]:
  [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]

running_attn_masks[i]:
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

reward_tensor_list[i]:
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
  # 新的奖励在最后一个 token

active_mask[i]: True


# ===== TURN 2: 追加 observation =====
obs_token_ids = [501, 502, 503]  # 3 tokens

running_states[i]:
  [101, 102, 103, 104, 105, 201, 202, 203, 301, 302, 303, 304, 401, 402, 501, 502, 503]
  # PROMPT + ACTION_1 + OBS_1 + ACTION_2 + OBS_2

running_action_masks[i]:
  [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0]

running_attn_masks[i]:
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

reward_tensor_list[i]:
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

active_mask[i]: False (假设 done=True)
```

### 5.3 状态更新关键代码

```python
def update_state_with_action(idx, action_token_ids, reward_value):
    """追加 action tokens"""
    # 1. 更新序列
    running_states[idx] = torch.cat([
        running_states[idx],
        action_token_ids
    ])

    # 2. 更新 vllm 输入（用于下轮生成）
    vllm_input_list[idx]["prompt_token_ids"] = _concat_vllm_input(
        vllm_input_list[idx]["prompt_token_ids"],
        action_token_ids,
        tokenizer=tokenizer,
    )

    # 3. 更新奖励（只在最后一个 token）
    action_rewards = torch.zeros_like(action_token_ids, dtype=torch.float)
    reward_tensor_list[idx] = torch.cat([
        reward_tensor_list[idx],
        action_rewards
    ])
    reward_tensor_list[idx][-1] += reward_value  # 最后一个 token 获得奖励

    # 4. 更新 action mask (action = 1)
    action_mask = torch.ones_like(action_token_ids, dtype=torch.int64)
    running_action_masks[idx] = torch.cat([
        running_action_masks[idx],
        action_mask
    ])

    # 5. 更新 attention mask (all valid = 1)
    attn_mask = torch.ones_like(action_token_ids, dtype=torch.int64)
    running_attn_masks[idx] = torch.cat([
        running_attn_masks[idx],
        attn_mask
    ])


def update_state_with_observation(idx, obs_token_ids, obs_multi_modal_data):
    """追加 observation tokens"""
    # 1. 更新序列
    running_states[idx] = torch.cat([
        running_states[idx],
        obs_token_ids
    ])

    # 2. 更新 vllm 输入
    vllm_input_list[idx]["prompt_token_ids"] = _concat_vllm_input(
        vllm_input_list[idx]["prompt_token_ids"],
        obs_token_ids,
        tokenizer=tokenizer,
    )

    # 3. 更新奖励（observation 奖励为 0）
    obs_rewards = torch.zeros(len(obs_token_ids), dtype=torch.float)
    reward_tensor_list[idx] = torch.cat([
        reward_tensor_list[idx],
        obs_rewards
    ])

    # 4. 更新 action mask (observation = 0)
    obs_mask = torch.zeros(len(obs_token_ids), dtype=torch.int64)
    running_action_masks[idx] = torch.cat([
        running_action_masks[idx],
        obs_mask
    ])

    # 5. 更新 attention mask (all valid = 1)
    attn_mask = torch.ones(len(obs_token_ids), dtype=torch.int64)
    running_attn_masks[idx] = torch.cat([
        running_attn_masks[idx],
        attn_mask
    ])

    # 6. 更新多模态数据
    if "image" in obs_multi_modal_data:
        if "multi_modal_data" not in vllm_input_list[idx]:
            vllm_input_list[idx]["multi_modal_data"] = {"image": []}
        vllm_input_list[idx]["multi_modal_data"]["image"] += obs_multi_modal_data["image"]

    # 7. 更新多模态输入（用于位置编码）
    if "multi_modal_inputs" in obs:
        mm_input_list[idx] = _merge_multi_modal_inputs(
            mm_input_list[idx],
            obs["multi_modal_inputs"]
        )
```

### 5.4 最终化处理

**位置**: `parallel_env.py:360-428`

```python
# 循环结束后，清理和标准化所有状态

# 1. 截断到最大长度
running_states = [state[:max_total_length] for state in running_states]

# 2. Pad 成相同长度
state_tensor = pad_2d_list_to_length(
    running_states,
    tokenizer.pad_token_id,
    max_total_length
).to(target_device)
# 输出: (batch_size×n, max_total_length)

# 3. Pad action masks
running_action_masks = [mask[:max_total_length] for mask in running_action_masks]
action_mask_tensor = pad_2d_list_to_length(
    running_action_masks,
    0,  # padding value
    max_total_length
).to(target_device)

# 4. Pad attention masks
running_attn_masks = [mask[:max_total_length] for mask in running_attn_masks]
attn_mask_tensor = pad_2d_list_to_length(
    running_attn_masks,
    0,
    max_total_length
).to(target_device)

# 5. 计算位置编码 (处理 Qwen2-VL 的特殊情况)
if "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__:
    # Qwen2-VL: 4D position IDs (text + 3D vision)
    position_ids_list = []
    for i in range(batch_size * sampling_params.n):
        vision_position_ids = get_rope_index(
            processor,
            input_ids=state_tensor[i, :],
            image_grid_thw=mm_input_list[i].get("image_grid_thw", None),
            video_grid_thw=mm_input_list[i].get("video_grid_thw", None),
            second_per_grid_ts=mm_input_list[i].get("second_per_grid_ts", None),
            attention_mask=attn_mask_tensor[i, :],
        )  # (3, seq_length)

        # 计算文本位置编码
        valid_mask = attn_mask_tensor[i, :].bool()
        text_position_ids = torch.ones((1, len(state_tensor[i, :])), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())

        # 拼接: (4, seq_length)
        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
        position_ids_list.append(position_ids)

    position_ids_tensor = torch.stack(position_ids_list, dim=0)
    # 输出: (batch_size×n, 4, max_total_length)
else:
    # 标准语言模型: 1D position IDs
    position_ids_tensor = compute_position_id_with_mask(attn_mask_tensor)
    # 输出: (batch_size×n, max_total_length)

# 6. Pad rewards
reward_tensor_list = [reward[:max_total_length] for reward in reward_tensor_list]
reward_tensor = pad_2d_list_to_length(
    reward_tensor_list,
    0.0,
    max_total_length
).to(target_device)

# 7. 提取响应部分（最后 response_length 个 tokens）
response = state_tensor[:, -config.response_length:]
env_reward = reward_tensor[:, -config.response_length:]

# 8. 转换工具调用计数
tool_call_tensor = torch.tensor(
    tool_call_cnt_list,
    dtype=torch.float32
).to(target_device).unsqueeze(1)
# 输出: (batch_size×n, 1)

# 9. 返回 DataProto
return DataProto.from_dict(
    tensors={
        "response": response,
        "action_mask": action_mask_tensor,
        "attention_mask": attn_mask_tensor,
        "position_ids": position_ids_tensor,
        "env_reward": env_reward,
        "tool_cnt": tool_call_tensor,
    },
    non_tensors={
        "multi_modal_inputs": mm_input_list
    } if processor is not None else None,
)
```

---

## 6. Agent 模式 vs 标准模式

### 6.1 对比表

| 特性 | Agent 模式 | 标准模式 |
|------|-----------|----------|
| **配置开关** | `config.agent.activate_agent=True` | `config.agent.activate_agent=False` |
| **入口函数** | `agent_rollout_loop()` | `inference_engine.generate()` |
| **代码位置** | `parallel_env.py` | `vllm_rollout_spmd.py:321-338` |
| **多轮迭代** | ✅ 是 (最多 `max_turns` 轮) | ❌ 否 (单次生成) |
| **工具调用** | ✅ 是 (ParallelEnv.step()) | ❌ 否 |
| **批次扩展时机** | 在 agent_rollout_loop 内部（初始化时） | 在 generate_sequences 后处理 |
| **批次扩展方式** | 交错复制 (interleaved) | 重复复制 (sequential) |
| **观察处理** | ✅ action + observation 交替 | ❌ 无观察 |
| **动作掩码** | ✅ 区分 action (1) 和 obs (0) | ❌ 全部为 1 (或不生成) |
| **奖励分配** | ✅ 每轮分配 (action 最后一个 token) | ❌ 无奖励 |
| **多模态更新** | ✅ 动态 (每轮可添加新图像) | ❌ 静态 (只有初始图像) |
| **输出字段** | response, action_mask, env_reward, tool_cnt | response only |
| **提前终止** | ✅ 是 (done=True 或达到长度限制) | ❌ 否 (生成到 max_tokens) |
| **并行粒度** | 样本级别 (不同样本可在不同 turn) | 批次级别 (所有样本同时生成) |

### 6.2 Agent 模式代码

**位置**: `vllm_rollout_spmd.py:311-319`

```python
if self.config.agent.activate_agent:
    agent_proto = agent_rollout_loop(
        config=self.config,
        vllm_engine=self.inference_engine,
        vllm_inputs=vllm_inputs,
        prompts=prompts,
        multi_modal_inputs=non_tensor_batch.get("multi_modal_inputs", None),
        sampling_params=self.sampling_params,
    )

    # 提取 response
    response = agent_proto.batch.pop("response")

    # 提取 agent 特定字段
    action_mask = agent_proto.batch.pop("action_mask")
    env_reward = agent_proto.batch.pop("env_reward")
    tool_cnt = agent_proto.batch.pop("tool_cnt")

    # 多模态输入
    mm_inputs_agent = agent_proto.non_tensor_batch.get("multi_modal_inputs", None)
```

### 6.3 标准模式代码

**位置**: `vllm_rollout_spmd.py:321-338`

```python
else:
    # 单次生成
    outputs = self.inference_engine.generate(
        prompts=vllm_inputs,
        sampling_params=self.sampling_params,
        use_tqdm=False,
    )

    # 提取生成的 token IDs
    response = []
    for output in outputs:
        for sample_id in range(len(output.outputs)):
            response.append(output.outputs[sample_id].token_ids)

    # Pad 成相同长度
    response = pad_2d_list_to_length(
        response,
        self.pad_token_id,
        max_length=self.config.response_length
    ).to(idx.device)
    # 输出: (batch_size, response_length) 或 (batch_size*n, response_length)

    # 标准模式没有 action_mask, env_reward, tool_cnt
    action_mask = None
    env_reward = None
    tool_cnt = None
```

### 6.4 输出字段对比

#### Agent 模式输出

```python
DataProto(
    batch={
        "prompts": (bs×n, prompt_len),
        "responses": (bs×n, response_len),
        "input_ids": (bs×n, max_len),
        "attention_mask": (bs×n, max_len),
        "position_ids": (bs×n, [4,] max_len),
        "action_mask": (bs×n, max_len),        # ⭐ Agent 特有
        "env_reward": (bs×n, response_len),    # ⭐ Agent 特有
        "tool_cnt": (bs×n, 1),                 # ⭐ Agent 特有
    },
    non_tensor_batch={
        "multi_modal_inputs": List[Dict],      # 动态更新
    }
)
```

#### 标准模式输出

```python
DataProto(
    batch={
        "prompts": (bs×n, prompt_len),
        "responses": (bs×n, response_len),
        "input_ids": (bs×n, max_len),
        "attention_mask": (bs×n, max_len),
        "position_ids": (bs×n, [4,] max_len),
        # 没有 action_mask, env_reward, tool_cnt
    },
    non_tensor_batch={
        "multi_modal_inputs": List[Dict],      # 静态（初始值）
    }
)
```

### 6.5 使用场景

#### Agent 模式适用于：
- ✅ 多步推理任务 (例如：数学题解答、代码生成)
- ✅ 需要工具调用 (例如：搜索、计算器、代码执行)
- ✅ 动态信息获取 (例如：FrameThinker 的帧选择)
- ✅ 复杂决策过程 (例如：游戏 AI、机器人控制)

#### 标准模式适用于：
- ✅ 单步生成任务 (例如：文本补全、翻译)
- ✅ 不需要外部交互 (例如：对话生成)
- ✅ 固定上下文 (例如：问答、摘要)

---

## 7. 并行执行机制

### 7.1 多层次并行

vLLM Rollout SPMD 实现了三个层次的并行：

```
┌────────────────────────────────────────────────────────────────┐
│ 层次 1: Tensor Parallel (TP)                                   │
│   └── 模型权重分片到多个 GPU                                   │
│       例如：8B 模型在 2 个 GPU 上，每个 GPU 4B 参数             │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ 层次 2: Batch Parallel                                          │
│   └── 多个样本同时生成（batch_size × n）                       │
│       例如：batch_size=4, n=2 → 8 个样本并行                   │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ 层次 3: Tool Execution Parallel                                 │
│   └── 工具调用使用 ThreadPoolExecutor 并行执行                 │
│       例如：concurrent_workers=4 → 4 个工具并行调用             │
└────────────────────────────────────────────────────────────────┘
```

### 7.2 Tensor Parallel 实现

**位置**: `parallel_env.py:202-228`

```python
from vllm.distributed import parallel_state as vllm_ps

# 获取 TP group
pg = vllm_ps.get_tp_group()

for step in range(config.agent.max_turns):
    # 1. 所有 rank 并行生成 (vLLM 内部处理 TP)
    actions = vllm_engine.generate(
        prompts=active_vllm_inputs,
        sampling_params=agent_sampling_params,
        use_tqdm=False,
    )

    # 2. 只在 rank 0 执行工具调用
    if pg.is_first_rank:
        obs_results = env.step(active_indices, actions)
    else:
        obs_results = None

    # 3. 广播结果到所有 rank
    obs_results = pg.broadcast_object(obs_results)
    observations, rewards, dones, info = obs_results

    # 4. 所有 rank 同步更新状态
    for idx, obs, act, rew, done in zip(...):
        # 更新状态逻辑
        pass
```

**为什么这样设计？**

1. **生成阶段**: 所有 TP rank 都参与，因为模型权重被分片
2. **工具调用**: 只在 rank 0 执行，避免重复计算和副作用
3. **状态同步**: 通过 broadcast 确保所有 rank 状态一致

### 7.3 Tool Execution Parallel

**位置**: `parallel_env.py:573-606`

```python
def step(self, active_indices, actions):
    """
    并行执行工具调用
    """
    # 1. 准备工具输入
    agent_inputs = []
    for i, idx, action in zip(real_indices, valid_indices, valid_actions):
        agent_inputs.append(dict(
            idx=i,
            valid_idx=idx,
            action=action,
            tool=self.tools[idx],
        ))

    # 2. 确定并行度
    num_workers = min(self.config.concurrent_workers, len(valid_actions))

    # 3. 创建进度条（可选）
    pbar = tqdm(
        total=len(valid_actions),
        desc=f"Tool calling on {num_workers} workers"
    ) if self.config.show_tqdm else None

    # 4. 并行执行
    if num_workers <= 1:
        # 顺序执行
        for agi in agent_inputs:
            subidx = agi["idx"]
            obs, reward, done, info = execute_tool_call(
                agi, self.tokenizer, self.processor, pbar=pbar
            )
            obs_list[subidx] = obs
            reward_list[subidx] = reward
            done_list[subidx] |= done
    else:
        # 并行执行
        partial_tool_func = partial(
            execute_tool_call,
            tokenizer=self.tokenizer,
            processor=self.processor,
            pbar=pbar,
        )
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            raw_outputs = list(executor.map(partial_tool_func, agent_inputs))

        # 收集结果
        for agi, raw in zip(agent_inputs, raw_outputs):
            obs, reward, done = raw[0], raw[1], raw[2]
            subidx = agi["idx"]
            obs_list[subidx] = obs
            reward_list[subidx] = reward
            done_list[subidx] |= done

    return obs_list, reward_list, done_list, {}
```

**并行效率分析**:

```python
# 假设有 8 个活跃样本，concurrent_workers=4

# 顺序执行 (num_workers=1):
# 总时间 = 8 × tool_exec_time

# 并行执行 (num_workers=4):
# 总时间 = ⌈8/4⌉ × tool_exec_time = 2 × tool_exec_time
# 加速比 = 8/2 = 4x
```

### 7.4 并行执行流程图

```
┌────────────────────────────────────────────────────────────────┐
│                    Tensor Parallel Group                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ TP Rank 0│  │ TP Rank 1│  │ TP Rank 2│  │ TP Rank 3│      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
└────────────────────────────────────────────────────────────────┘
         │              │              │              │
         ↓              ↓              ↓              ↓
┌────────────────────────────────────────────────────────────────┐
│                vLLM Generate (所有 rank 并行)                   │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ 模型权重分片，共同完成推理                              │  │
│  │ 输入: active_vllm_inputs (4 个样本)                    │  │
│  │ 输出: actions (4 个样本)                                │  │
│  └─────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
         │              │              │              │
         ↓              ↓              ↓              ↓
┌────────────────────────────────────────────────────────────────┐
│            Tool Execution (只在 Rank 0)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ TP Rank 0│  │ TP Rank 1│  │ TP Rank 2│  │ TP Rank 3│      │
│  │   执行   │  │   等待   │  │   等待   │  │   等待   │      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
│       │                                                         │
│       ↓                                                         │
│  ┌──────────────────────────────────────┐                     │
│  │  ThreadPoolExecutor (4 workers)      │                     │
│  │  ┌────────┐ ┌────────┐ ┌────────┐  │                     │
│  │  │Worker 1│ │Worker 2│ │Worker 3│  │                     │
│  │  │Tool A  │ │Tool B  │ │Tool C  │  │                     │
│  │  └────────┘ └────────┘ └────────┘  │                     │
│  │  ┌────────┐                         │                     │
│  │  │Worker 4│                         │                     │
│  │  │Tool D  │                         │                     │
│  │  └────────┘                         │                     │
│  └──────────────────────────────────────┘                     │
│       │                                                         │
│       ↓                                                         │
│  obs_results = [obs_A, obs_B, obs_C, obs_D]                   │
└────────────────────────────────────────────────────────────────┘
         │              │              │              │
         ↓              ↓              ↓              ↓
┌────────────────────────────────────────────────────────────────┐
│            Broadcast Results (Rank 0 → All Ranks)               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ TP Rank 0│  │ TP Rank 1│  │ TP Rank 2│  │ TP Rank 3│      │
│  │  发送    │─→│  接收    │  │  接收    │  │  接收    │      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
└────────────────────────────────────────────────────────────────┘
         │              │              │              │
         ↓              ↓              ↓              ↓
┌────────────────────────────────────────────────────────────────┐
│            Update States (所有 rank 同步)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ TP Rank 0│  │ TP Rank 1│  │ TP Rank 2│  │ TP Rank 3│      │
│  │  更新    │  │  更新    │  │  更新    │  │  更新    │      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
└────────────────────────────────────────────────────────────────┘
```

---

## 8. 完整数据流

### 8.1 端到端追踪

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT: DataProto from RL trainer                                 │
│ ──────────────────────────────────────────────────────────────── │
│ batch = {                                                        │
│   "input_ids": (2, 1024),          # 2 个样本，prompt 长度 1024 │
│   "attention_mask": (2, 1024),                                   │
│   "position_ids": (2, 4, 1024),    # Qwen2-VL 4D 位置编码       │
│ }                                                                │
│ non_tensor_batch = {                                             │
│   "env_name": ["think_with_video", "think_with_video"],        │
│   "video_path": ["/path/to/v1.mp4", "/path/to/v2.mp4"],        │
│   "fps": [30.0, 30.0],                                          │
│   "total_frames": [300, 450],                                   │
│   "multi_modal_inputs": [                                       │
│     {"image_grid_thw": ...},  # 样本 0 的初始帧                │
│     {"image_grid_thw": ...},  # 样本 1 的初始帧                │
│   ]                                                              │
│ }                                                                │
│                                                                  │
│ sampling_params.n = 2  # 每个样本生成 2 条轨迹                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PREPROCESSING (vllm_rollout_spmd.py:238-287)                    │
│ ──────────────────────────────────────────────────────────────── │
│ # 提取并准备 vLLM 输入                                          │
│ vllm_inputs = [                                                  │
│   {                                                              │
│     "prompt_token_ids": [101, 102, ..., 1024],                  │
│     "multi_modal_data": {"image": [PIL.Image, ...]},           │
│   },                                                             │
│   {                                                              │
│     "prompt_token_ids": [101, 102, ..., 1024],                  │
│     "multi_modal_data": {"image": [PIL.Image, ...]},           │
│   },                                                             │
│ ]                                                                │
│                                                                  │
│ multi_modal_inputs = [                                           │
│   {"image_grid_thw": torch.tensor([[8, 24, 24]]), ...},        │
│   {"image_grid_thw": torch.tensor([[8, 24, 24]]), ...},        │
│ ]                                                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ AGENT_ROLLOUT_LOOP - INITIALIZATION                             │
│ (parallel_env.py:174-200)                                       │
│ ──────────────────────────────────────────────────────────────── │
│ # 批次扩展: 2 → 2×2 = 4                                         │
│ expanded_batch = [                                               │
│   Sample_0_Rollout_1,  # idx=0                                  │
│   Sample_0_Rollout_2,  # idx=1                                  │
│   Sample_1_Rollout_1,  # idx=2                                  │
│   Sample_1_Rollout_2,  # idx=3                                  │
│ ]                                                                │
│                                                                  │
│ # 初始状态                                                       │
│ running_states = [                                               │
│   [101, 102, ..., 1024],  # idx=0, 长度 1024                   │
│   [101, 102, ..., 1024],  # idx=1, 长度 1024                   │
│   [101, 102, ..., 1024],  # idx=2, 长度 1024                   │
│   [101, 102, ..., 1024],  # idx=3, 长度 1024                   │
│ ]                                                                │
│                                                                  │
│ active_mask = [True, True, True, True]                          │
│ tool_call_cnt_list = [0, 0, 0, 0]                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ TURN 1: Generate Actions                                         │
│ (parallel_env.py:217-221)                                       │
│ ──────────────────────────────────────────────────────────────── │
│ # 所有 4 个样本活跃                                              │
│ active_indices = [0, 1, 2, 3]                                   │
│                                                                  │
│ # 生成 (agent_sampling_params.n=1)                              │
│ actions = vllm_engine.generate(...)                             │
│                                                                  │
│ # 输出示例:                                                      │
│ actions[0].text = "<think>需要查看 100-200 帧</think>          │
│                    <action>choose frames between 100 and 200     │
│                    </action>"                                    │
│ actions[0].token_ids = [201, 202, ..., 220]  # 20 tokens       │
│                                                                  │
│ actions[1].text = "<think>需要查看 50-150 帧</think>           │
│                    <action>choose frames between 50 and 150      │
│                    </action>"                                    │
│ actions[1].token_ids = [201, 202, ..., 218]  # 18 tokens       │
│                                                                  │
│ # ... (actions[2], actions[3] 类似)                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ TURN 1: Execute Tools                                            │
│ (parallel_env.py:223-228, 573-606)                              │
│ ──────────────────────────────────────────────────────────────── │
│ # 解析 actions 并调用工具                                        │
│ # 例如: "choose frames between 100 and 200"                     │
│ #   → extract_frames(video_path, [100, 110, ..., 200])         │
│ #   → return {"multi_modal_data": {"image": [frame1, ...]}}    │
│                                                                  │
│ # 并行执行 (ThreadPoolExecutor, 4 workers)                      │
│ observations = [                                                 │
│   {                                                              │
│     "prompt_token_ids_vllm": [301, 302, ..., 320],             │
│     "prompt_token_ids_model": tensor([301, 302, ..., 320]),    │
│     "multi_modal_data": {"image": [frame1, frame2, ...]},      │
│     "multi_modal_inputs": {"image_grid_thw": ...},             │
│   },                                                             │
│   # ... (3 more observations)                                   │
│ ]                                                                │
│                                                                  │
│ rewards = [0.5, 0.3, 0.4, 0.6]  # 每个 action 的奖励            │
│ dones = [False, True, False, False]  # idx=1 完成               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ TURN 1: Update States                                            │
│ (parallel_env.py:231-352)                                       │
│ ──────────────────────────────────────────────────────────────── │
│ # 更新每个样本                                                   │
│ for idx in [0, 1, 2, 3]:                                        │
│   # 追加 action tokens                                          │
│   running_states[idx] += actions[idx].token_ids                 │
│   running_action_masks[idx] += [1] * len(actions[idx].token_ids) │
│   reward_tensor_list[idx][-1] += rewards[idx]                   │
│                                                                  │
│   # 检查完成                                                     │
│   if dones[idx]:                                                 │
│     active_mask[idx] = False  # idx=1 设为 False                │
│     continue                                                     │
│                                                                  │
│   # 追加 observation tokens                                     │
│   running_states[idx] += observations[idx]["prompt_token_ids_model"] │
│   running_action_masks[idx] += [0] * len(obs_tokens)           │
│                                                                  │
│   # 更新多模态数据                                               │
│   vllm_input_list[idx]["multi_modal_data"]["image"] += new_frames │
│   mm_input_list[idx] = merge_mm_inputs(...)                     │
│                                                                  │
│ # 结果:                                                          │
│ running_states[0]: 长度 1024 + 20 + 30 = 1074                   │
│ running_states[1]: 长度 1024 + 18 (完成，无 obs)                │
│ running_states[2]: 长度 1024 + 19 + 28 = 1071                   │
│ running_states[3]: 长度 1024 + 21 + 32 = 1077                   │
│                                                                  │
│ active_mask = [True, False, True, True]  # 3 个活跃              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ TURN 2: Generate Actions (只对活跃样本)                         │
│ ──────────────────────────────────────────────────────────────── │
│ active_indices = [0, 2, 3]  # 3 个活跃                          │
│                                                                  │
│ actions = vllm_engine.generate([                                 │
│   vllm_input_list[0],  # 包含 TURN 1 的 action + obs            │
│   vllm_input_list[2],                                            │
│   vllm_input_list[3],                                            │
│ ])                                                               │
│                                                                  │
│ # ... (执行工具，更新状态)                                       │
│                                                                  │
│ # 假设 idx=2 也完成了                                            │
│ active_mask = [True, False, False, True]  # 2 个活跃            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ TURN 3, 4, ... (继续直到全部完成)                               │
│ ──────────────────────────────────────────────────────────────── │
│ # 最终 active_mask = [False, False, False, False]              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ FINALIZATION (parallel_env.py:360-428)                          │
│ ──────────────────────────────────────────────────────────────── │
│ # 截断和 Pad                                                     │
│ max_total_length = 2048                                          │
│                                                                  │
│ state_tensor = pad_2d_list_to_length(                           │
│   running_states,                                                │
│   pad_value=tokenizer.pad_token_id,                              │
│   max_length=2048                                                │
│ )                                                                │
│ # 输出: (4, 2048)                                                │
│                                                                  │
│ action_mask_tensor = pad_2d_list_to_length(...)                 │
│ # 输出: (4, 2048)                                                │
│                                                                  │
│ attn_mask_tensor = pad_2d_list_to_length(...)                   │
│ # 输出: (4, 2048)                                                │
│                                                                  │
│ reward_tensor = pad_2d_list_to_length(...)                      │
│ # 输出: (4, 2048)                                                │
│                                                                  │
│ # 计算位置编码 (Qwen2-VL)                                       │
│ position_ids_tensor = compute_position_ids(...)                 │
│ # 输出: (4, 4, 2048)                                             │
│                                                                  │
│ # 提取 response (最后 response_length tokens)                   │
│ response_length = 1024                                           │
│ response = state_tensor[:, -1024:]                               │
│ # 输出: (4, 1024)                                                │
│                                                                  │
│ env_reward = reward_tensor[:, -1024:]                            │
│ # 输出: (4, 1024)                                                │
│                                                                  │
│ tool_cnt = torch.tensor([[3], [1], [2], [3]], dtype=float32)   │
│ # 输出: (4, 1)                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ RETURN agent_proto (DataProto)                                   │
│ ──────────────────────────────────────────────────────────────── │
│ tensors = {                                                      │
│   "response": (4, 1024),                                         │
│   "action_mask": (4, 2048),                                      │
│   "attention_mask": (4, 2048),                                   │
│   "position_ids": (4, 4, 2048),                                  │
│   "env_reward": (4, 1024),                                       │
│   "tool_cnt": (4, 1),                                            │
│ }                                                                │
│ non_tensors = {                                                  │
│   "multi_modal_inputs": [                                        │
│     {"image_grid_thw": ...},  # 样本 0-R1, 包含所有帧          │
│     {"image_grid_thw": ...},  # 样本 0-R2                       │
│     {"image_grid_thw": ...},  # 样本 1-R1                       │
│     {"image_grid_thw": ...},  # 样本 1-R2                       │
│   ]                                                              │
│ }                                                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ BACK TO generate_sequences (vllm_rollout_spmd.py:320-421)       │
│ ──────────────────────────────────────────────────────────────── │
│ # 提取 response                                                  │
│ response = agent_proto.batch.pop("response")  # (4, 1024)       │
│                                                                  │
│ # 提取 agent 特定字段                                            │
│ action_mask = agent_proto.batch.pop("action_mask")  # (4, 2048) │
│ env_reward = agent_proto.batch.pop("env_reward")    # (4, 1024) │
│ tool_cnt = agent_proto.batch.pop("tool_cnt")        # (4, 1)    │
│                                                                  │
│ # 拼接 prompt + response                                         │
│ sequences = cat([                                                │
│   prompts.batch["input_ids"],  # (4, 1024) - 已扩展             │
│   response                       # (4, 1024)                     │
│ ], dim=-1)                                                       │
│ # 输出: (4, 2048)                                                │
│                                                                  │
│ # 拼接 attention_mask                                            │
│ attention_mask = cat([                                           │
│   prompts.batch["attention_mask"],  # (4, 1024)                 │
│   torch.ones(4, 1024)                # Response 部分全 1         │
│ ], dim=-1)                                                       │
│ # 输出: (4, 2048)                                                │
│                                                                  │
│ # 从 agent_proto 获取完整的 position_ids 和 action_mask         │
│ position_ids = agent_proto.batch["position_ids"]  # (4, 4, 2048)│
│ action_mask = action_mask  # (4, 2048)                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT: Final DataProto                                          │
│ ──────────────────────────────────────────────────────────────── │
│ batch = {                                                        │
│   "prompts": (4, 1024),                                          │
│   "responses": (4, 1024),                                        │
│   "input_ids": (4, 2048),                                        │
│   "attention_mask": (4, 2048),                                   │
│   "position_ids": (4, 4, 2048),                                  │
│   "action_mask": (4, 2048),       # Agent 特有                   │
│   "env_reward": (4, 1024),        # Agent 特有                   │
│   "tool_cnt": (4, 1),             # Agent 特有                   │
│ }                                                                │
│ non_tensor_batch = {                                             │
│   "multi_modal_inputs": List[Dict]  # 动态更新的多模态数据      │
│ }                                                                │
│                                                                  │
│ # 返回给 RL trainer 用于策略更新                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 数据形状变化追踪

```python
# 阶段 1: 输入
batch_size = 2
n = 2

input_ids: (2, 1024)
attention_mask: (2, 1024)
position_ids: (2, 4, 1024)

# 阶段 2: 批次扩展 (在 agent_rollout_loop 内部)
# 2 → 2×2 = 4

running_states: List[Tensor] of length 4
  - running_states[0]: (1024,)
  - running_states[1]: (1024,)
  - running_states[2]: (1024,)
  - running_states[3]: (1024,)

# 阶段 3: Turn 1 后
# 追加 action + observation

running_states[0]: (1074,)  # 1024 + 20 (action) + 30 (obs)
running_states[1]: (1042,)  # 1024 + 18 (action), 完成
running_states[2]: (1071,)  # 1024 + 19 (action) + 28 (obs)
running_states[3]: (1077,)  # 1024 + 21 (action) + 32 (obs)

# 阶段 4: Turn 2, 3, ... 后
# 持续增长，不同样本长度不同

running_states[0]: (1234,)  # 可能经过 3 turns
running_states[1]: (1042,)  # 只有 1 turn
running_states[2]: (1189,)  # 可能经过 2 turns
running_states[3]: (1256,)  # 可能经过 3 turns

# 阶段 5: Pad 到统一长度
max_total_length = 2048

state_tensor: (4, 2048)
action_mask_tensor: (4, 2048)
attn_mask_tensor: (4, 2048)
reward_tensor: (4, 2048)
position_ids_tensor: (4, 4, 2048)  # Qwen2-VL

# 阶段 6: 提取 response
response_length = 1024

response: (4, 1024)  # 最后 1024 tokens
env_reward: (4, 1024)  # 对应的奖励

# 阶段 7: 最终输出 (拼接 prompt + response)
sequences: (4, 2048)
```

---

## 9. 关键设计洞察

### 9.1 为什么 agent_sampling_params.n=1？

这是整个设计的核心决策之一。

#### 问题场景

```python
batch_size = 2
sampling_params.n = 2  # 每个样本生成 2 条轨迹
max_turns = 3          # 最多 3 轮
```

#### ❌ 错误方案：内层也使用 n>1

```python
# 如果 agent_sampling_params.n = 2
# 每个 turn 生成 2 个分支

Turn 0: 2 × 2 = 4 个样本
Turn 1: 4 × 2 = 8 个分支
Turn 2: 8 × 2 = 16 个分支
Turn 3: 16 × 2 = 32 个分支

# 指数爆炸！
```

#### ✅ 正确方案：外层扩展 + 内层串行

```python
# 初始化时扩展 (外层)
Turn 0: 2 × 2 = 4 个样本（并行）

# 每轮生成时 n=1 (内层)
Turn 1: 4 个样本，各生成 1 次 = 4 条轨迹
Turn 2: 4 个样本，各生成 1 次 = 4 条轨迹
Turn 3: 4 个样本，各生成 1 次 = 4 条轨迹

# 最终: 4 条完整轨迹（可控）
```

### 9.2 为什么使用交错扩展？

#### 对比两种方案

```python
# 方案 A: 交错扩展（实际使用）
[S0-R1, S0-R2, S1-R1, S1-R2, S2-R1, S2-R2]

# 方案 B: 分组扩展
[S0-R1, S0-R2], [S1-R1, S1-R2], [S2-R1, S2-R2]
```

#### 交错扩展的优势

1. **索引管理简单**
   ```python
   # 给定原始样本索引 i，其 n 个 rollout 的索引为:
   rollout_indices = [i * n + j for j in range(n)]

   # 例如: sample 1 (i=1), n=2
   # rollout_indices = [1*2+0, 1*2+1] = [2, 3]
   ```

2. **内存局部性好**
   - 同一样本的不同 rollout 在内存中相邻
   - CPU 缓存命中率更高

3. **调试友好**
   ```python
   # 打印时可以看到同一样本的不同轨迹
   for i in range(0, len(expanded_batch), n):
       print(f"Sample {i//n}:")
       for j in range(n):
           print(f"  Rollout {j}: {expanded_batch[i+j]}")
   ```

### 9.3 为什么 action_mask 区分 action 和 observation？

#### 策略梯度公式

```python
# 策略梯度: ∇θ J(θ) = E[∑ ∇log π(a|s) * A(s,a)]
# 其中 a 是 action，A 是 advantage

# 只有 action tokens 需要梯度
loss = -log_probs[action_mask == 1] * advantages

# observation tokens 不需要梯度
# loss 不包括 log_probs[action_mask == 0]
```

#### action_mask 的作用

```python
# PROMPT: 不是 action，mask=0
# ACTION: 模型生成，需要梯度，mask=1
# OBS: 环境返回，不需要梯度，mask=0

sequence = [P, P, P, A, A, A, O, O, O, A, A, O, O]
action_mask = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0]
              ↑ Prompt  ↑ Act1  ↑ Obs1  ↑A2 ↑Obs2

# 计算 loss 时只使用 action tokens
valid_log_probs = log_probs[action_mask == 1]  # [A1_1, A1_2, A1_3, A2_1, A2_2]
```

### 9.4 为什么奖励只在最后一个 action token？

#### 延迟奖励 (Delayed Reward)

```python
# Turn 1:
ACTION: "choose frames between 100 and 200"
       ↓
OBS: [frame_100, frame_110, ..., frame_200]
       ↓
REWARD: 0.5 (观察到相关帧)

# 奖励分配:
sequence = [P, P, P, A1, A2, A3, O1, O2, O3]
reward =   [0, 0, 0, 0,  0,  0.5, 0,  0,  0]
                        ↑ 最后一个 action token
```

#### 原因

1. **信用分配 (Credit Assignment)**
   - 整个 action 序列是一个完整的决策
   - 奖励归因于完整决策，而非中间 token

2. **稳定训练**
   - 只有一个位置获得奖励，梯度更新更稳定
   - 避免稀疏奖励在序列中扩散

3. **与 PPO 算法一致**
   - PPO 通常在 trajectory 的特定位置分配奖励
   - 便于计算 advantage 和 value

### 9.5 为什么使用 ThreadPoolExecutor 而非 ProcessPoolExecutor？

#### 对比

| 特性 | ThreadPoolExecutor | ProcessPoolExecutor |
|------|-------------------|---------------------|
| **并行类型** | 多线程 | 多进程 |
| **GIL 影响** | 受限于 GIL（I/O 密集型仍有效） | 不受 GIL 限制 |
| **内存开销** | 低（共享内存） | 高（每个进程独立内存） |
| **通信开销** | 低（共享对象） | 高（序列化/反序列化） |
| **启动时间** | 快 | 慢 |
| **适用场景** | I/O 密集型 | CPU 密集型 |

#### 工具调用特点

```python
# 工具调用通常是 I/O 密集型:
def execute_tool_call(sample):
    # 1. 文件 I/O: 读取视频帧
    frames = extract_frames(video_path, frame_indices)

    # 2. 图像处理: 相对轻量
    scaled_frames = [scale_frame(f) for f in frames]

    # 3. Tokenization: 受 GIL 影响小
    obs_tokens = tokenizer.encode(obs_text)

    return obs_tokens, frames
```

#### 为什么选择 ThreadPoolExecutor？

1. **I/O 密集型**: 工具调用主要是文件 I/O（读取视频帧）
2. **低内存开销**: 避免复制大量图像数据
3. **快速启动**: 线程创建比进程快
4. **共享 tokenizer**: 避免每个进程加载独立的 tokenizer

### 9.6 为什么只在 rank 0 执行工具调用？

#### Tensor Parallel 的特点

```python
# 模型权重分片到多个 GPU
# 例如: 8B 模型在 4 个 GPU 上

GPU 0: 参数 [0:2B]
GPU 1: 参数 [2B:4B]
GPU 2: 参数 [4B:6B]
GPU 3: 参数 [6B:8B]

# 推理时，所有 GPU 都参与计算
# 最终输出在所有 GPU 上相同（通过 all-reduce）
```

#### 工具调用的特点

```python
# 工具调用有副作用:
def execute_tool_call(action):
    # 1. 文件系统操作
    frames = extract_frames(video_path, frame_indices)

    # 2. 网络请求（可能）
    result = requests.get(api_url)

    # 3. 数据库操作（可能）
    db.insert(...)

    return obs
```

#### 为什么只在 rank 0？

1. **避免重复执行**: 工具调用有副作用，多次执行会导致错误结果
2. **节省计算**: 工具调用不需要模型参与，单个 rank 执行即可
3. **同步简单**: 通过 broadcast 将结果分发到所有 rank

```python
# Rank 0 执行工具
if pg.is_first_rank:
    obs_results = env.step(active_indices, actions)
else:
    obs_results = None

# 广播到所有 rank
obs_results = pg.broadcast_object(obs_results)

# 所有 rank 同步更新状态
for idx, obs in zip(active_indices, obs_results):
    update_state(idx, obs)
```

---

## 10. 实际执行示例

### 10.1 完整示例场景

```python
# 配置
batch_size = 2               # 2 个视频问题
sampling_params.n = 2         # 每个问题生成 2 条轨迹
max_turns = 3                 # 最多 3 轮交互
prompt_length = 1024          # Prompt 长度
response_length = 1024        # Response 长度
max_total_length = 2048       # 最大总长度
```

### 10.2 逐轮执行追踪

#### 初始化

```
样本:
  Sample 0: "视频中哪个场景最重要？"
  Sample 1: "视频的主要事件是什么？"

批次扩展: 2 → 2×2 = 4
  [S0-R1, S0-R2, S1-R1, S1-R2]

初始状态:
  running_states[0]: [101, 102, ..., 1124] (1024 tokens) PROMPT
  running_states[1]: [101, 102, ..., 1124] (1024 tokens) PROMPT
  running_states[2]: [105, 106, ..., 1128] (1024 tokens) PROMPT
  running_states[3]: [105, 106, ..., 1128] (1024 tokens) PROMPT

active_mask: [T, T, T, T]
tool_call_cnt: [0, 0, 0, 0]
```

#### Turn 1

```
┌─────────────────────────────────────────────────────────────┐
│ 活跃样本: 4 个 [S0-R1, S0-R2, S1-R1, S1-R2]                │
└─────────────────────────────────────────────────────────────┘

1. 生成 Actions:
   ────────────────────────────────────────────────────────────
   S0-R1: "<think>需要查看中间部分</think>
           <action>choose frames between 100 and 200</action>"
          Token IDs: [2001, ..., 2020] (20 tokens)

   S0-R2: "<think>应该看开头</think>
           <action>choose frames between 0 and 100</action>"
          Token IDs: [2001, ..., 2018] (18 tokens)

   S1-R1: "<think>检查结尾</think>
           <action>choose frames between 200 and 300</action>"
          Token IDs: [2001, ..., 2019] (19 tokens)

   S1-R2: "<think>全面查看</think>
           <action>choose frames between 50 and 250</action>"
          Token IDs: [2001, ..., 2021] (21 tokens)

2. 执行工具 (ThreadPoolExecutor, 4 workers):
   ────────────────────────────────────────────────────────────
   Worker 1: extract_frames(video_0, [100, 110, ..., 200])
            → 8 frames
   Worker 2: extract_frames(video_0, [0, 10, ..., 100])
            → 8 frames
   Worker 3: extract_frames(video_1, [200, 210, ..., 300])
            → 8 frames
   Worker 4: extract_frames(video_1, [50, 75, ..., 250])
            → 8 frames

3. 观察结果:
   ────────────────────────────────────────────────────────────
   S0-R1: obs_text = "Frame 100: ...\nFrame 110: ...\n..."
          obs_tokens = [3001, ..., 3030] (30 tokens)
          reward = 0.5
          done = False

   S0-R2: obs_text = "Frame 0: ...\nFrame 10: ...\n..."
          obs_tokens = [3001, ..., 3025] (25 tokens)
          reward = 0.3
          done = True  ← 完成！

   S1-R1: obs_text = "Frame 200: ...\nFrame 210: ...\n..."
          obs_tokens = [3001, ..., 3028] (28 tokens)
          reward = 0.4
          done = False

   S1-R2: obs_text = "Frame 50: ...\nFrame 75: ...\n..."
          obs_tokens = [3001, ..., 3032] (32 tokens)
          reward = 0.6
          done = False

4. 更新状态:
   ────────────────────────────────────────────────────────────
   S0-R1:
     running_states[0] = [PROMPT][ACTION][OBS]
                       = 1024 + 20 + 30 = 1074 tokens
     action_mask[0] = [0...0][1...1][0...0]
     reward[0] = [0...0][0...0.5][0...0]
     active_mask[0] = True
     tool_call_cnt[0] = 1

   S0-R2:
     running_states[1] = [PROMPT][ACTION]  ← 无 OBS (done=True)
                       = 1024 + 18 = 1042 tokens
     action_mask[1] = [0...0][1...1]
     reward[1] = [0...0][0...0.3]
     active_mask[1] = False  ← 完成
     tool_call_cnt[1] = 1

   S1-R1:
     running_states[2] = [PROMPT][ACTION][OBS]
                       = 1024 + 19 + 28 = 1071 tokens
     action_mask[2] = [0...0][1...1][0...0]
     reward[2] = [0...0][0...0.4][0...0]
     active_mask[2] = True
     tool_call_cnt[2] = 1

   S1-R2:
     running_states[3] = [PROMPT][ACTION][OBS]
                       = 1024 + 21 + 32 = 1077 tokens
     action_mask[3] = [0...0][1...1][0...0]
     reward[3] = [0...0][0...0.6][0...0]
     active_mask[3] = True
     tool_call_cnt[3] = 1

5. 检查活跃:
   ────────────────────────────────────────────────────────────
   active_mask = [T, F, T, T]
   num_active = 3
```

#### Turn 2

```
┌─────────────────────────────────────────────────────────────┐
│ 活跃样本: 3 个 [S0-R1, S1-R1, S1-R2]                       │
│ (S0-R2 已完成，不参与)                                      │
└─────────────────────────────────────────────────────────────┘

1. 生成 Actions:
   ────────────────────────────────────────────────────────────
   S0-R1: "<think>需要更细致</think>
           <action>choose frames between 150 and 180</action>"
          Token IDs: [4001, ..., 4018] (18 tokens)

   S1-R1: "<think>看到关键信息</think>
           <action>output answer: 在 250 帧附近发生了重要事件</action>"
          Token IDs: [4001, ..., 4025] (25 tokens)

   S1-R2: "<think>需要确认</think>
           <action>choose frames between 100 and 150</action>"
          Token IDs: [4001, ..., 4019] (19 tokens)

2. 执行工具:
   ────────────────────────────────────────────────────────────
   S0-R1: extract_frames(...) → obs_tokens (26 tokens)
          reward = 0.7
          done = False

   S1-R1: No tool (output answer)
          reward = 1.0  ← 正确答案
          done = True  ← 完成！

   S1-R2: extract_frames(...) → obs_tokens (29 tokens)
          reward = 0.5
          done = False

3. 更新状态:
   ────────────────────────────────────────────────────────────
   S0-R1:
     running_states[0] = [PROMPT][A1][O1][A2][O2]
                       = 1074 + 18 + 26 = 1118 tokens
     tool_call_cnt[0] = 2
     active_mask[0] = True

   S0-R2: (inactive, 不更新)

   S1-R1:
     running_states[2] = [PROMPT][A1][O1][A2]  ← 无 O2 (done)
                       = 1071 + 25 = 1096 tokens
     tool_call_cnt[2] = 2  ← 虽然无工具调用，但算一次
     active_mask[2] = False  ← 完成

   S1-R2:
     running_states[3] = [PROMPT][A1][O1][A2][O2]
                       = 1077 + 19 + 29 = 1125 tokens
     tool_call_cnt[3] = 2
     active_mask[3] = True

4. 检查活跃:
   ────────────────────────────────────────────────────────────
   active_mask = [T, F, F, T]
   num_active = 2
```

#### Turn 3

```
┌─────────────────────────────────────────────────────────────┐
│ 活跃样本: 2 个 [S0-R1, S1-R2]                              │
└─────────────────────────────────────────────────────────────┘

1. 生成 Actions:
   ────────────────────────────────────────────────────────────
   S0-R1: "<think>已收集足够信息</think>
           <action>output answer: 场景 150-180 最重要</action>"
          Token IDs: [5001, ..., 5022] (22 tokens)

   S1-R2: "<think>找到关键</think>
           <action>output answer: 主要事件在 50-250 帧之间</action>"
          Token IDs: [5001, ..., 5024] (24 tokens)

2. 执行工具:
   ────────────────────────────────────────────────────────────
   S0-R1: No tool (output answer)
          reward = 0.8
          done = True  ← 完成！

   S1-R2: No tool (output answer)
          reward = 0.9
          done = True  ← 完成！

3. 更新状态:
   ────────────────────────────────────────────────────────────
   S0-R1:
     running_states[0] = [PROMPT][A1][O1][A2][O2][A3]
                       = 1118 + 22 = 1140 tokens
     tool_call_cnt[0] = 2  ← 无工具调用，不增加
     active_mask[0] = False  ← 完成

   S1-R2:
     running_states[3] = [PROMPT][A1][O1][A2][O2][A3]
                       = 1125 + 24 = 1149 tokens
     tool_call_cnt[3] = 2  ← 无工具调用，不增加
     active_mask[3] = False  ← 完成

4. 检查活跃:
   ────────────────────────────────────────────────────────────
   active_mask = [F, F, F, F]
   num_active = 0  → 退出循环
```

#### 最终化

```
1. 最终状态长度:
   ────────────────────────────────────────────────────────────
   S0-R1: 1140 tokens (3 turns)
   S0-R2: 1042 tokens (1 turn)
   S1-R1: 1096 tokens (2 turns)
   S1-R2: 1149 tokens (3 turns)

2. Pad 到 max_total_length=2048:
   ────────────────────────────────────────────────────────────
   state_tensor: (4, 2048)
     [0]: [101, ..., 1140, PAD, ..., PAD]
     [1]: [101, ..., 1042, PAD, ..., PAD]
     [2]: [105, ..., 1096, PAD, ..., PAD]
     [3]: [105, ..., 1149, PAD, ..., PAD]

3. 提取 response (最后 1024 tokens):
   ────────────────────────────────────────────────────────────
   response: (4, 1024)
     [0]: 包含 [A2][O2][A3]
     [1]: 包含 [ACTION] 的后半部分
     [2]: 包含 [A2]
     [3]: 包含 [A2][O2][A3]

4. 工具调用统计:
   ────────────────────────────────────────────────────────────
   tool_cnt: [[2], [1], [2], [2]]

5. 返回:
   ────────────────────────────────────────────────────────────
   DataProto(
     batch={
       "response": (4, 1024),
       "action_mask": (4, 2048),
       "attention_mask": (4, 2048),
       "position_ids": (4, 4, 2048),
       "env_reward": (4, 1024),
       "tool_cnt": (4, 1),
     },
     non_tensor_batch={
       "multi_modal_inputs": [...]  # 4 个元素，包含所有帧
     }
   )
```

### 10.3 性能指标

```python
# 总体统计
total_samples = 4
total_turns = 3 (max)
actual_turns = [3, 1, 2, 3]  # 每个样本的实际 turns
avg_turns = 2.25

# 生成统计
total_generations = sum(actual_turns) = 9
avg_tokens_per_action = 19.6
total_action_tokens = 9 × 19.6 ≈ 176

# 工具调用统计
total_tool_calls = sum([2, 1, 2, 2]) = 7
avg_tools_per_sample = 1.75

# 时间统计 (假设)
time_per_generation = 0.5s
time_per_tool_call = 0.3s
time_per_turn = 0.5s (generation) + 0.3s (tool) = 0.8s

# 顺序执行 vs 并行执行
sequential_time = 9 × 0.5 + 7 × 0.3 = 6.6s
parallel_time = 3 × 0.8 = 2.4s  (3 turns, 并行 4 samples)
speedup = 6.6 / 2.4 ≈ 2.75x
```

---

## 11. 代码走查

### 11.1 vLLMRollout.__init__()

**位置**: `vllm_rollout_spmd.py:98-203`

```python
def __init__(self, config: DictConfig):
    """
    初始化 vLLM rollout worker

    核心功能:
    1. 加载 vLLM 引擎
    2. 配置 sampling parameters
    3. 初始化 tokenizer 和 processor
    """
    self.config = config

    # 1. 构建 vLLM 引擎配置
    # ────────────────────────────────────────────────────
    engine_args = EngineArgs(
        model=config.model.path,
        tensor_parallel_size=config.tensor_model_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_num_batched_tokens=config.max_num_batched_tokens,
        # ... 其他配置
    )

    # 2. 创建 vLLM 引擎
    # ────────────────────────────────────────────────────
    self.inference_engine = vllm.LLM(**engine_args.create_engine_config())

    # 3. 配置 sampling parameters
    # ────────────────────────────────────────────────────
    kwargs = dict(
        n=1,                                  # 默认 n=1
        logprobs=0,                           # 不返回 log 概率
        max_tokens=config.response_length,    # 最大生成长度
    )

    # 禁用自动 detokenize (由 agent 控制)
    if vllm_version != "0.3.1":
        kwargs["detokenize"] = False

    # 从配置文件读取并覆盖
    for k in config.keys():
        if hasattr(SamplingParams(), str(k)):
            kwargs[k] = config.get(k)

    print(f"kwargs: {kwargs}")
    self.sampling_params = SamplingParams(**kwargs)

    # 4. 加载 tokenizer 和 processor
    # ────────────────────────────────────────────────────
    self.tokenizer = hf_tokenizer(config.model.path)
    self.processor = hf_processor(config.model.path)
    self.pad_token_id = self.tokenizer.pad_token_id
```

**关键点**:
- `sampling_params.n` 默认为 1，可通过配置覆盖
- `sampling_params.stop` 默认为 `[]`，可通过 `config.stop` 设置
- vLLM 引擎支持 tensor parallelism

### 11.2 vLLMRollout.generate_sequences()

**位置**: `vllm_rollout_spmd.py:224-421`

```python
@torch.no_grad()
def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
    """
    主生成方法

    流程:
    1. 预处理输入
    2. 分支: Agent 模式 vs 标准模式
    3. 后处理: 拼接、计算位置编码
    4. 返回 DataProto
    """

    # ═══════════════════════════════════════════════════════
    # STEP 1: 预处理
    # ═══════════════════════════════════════════════════════

    # 提取 tensor 和 non-tensor 数据
    tensor_batch = prompts.batch
    non_tensor_batch = prompts.non_tensor_batch

    # 准备 vLLM 输入格式
    idx = tensor_batch["input_ids"]
    vllm_inputs = []

    for i in range(len(idx)):
        # 基础输入
        vllm_input = {
            "prompt_token_ids": idx[i, :].tolist(),
        }

        # 多模态数据 (如果有)
        if "multi_modal_data" in non_tensor_batch:
            mm_data = non_tensor_batch["multi_modal_data"][i]
            if mm_data:
                vllm_input["multi_modal_data"] = mm_data

        vllm_inputs.append(vllm_input)

    # ═══════════════════════════════════════════════════════
    # STEP 2: 模式分支
    # ═══════════════════════════════════════════════════════

    if self.config.agent.activate_agent:
        # ───────────────────────────────────────────────────
        # Agent 模式: 多轮交互
        # ───────────────────────────────────────────────────
        agent_proto = agent_rollout_loop(
            config=self.config,
            vllm_engine=self.inference_engine,
            vllm_inputs=vllm_inputs,
            prompts=prompts,
            multi_modal_inputs=non_tensor_batch.get("multi_modal_inputs", None),
            sampling_params=self.sampling_params,
        )

        # 提取结果
        response = agent_proto.batch.pop("response")
        action_mask = agent_proto.batch.pop("action_mask")
        env_reward = agent_proto.batch.pop("env_reward")
        tool_cnt = agent_proto.batch.pop("tool_cnt")
        mm_inputs_agent = agent_proto.non_tensor_batch.get("multi_modal_inputs", None)

    else:
        # ───────────────────────────────────────────────────
        # 标准模式: 单次生成
        # ───────────────────────────────────────────────────
        outputs = self.inference_engine.generate(
            prompts=vllm_inputs,
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )

        # 提取 token IDs
        response = []
        for output in outputs:
            for sample_id in range(len(output.outputs)):
                response.append(output.outputs[sample_id].token_ids)

        # Pad 成相同长度
        response = pad_2d_list_to_length(
            response,
            self.pad_token_id,
            max_length=self.config.response_length
        ).to(idx.device)

        # 标准模式没有这些字段
        action_mask = None
        env_reward = None
        tool_cnt = None
        mm_inputs_agent = None

    # ═══════════════════════════════════════════════════════
    # STEP 3: 批次扩展 (标准模式)
    # ═══════════════════════════════════════════════════════

    if not self.config.agent.activate_agent and self.config.rollout.n > 1:
        # 扩展: (bs, len) → (bs*n, len)
        expanded_response = []
        for i in range(len(response)):
            for _ in range(self.config.rollout.n):
                expanded_response.append(response[i])
        response = torch.stack(expanded_response)

        # 同样扩展 prompts
        # ... (类似逻辑)

    # ═══════════════════════════════════════════════════════
    # STEP 4: 拼接 prompt + response
    # ═══════════════════════════════════════════════════════

    sequences = torch.cat([idx, response], dim=-1)
    # 输出: (batch_size * n, prompt_len + response_len)

    # ═══════════════════════════════════════════════════════
    # STEP 5: 计算 attention_mask
    # ═══════════════════════════════════════════════════════

    prompt_attn_mask = tensor_batch["attention_mask"]
    response_attn_mask = torch.ones_like(response, dtype=torch.int64)
    attention_mask = torch.cat([prompt_attn_mask, response_attn_mask], dim=-1)

    # ═══════════════════════════════════════════════════════
    # STEP 6: 计算 position_ids
    # ═══════════════════════════════════════════════════════

    if self.config.agent.activate_agent:
        # Agent 模式: 从 agent_proto 获取完整的 position_ids
        position_ids = agent_proto.batch["position_ids"]
        # 可能是 (bs*n, 4, max_total_length) 对于 Qwen2-VL
    else:
        # 标准模式: 计算位置编码
        if "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # Qwen2-VL: 4D position IDs
            # ... (复杂的计算逻辑)
            position_ids = compute_qwen2vl_position_ids(...)
        else:
            # 标准 LM: 1D position IDs
            position_ids = compute_position_id_with_mask(attention_mask)

    # ═══════════════════════════════════════════════════════
    # STEP 7: 构建返回 DataProto
    # ═══════════════════════════════════════════════════════

    output_batch = {
        "prompts": idx,
        "responses": response,
        "input_ids": sequences,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    # Agent 特定字段
    if self.config.agent.activate_agent:
        output_batch["action_mask"] = action_mask
        output_batch["env_reward"] = env_reward
        output_batch["tool_cnt"] = tool_cnt

    output_non_tensor_batch = {}
    if mm_inputs_agent is not None:
        output_non_tensor_batch["multi_modal_inputs"] = mm_inputs_agent

    return DataProto(
        batch=output_batch,
        non_tensor_batch=output_non_tensor_batch
    )
```

---

## 总结

### 核心机制

1. **批次扩展**: 交错复制，`batch_size × n` 个样本并行
2. **多轮迭代**: 最多 `max_turns` 轮，动态活跃跟踪
3. **状态管理**: 逐轮增长的 token 序列、masks、rewards
4. **并行执行**: TP + Batch + Tool 三层并行
5. **Agent vs 标准**: 两种模式适应不同任务需求

### 设计亮点

- ✅ **外层并行 + 内层串行**: 避免组合爆炸，保持可控
- ✅ **动态终止**: 提前结束节省计算
- ✅ **交错扩展**: 简化索引管理
- ✅ **action_mask 区分**: 准确的策略梯度
- ✅ **只在 rank 0 执行工具**: 避免重复和副作用
- ✅ **ThreadPoolExecutor**: 适配 I/O 密集型工具调用

### 适用场景

- **Agent 模式**: 多步推理、工具调用、动态信息获取
- **标准模式**: 单步生成、固定上下文

---

**文档版本**: 1.0
**最后更新**: 2025-01-10
**维护者**: FrameThinker Team
