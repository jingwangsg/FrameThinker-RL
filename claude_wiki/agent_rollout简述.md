# Agent Rollout 机制详解：sampling_params 完整追踪

## 目录
1. [sampling_params 概述](#1-sampling_params-概述)
2. [sampling_params.stop 详解](#2-sampling_paramsstop-详解)
3. [初始化位置](#3-初始化位置)
4. [配置来源](#4-配置来源)
5. [agent_rollout_loop 中的处理](#5-agent_rollout_loop-中的处理)
6. [完整数据流程](#6-完整数据流程)
7. [重要属性对比](#7-重要属性对比)
8. [设计原理解析](#8-设计原理解析)

---

## 1. sampling_params 概述

### 1.1 基本信息

**类型**: `vllm.sampling_params.SamplingParams`

**来源**: vLLM 库的采样参数类

**用途**: 控制 vLLM 推理引擎的文本生成行为，包括采样策略、停止条件、输出格式等

### 1.2 主要属性

```python
class SamplingParams:
    n: int = 1                              # 每个 prompt 生成的序列数量
    max_tokens: int = 16                    # 最大生成 token 数
    temperature: float = 1.0                # 采样温度
    top_k: int = 0                          # Top-k 采样（-1 表示禁用）
    top_p: float = 1.0                      # Top-p (nucleus) 采样
    stop: List[str] = []                    # 停止字符串列表
    stop_token_ids: List[int] = []          # 停止 token ID 列表
    ignore_eos: bool = False                # 是否忽略 EOS token
    detokenize: bool = True                 # 是否返回文本（vs token IDs）
    skip_special_tokens: bool = True        # 输出时是否跳过特殊 token
    spaces_between_special_tokens: bool = True  # 特殊 token 之间是否添加空格
    include_stop_str_in_output: bool = False    # 输出中是否包含停止字符串
    logprobs: Optional[int] = None          # 返回的 log 概率数量
    best_of: Optional[int] = None           # Best-of 采样的序列数量
```

---

## 2. sampling_params.stop 详解

### 2.1 定义

**类型**: `List[str]` (字符串列表)

**默认值**: `[]` (空列表)

**作用**: 当模型生成的文本包含列表中的任意字符串时，立即停止生成

### 2.2 典型使用场景

```python
# 示例 1: 限制代码块生成
sampling_params.stop = ["```"]

# 示例 2: 限制工具调用
sampling_params.stop = ["</search>", "</function>", "</tool>"]

# 示例 3: 限制多轮对话
sampling_params.stop = ["<|im_end|>", "<|endoftext|>"]
```

### 2.3 在 FrameThinker 中的使用

默认情况下，`sampling_params.stop = []`，即不设置额外的停止条件。

但可以通过配置文件的 `actor_rollout_ref.rollout.agent.custom_stop` 来添加自定义停止字符串：

```yaml
actor_rollout_ref:
  rollout:
    agent:
      custom_stop: ["</search>", "```"]  # 自定义停止符
```

---

## 3. 初始化位置

### 3.1 创建位置

**文件**: `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`

**行数**: 187-203

**代码**:
```python
def __init__(self, config):
    # 构建基础参数
    kwargs = dict(
        n=1,                                    # 默认生成 1 个序列
        logprobs=0,                             # 不返回 log 概率
        max_tokens=config.response_length,      # 最大生成长度
    )

    # vLLM 版本 != 0.3.1 时禁用自动 detokenize
    if vllm_version != "0.3.1":
        kwargs["detokenize"] = False

    # 从配置文件中读取并覆盖参数
    for k in config.keys():
        if hasattr(SamplingParams(), str(k)):
            kwargs[k] = config.get(k)

    print(f"kwargs: {kwargs}")

    # 创建 SamplingParams 实例
    self.sampling_params = SamplingParams(**kwargs)
```

### 3.2 默认值来源

1. **硬编码默认值**: `n=1`, `logprobs=0`, `detokenize=False`
2. **配置文件值**: `max_tokens`, `temperature`, `top_k`, `top_p` 等
3. **vLLM 库默认值**: 未被覆盖的其他参数（如 `stop=[]`）

---

## 4. 配置来源

### 4.1 配置文件

**文件**: `verl/trainer/config/ppo_trainer.yaml`

**相关配置**:
```yaml
rollout:
  name: vllm
  temperature: 1.0              # 采样温度
  top_k: -1                     # Top-k 采样（-1 表示禁用）
  top_p: 1.0                    # Top-p 采样
  response_length: ${data.max_response_length}  # 最大响应长度
  n: 1                          # 生成序列数（GRPO 时可能 > 1）

  agent:
    activate_agent: False       # 是否启用 agent 模式
    single_response_max_tokens: 32768  # 单次响应最大 token 数
    max_turns: 50               # 最大交互轮数
    custom_stop: []             # 自定义停止字符串列表 ⭐
```

### 4.2 训练脚本中的覆盖

**文件**: `examples/agent/train_frame_thinker.sh`

**关键参数覆盖**:
```bash
actor_rollout_ref.rollout.n=8                           # 每个样本生成 8 条轨迹
actor_rollout_ref.rollout.agent.activate_agent=True     # 启用 agent 模式
actor_rollout_ref.rollout.agent.max_turns=5             # 最多 5 轮交互
actor_rollout_ref.rollout.agent.custom_stop=[]          # 不设置自定义停止符
```

### 4.3 最终配置效果

基于上述配置，`sampling_params` 的实际值为：

```python
sampling_params = SamplingParams(
    n=8,                        # 从训练脚本覆盖
    max_tokens=8192,            # 从 data.max_response_length
    temperature=1.0,            # 从配置文件
    top_k=-1,                   # 从配置文件
    top_p=1.0,                  # 从配置文件
    detokenize=False,           # 硬编码
    logprobs=0,                 # 硬编码
    stop=[],                    # vLLM 默认值（未在配置中设置）
    # ... 其他参数使用默认值
)
```

---

## 5. agent_rollout_loop 中的处理

### 5.1 函数签名

**文件**: `verl/workers/agent/parallel_env.py`

**函数定义** (第 126-128 行):
```python
def agent_rollout_loop(
    config, vllm_engine, vllm_inputs, prompts, multi_modal_inputs, sampling_params
):
```

### 5.2 克隆和修改流程

**代码位置**: 第 131-150 行

```python
# 1. 克隆原始参数（避免修改共享对象）
agent_sampling_params = sampling_params.clone()

# 2. 强制设置 agent 模式需要的参数
agent_sampling_params.detokenize = True                    # ⭐ 需要文本输出
agent_sampling_params.skip_special_tokens = False          # ⭐ 保留特殊 token
agent_sampling_params.spaces_between_special_tokens = False
agent_sampling_params.n = 1                                # ⭐ 每个 turn 只生成一次
agent_sampling_params.include_stop_str_in_output = True    # ⭐ 输出包含停止符

# 3. 计算单次响应的最大长度
max_generated_tokens = min(
    config.agent.single_response_max_tokens,  # 例如 8192
    config.response_length                     # 例如 8192
)
agent_sampling_params.max_tokens = max_generated_tokens

# 4. 合并停止字符串
custom_stop = list(config.agent.custom_stop)
if custom_stop:
    prev_stop = sampling_params.stop if sampling_params.stop else []
    agent_sampling_params.stop = prev_stop + custom_stop
    print(
        f" [DEBUG stop] {type(prev_stop)=}, {type(custom_stop)=}, "
        f"{type(agent_sampling_params.stop)=}"
    )
```

### 5.3 关键修改说明

| 参数 | 原始值 | Agent 模式值 | 修改原因 |
|------|--------|-------------|----------|
| `detokenize` | `False` | `True` | Agent 需要解析文本中的 `<think>` 和 `<action>` 标签 |
| `skip_special_tokens` | `True` | `False` | 需要保留特殊 token 来正确解析结构化输出 |
| `n` | `8` (外层) | `1` | 每个 turn 只生成一个响应，避免组合爆炸 |
| `include_stop_str_in_output` | `False` | `True` | Agent 需要看到停止符来判断生成状态 |
| `stop` | `[]` | `[] + custom_stop` | 合并自定义停止符（如果配置了） |

### 5.4 stop 参数的合并逻辑

```python
# 情况 1: 没有自定义停止符
sampling_params.stop = []
custom_stop = []
agent_sampling_params.stop = [] + [] = []

# 情况 2: 有自定义停止符
sampling_params.stop = []
custom_stop = ["</search>", "```"]
agent_sampling_params.stop = [] + ["</search>", "```"] = ["</search>", "```"]

# 情况 3: 原始参数也有停止符（不常见）
sampling_params.stop = ["<|endoftext|>"]
custom_stop = ["</search>"]
agent_sampling_params.stop = ["<|endoftext|>"] + ["</search>"]
                           = ["<|endoftext|>", "</search>"]
```

### 5.5 使用位置

**环境重置** (第 185 行):
```python
env.reset(prompts, vllm_inputs, n=sampling_params.n)
# 使用原始的 n=8 来扩展批次大小
```

**生成调用** (第 217-221 行):
```python
actions = vllm_engine.generate(
    prompts=active_vllm_inputs,
    sampling_params=agent_sampling_params,  # ⭐ 使用修改后的参数
    use_tqdm=False,
)
```

---

## 6. 完整数据流程

```
┌─────────────────────────────────────────────────────────────────────┐
│ 第 1 步: 配置文件 (YAML)                                            │
│ ──────────────────────────────────────────────────────────────────  │
│ actor_rollout_ref.rollout:                                          │
│   temperature: 1.0                                                  │
│   top_k: -1                                                         │
│   top_p: 1.0                                                        │
│   n: 1  (训练脚本覆盖为 8)                                          │
│   response_length: 8192                                             │
│   agent:                                                            │
│     custom_stop: []                                                 │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 第 2 步: vLLMRollout.__init__()                                    │
│ ──────────────────────────────────────────────────────────────────  │
│ 文件: verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:203   │
│                                                                     │
│ kwargs = {                                                          │
│   n: 8,                     # 从配置覆盖                           │
│   logprobs: 0,              # 硬编码                               │
│   max_tokens: 8192,         # 从 config.response_length            │
│   detokenize: False,        # 硬编码                               │
│   temperature: 1.0,         # 从配置                               │
│   top_k: -1,                # 从配置                               │
│   top_p: 1.0,               # 从配置                               │
│ }                                                                   │
│ self.sampling_params = SamplingParams(**kwargs)                    │
│ # sampling_params.stop = [] (vLLM 默认值)                          │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 第 3 步: vLLMRollout.generate_sequences()                          │
│ ──────────────────────────────────────────────────────────────────  │
│ 文件: verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:311   │
│                                                                     │
│ if self.config.agent.activate_agent:                               │
│     agent_proto = agent_rollout_loop(                              │
│         config=self.config,                                        │
│         vllm_engine=self.inference_engine,                         │
│         vllm_inputs=vllm_inputs,                                   │
│         prompts=prompts,                                           │
│         multi_modal_inputs=...,                                    │
│         sampling_params=self.sampling_params  # ⭐ 传递参数         │
│     )                                                               │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 第 4 步: agent_rollout_loop() - 克隆和修改                         │
│ ──────────────────────────────────────────────────────────────────  │
│ 文件: verl/workers/agent/parallel_env.py:131-150                   │
│                                                                     │
│ # 克隆原始参数                                                      │
│ agent_sampling_params = sampling_params.clone()                    │
│                                                                     │
│ # 修改为 agent 模式需要的设置                                      │
│ agent_sampling_params.detokenize = True                            │
│ agent_sampling_params.skip_special_tokens = False                  │
│ agent_sampling_params.n = 1                                        │
│ agent_sampling_params.include_stop_str_in_output = True            │
│ agent_sampling_params.max_tokens = min(8192, 8192) = 8192          │
│                                                                     │
│ # 合并停止字符串                                                    │
│ prev_stop = sampling_params.stop or []  # []                       │
│ custom_stop = list(config.agent.custom_stop)  # []                 │
│ agent_sampling_params.stop = [] + [] = []                          │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 第 5 步: 批次扩展 (使用原始 n)                                     │
│ ──────────────────────────────────────────────────────────────────  │
│ 文件: verl/workers/agent/parallel_env.py:185-200                   │
│                                                                     │
│ env.reset(prompts, vllm_inputs, n=sampling_params.n)  # n=8        │
│                                                                     │
│ # 为每个原始样本创建 8 个副本                                      │
│ for i in range(batch_size):                                        │
│     for _ in range(sampling_params.n):  # 8 次                     │
│         vllm_input_list.append(deepcopy(vllm_inputs[i]))           │
│         # ... 复制其他状态                                         │
│                                                                     │
│ # 结果: batch_size × 8 个并行 agent 实例                           │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 第 6 步: 多轮生成循环                                              │
│ ──────────────────────────────────────────────────────────────────  │
│ 文件: verl/workers/agent/parallel_env.py:204-358                   │
│                                                                     │
│ for step in range(config.agent.max_turns):  # 最多 5 轮            │
│     # 生成 action (使用 agent_sampling_params)                     │
│     actions = vllm_engine.generate(                                │
│         prompts=active_vllm_inputs,                                │
│         sampling_params=agent_sampling_params,  # ⭐ n=1           │
│         use_tqdm=False,                                            │
│     )                                                               │
│                                                                     │
│     # 执行工具调用                                                  │
│     obs_results = env.step(active_indices, actions)                │
│     observations, rewards, dones, info = obs_results               │
│                                                                     │
│     # 更新状态、拼接 token、判断是否结束                           │
│     for idx, obs, act, rew, done in zip(...):                      │
│         # ... 状态更新逻辑                                         │
│                                                                     │
│ # 结果: 每个并行实例生成一条完整的多轮交互轨迹                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 第 7 步: 返回结果                                                   │
│ ──────────────────────────────────────────────────────────────────  │
│ 返回 DataProto 包含:                                                │
│   - response: 生成的响应 token                                      │
│   - action_mask: 动作掩码（区分 action 和 observation）            │
│   - attention_mask: 注意力掩码                                      │
│   - position_ids: 位置编码                                          │
│   - env_reward: 环境奖励                                            │
│   - tool_cnt: 工具调用次数                                          │
│   - multi_modal_inputs: 多模态输入（图像、视频）                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. 重要属性对比

### 7.1 参数对比表

| 参数名 | 原始 sampling_params | agent_sampling_params | 说明 |
|--------|---------------------|----------------------|------|
| **n** | `8` (配置) | **`1`** | 外层控制轨迹数量，内层每步只生成一次 |
| **max_tokens** | `8192` | `min(8192, 8192) = 8192` | Agent 可能限制单次响应长度 |
| **temperature** | `1.0` | `1.0` (继承) | 采样温度保持不变 |
| **top_k** | `-1` | `-1` (继承) | Top-k 采样保持不变 |
| **top_p** | `1.0` | `1.0` (继承) | Top-p 采样保持不变 |
| **detokenize** | `False` | **`True`** | Agent 需要文本输出来解析标签 |
| **skip_special_tokens** | `True` (默认) | **`False`** | Agent 需要保留特殊 token |
| **spaces_between_special_tokens** | `True` (默认) | **`False`** | 紧凑的特殊 token 格式 |
| **include_stop_str_in_output** | `False` (默认) | **`True`** | Agent 需要看到停止符 |
| **stop** | `[]` | `[] + custom_stop` | 合并自定义停止符 |
| **logprobs** | `0` | `0` (继承) | 不返回 log 概率 |

### 7.2 为什么需要这些修改？

#### detokenize = True
```python
# Agent 需要解析文本中的结构化标签
response = "<think>我需要查看 100-200 帧</think><action>choose frames between 100 and 200</action>"
think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
action_match = re.search(r"<action>(.*?)</action>", response, re.DOTALL)
```

#### skip_special_tokens = False
```python
# 需要保留 <|im_start|>, <|im_end|> 等特殊 token 来正确解析对话结构
# 例如 Qwen 的对话格式:
# <|im_start|>system\n...<|im_end|>
# <|im_start|>user\n...<|im_end|>
# <|im_start|>assistant\n...<|im_end|>
```

#### n = 1
```python
# 避免组合爆炸:
# 如果 outer_n=8, inner_n=8, max_turns=5
# 总轨迹数 = 8 × 8^5 = 262,144 条 ❌

# 正确做法:
# outer_n=8 在初始化时扩展，inner_n=1 每步生成一次
# 总轨迹数 = 8 × 1^5 = 8 条 ✅
```

#### include_stop_str_in_output = True
```python
# Agent 可能需要看到停止符来判断生成是否完整
# 例如，如果设置 stop=["```"], 生成代码块时:
output_with_stop = "print('hello')\n```"  # 包含停止符
output_without_stop = "print('hello')\n"   # 不包含停止符
# Agent 可以根据是否有 ``` 来判断代码块是否结束
```

---

## 8. 设计原理解析

### 8.1 为什么要克隆 sampling_params？

```python
agent_sampling_params = sampling_params.clone()
```

**原因**:
1. **避免污染原始对象**: `sampling_params` 可能被多个地方共享
2. **保持原始配置**: 其他非 agent 数据可能需要原始参数
3. **线程安全**: 在并行环境中修改共享对象可能导致竞争条件

### 8.2 为什么 agent_sampling_params.n = 1？

这是一个非常重要的设计决策：

#### 问题场景
```
假设有 4 个问题，每个问题需要生成 8 条轨迹，每条轨迹最多 5 轮交互
```

#### ❌ 错误方案: 内层也用 n=8
```python
# 每个 turn 生成 8 个分支
turn_1: 4 × 8 = 32 个分支
turn_2: 32 × 8 = 256 个分支
turn_3: 256 × 8 = 2,048 个分支
turn_4: 2,048 × 8 = 16,384 个分支
turn_5: 16,384 × 8 = 131,072 个分支  # 指数爆炸！
```

#### ✅ 正确方案: 外层扩展，内层串行
```python
# 初始化时扩展批次
for i in range(batch_size):           # 4 个问题
    for _ in range(sampling_params.n): # 8 条轨迹
        vllm_input_list.append(...)    # 创建 4×8=32 个并行实例

# 每轮生成时，每个实例只生成一次 (n=1)
turn_1: 32 个实例，各生成 1 次 = 32 条轨迹
turn_2: 32 个实例，各生成 1 次 = 32 条轨迹
turn_3: 32 个实例，各生成 1 次 = 32 条轨迹
turn_4: 32 个实例，各生成 1 次 = 32 条轨迹
turn_5: 32 个实例，各生成 1 次 = 32 条轨迹

# 最终: 32 条完整的多轮交互轨迹（每个问题 8 条）
```

### 8.3 外层并行 vs 内层串行

```
┌─────────────────────────────────────────────────────────────────┐
│ 问题 1                                                          │
├─────────────────────────────────────────────────────────────────┤
│ 轨迹 1.1: Turn 1 → Turn 2 → Turn 3 → Turn 4 → Turn 5  (串行)  │
│ 轨迹 1.2: Turn 1 → Turn 2 → Turn 3 → Turn 4 → Turn 5  (串行)  │
│ 轨迹 1.3: Turn 1 → Turn 2 → Turn 3 → Turn 4 → Turn 5  (串行)  │
│ ...                                                             │
│ 轨迹 1.8: Turn 1 → Turn 2 → Turn 3 → Turn 4 → Turn 5  (串行)  │
└─────────────────────────────────────────────────────────────────┘
         ↑ 8 条轨迹并行执行 (外层并行)

每个 Turn 内: n=1，只生成一次 (内层串行)
```

**优势**:
1. **可控的计算量**: 总轨迹数 = `batch_size × outer_n`
2. **高效的并行**: 所有轨迹可以同时进行，充分利用 GPU
3. **逻辑清晰**: 每条轨迹是独立的推理链，易于理解和调试
4. **内存可控**: 避免指数级的内存占用

### 8.4 stop 参数的灵活性

```python
# 默认不设置停止符
config.agent.custom_stop = []
agent_sampling_params.stop = []

# 可以通过配置添加自定义停止符
config.agent.custom_stop = ["</search>", "```", "</function>"]
agent_sampling_params.stop = ["</search>", "```", "</function>"]

# 如果原始参数也有停止符（不常见），会合并
sampling_params.stop = ["<|endoftext|>"]
config.agent.custom_stop = ["</search>"]
agent_sampling_params.stop = ["<|endoftext|>", "</search>"]
```

**用途**:
- 限制工具调用的格式
- 防止生成过长的无关内容
- 强制特定的输出结构

---

## 总结

### 关键要点

1. **sampling_params 是 vLLM 的采样参数类**，控制文本生成行为

2. **sampling_params.stop 是停止字符串列表**，默认为空，可通过配置添加

3. **初始化在 vLLMRollout.__init__()**，从配置文件读取参数

4. **agent_rollout_loop 会克隆并修改参数**，适配 agent 模式需求

5. **agent_sampling_params.n=1 是关键设计**，避免组合爆炸，保持可控的并行度

6. **外层并行 + 内层串行**，实现高效的多轨迹多轮交互生成

### 配置建议

```yaml
# 推荐配置
actor_rollout_ref:
  rollout:
    n: 8                    # 每个样本生成 8 条轨迹
    temperature: 1.0        # 保持一定的随机性
    top_p: 1.0              # 使用 nucleus 采样
    agent:
      activate_agent: True
      max_turns: 5          # 最多 5 轮交互
      custom_stop: []       # 根据需要添加停止符
```

### 调试技巧

```python
# 打印 sampling_params 信息
print(f"Original n: {sampling_params.n}")
print(f"Original stop: {sampling_params.stop}")
print(f"Original detokenize: {sampling_params.detokenize}")

# 打印 agent_sampling_params 修改
print(f"Agent n: {agent_sampling_params.n}")
print(f"Agent stop: {agent_sampling_params.stop}")
print(f"Agent detokenize: {agent_sampling_params.detokenize}")

# 打印批次大小
print(f"Batch size: {batch_size}")
print(f"Total parallel instances: {batch_size * sampling_params.n}")
```

---

**文档版本**: 1.0
**最后更新**: 2025-01-10
**维护者**: FrameThinker Team
