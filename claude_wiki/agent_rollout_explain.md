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
9. [Done 状态判断机制](#9-done-状态判断机制)

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

### 2.2 默认停止机制：EOS Token

虽然 `sampling_params.stop = []`，但模型仍然会在合适的时候停止生成。这是因为 **vLLM 会自动使用 tokenizer 的 `eos_token` 作为默认停止条件**。

#### 2.2.1 对于 Qwen2.5-VL-7B-Instruct

```python
# Tokenizer 的特殊 token 配置
tokenizer.eos_token = '<|im_end|>'           # EOS token (结束标记)
tokenizer.eos_token_id = 151645              # EOS token ID
tokenizer.pad_token = '<|endoftext|>'        # Padding token
```

#### 2.2.2 停止条件的完整逻辑

vLLM 的停止条件由以下几部分组成：

```python
# 实际停止条件 = 自动 EOS + 自定义停止符
实际停止条件 = {
    # 1. 自动添加（如果 ignore_eos=False）
    tokenizer.eos_token_id,                  # 151645: '<|im_end|>'

    # 2. 用户指定的字符串停止符
    *SamplingParams.stop,                    # 例如: ["</search>", "```"]

    # 3. 用户指定的 token ID 停止符
    *SamplingParams.stop_token_ids,          # 例如: [151643]
}
```

#### 2.2.3 关键参数：ignore_eos

**配置位置**: `verl/trainer/config/ppo_trainer.yaml:97`

```yaml
rollout:
  ignore_eos: False  # 默认值，不忽略 EOS token
```

**行为说明**:

| ignore_eos | 行为 | 使用场景 |
|-----------|------|---------|
| `False` (默认) | 遇到 `tokenizer.eos_token_id` 时停止 | 正常对话生成 |
| `True` | 忽略 EOS token，继续生成直到 `max_tokens` | 强制生成固定长度 |

**示例**:
```python
# ignore_eos=False (默认)
prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
output = "Hi there!<|im_end|>"  # ✅ 在 <|im_end|> 处自动停止

# ignore_eos=True
prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
output = "Hi there!<|im_end|><|im_start|>user\n..."  # ❌ 继续生成，破坏格式
```

#### 2.2.4 为什么代码能正常停止？

即使没有显式设置 `stop=["<|im_end|>"]`，模型也能正确停止，原因是：

```python
# verl/trainer/config/ppo_trainer.yaml
rollout:
  ignore_eos: False  # ⭐ 不忽略 EOS token

agent:
  custom_stop: []    # ⭐ 不需要额外的停止符
```

因此，实际的停止条件为：
```python
# 自动生效（因为 ignore_eos=False）
停止条件 = ['<|im_end|>']  # tokenizer.eos_token
```

这也是为什么 Qwen 模型能够正确地在每个回复结束时停止，遵循 ChatML 对话格式。

### 2.3 典型使用场景

```python
# 示例 1: 限制代码块生成
sampling_params.stop = ["```"]

# 示例 2: 限制工具调用
sampling_params.stop = ["</search>", "</function>", "</tool>"]

# 示例 3: 限制多轮对话
sampling_params.stop = ["<|im_end|>", "<|endoftext|>"]
```

### 2.4 在 FrameThinker 中的使用

默认情况下，`sampling_params.stop = []`，即不设置**额外**的停止条件（但 `<|im_end|>` 会自动生效）。

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

2. **默认停止机制**：
   - 虽然 `sampling_params.stop = []`，但 vLLM 会自动使用 `tokenizer.eos_token` 作为停止条件
   - 对于 Qwen2.5-VL，默认停止 token 是 `<|im_end|>` (token ID: 151645)
   - 通过 `ignore_eos=False` 控制是否启用自动 EOS 停止（默认启用）

3. **sampling_params.stop 是额外的停止字符串列表**，默认为空，可通过 `custom_stop` 配置添加

4. **初始化在 vLLMRollout.__init__()**，从配置文件读取参数

5. **agent_rollout_loop 会克隆并修改参数**，适配 agent 模式需求

6. **agent_sampling_params.n=1 是关键设计**，避免组合爆炸，保持可控的并行度

7. **外层并行 + 内层串行**，实现高效的多轨迹多轮交互生成

### 配置建议

```yaml
# 推荐配置
actor_rollout_ref:
  rollout:
    n: 8                    # 每个样本生成 8 条轨迹
    temperature: 1.0        # 保持一定的随机性
    top_p: 1.0              # 使用 nucleus 采样
    ignore_eos: False       # ⭐ 启用自动 EOS 停止（默认值，强烈推荐保持）
    agent:
      activate_agent: True
      max_turns: 5          # 最多 5 轮交互
      custom_stop: []       # 根据需要添加额外停止符（可选）
```

**注意事项**:
- 保持 `ignore_eos: False` 以确保模型在 `<|im_end|>` 处正确停止
- 只有在需要限制特定格式时才添加 `custom_stop`
- 不要将 `<|im_end|>` 添加到 `custom_stop`，因为它已经自动生效

### 调试技巧

```python
# 1. 打印 sampling_params 信息
print(f"Original n: {sampling_params.n}")
print(f"Original stop: {sampling_params.stop}")
print(f"Original ignore_eos: {sampling_params.ignore_eos}")
print(f"Original detokenize: {sampling_params.detokenize}")

# 2. 打印 agent_sampling_params 修改
print(f"Agent n: {agent_sampling_params.n}")
print(f"Agent stop: {agent_sampling_params.stop}")
print(f"Agent ignore_eos: {agent_sampling_params.ignore_eos}")
print(f"Agent detokenize: {agent_sampling_params.detokenize}")

# 3. 检查 tokenizer 的 EOS token
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
print(f"EOS token: {tokenizer.eos_token}")
print(f"EOS token ID: {tokenizer.eos_token_id}")

# 4. 打印批次大小
print(f"Batch size: {batch_size}")
print(f"Total parallel instances: {batch_size * sampling_params.n}")

# 5. 验证停止条件
print(f"\n实际停止条件:")
if not sampling_params.ignore_eos:
    print(f"  - EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
if sampling_params.stop:
    print(f"  - Custom stop strings: {sampling_params.stop}")
if not sampling_params.ignore_eos or sampling_params.stop:
    print(f"  总计: {'自动 EOS' if not sampling_params.ignore_eos else ''}" +
          f"{' + ' if not sampling_params.ignore_eos and sampling_params.stop else ''}" +
          f"{'自定义停止符' if sampling_params.stop else ''}")
else:
    print(f"  - 仅 max_tokens 限制")
```

---

## 9. Done 状态判断机制

### 9.1 Done 状态概述

#### 9.1.1 什么是 Done 状态？

在 Agent Rollout 中，`done` 是一个布尔值标志，表示某个 rollout 实例是否应该停止继续交互。当 `done=True` 时，该实例会被标记为不活跃（`active_mask[idx] = False`），不再参与后续的生成轮次。

#### 9.1.2 Done 状态的作用

- **提前终止轨迹**：当模型给出最终答案或遇到错误时，无需继续执行
- **节省计算资源**：减少无效的生成和工具调用
- **保证轨迹质量**：避免超出长度限制或格式错误导致的无效轨迹

### 9.2 Done 状态判断的三个层级

Done 状态的判断分为三个层级，按照执行顺序依次为：

```
┌─────────────────────────────────────────────────────────────────┐
│ 层级 1: vLLM 生成层（finish_reason）                            │
│ ──────────────────────────────────────────────────────────────  │
│ 文件: verl/workers/agent/parallel_env.py:548-560               │
│ 判断条件:                                                        │
│   - finish_reason == "length" → done=True                      │
│   - len(token_ids) == 0 → done=True                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 层级 2: Tool 执行层（tool.execute）                             │
│ ──────────────────────────────────────────────────────────────  │
│ 文件: verl/workers/agent/envs/visual_agent/think_with_video.py │
│ 判断条件:                                                        │
│   - 无效的 action 格式 → done=True                              │
│   - 帧范围超出视频范围 → done=True                              │
│   - 有效的工具调用 → done=False                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 层级 3: Agent Loop 层（长度和轮次检查）                         │
│ ──────────────────────────────────────────────────────────────  │
│ 文件: verl/workers/agent/parallel_env.py:268-358               │
│ 判断条件:                                                        │
│   - 序列长度 >= max_total_length → active_mask=False           │
│   - step == max_turns - 1 → active_mask=False                  │
│   - done=True (来自上层) → active_mask=False                    │
└─────────────────────────────────────────────────────────────────┘
```

### 9.3 完整判断流程

```
开始生成
    ↓
vLLM 生成 action
    ↓
[层级 1] 检查 finish_reason
    ├─ finish_reason == "length"? ──→ done=True ──→ 跳过工具调用
    ├─ len(token_ids) == 0? ──→ done=True ──→ 跳过工具调用
    └─ 否 → done=False ──→ 继续
    ↓
执行工具调用 (tool.execute)
    ↓
[层级 2] 工具返回 done 状态
    ├─ 解析 action 失败? ──→ done=True
    ├─ 帧范围无效? ──→ done=True
    ├─ 时间戳查询? ──→ done=False (返回帧号，继续)
    ├─ zoom in frame? ──→ done=False (返回单帧，继续)
    └─ choose frames? ──→ done=False (返回多帧，继续)
    ↓
合并 done 状态: done_list[idx] |= done
    ↓
[层级 3] Agent Loop 检查
    ├─ 序列长度 >= max_total_length? ──→ active_mask[idx]=False
    ├─ step == max_turns - 1? ──→ active_mask[idx]=False
    ├─ done == True? ──→ active_mask[idx]=False
    └─ 否 → 添加 observation，继续下一轮
    ↓
检查 active_mask
    ├─ sum(active_mask) == 0? ──→ 所有实例完成，退出循环
    └─ 否 → 进入下一个 turn
```

### 9.4 具体判断条件

#### 9.4.1 层级 1：vLLM 生成层

**文件位置**: `verl/workers/agent/parallel_env.py:548-560`

**判断条件详解**:

| 条件 | 含义 | Done 值 | 原因 |
|------|------|---------|------|
| `finish_reason == "length"` | 生成达到 `max_tokens` 限制 | `True` | 无法继续生成有效内容 |
| `len(token_ids) == 0` | 没有生成任何 token | `True` | 模型可能遇到错误或停止 |
| 其他情况 | 正常生成 | `False` (初始) | 需要进一步检查工具执行结果 |

#### 9.4.2 层级 2：Tool 执行层

**文件位置**: `verl/workers/agent/envs/visual_agent/think_with_video.py:42-142`

**Done 条件总结**:

| 条件 | Done 值 | 说明 |
|------|---------|------|
| 无法解析 action 格式 | `True` | 正则匹配失败 |
| `start_frame > total_frames` | `True` | 起始帧超出视频范围 |
| `start_frame >= end_frame - num_frames_per_sample` | `True` | 范围太小，无法采样 |
| `len(focused_images_data) == 0` | `True` | 没有提取到任何帧 |
| 时间戳格式错误 | `True` | 无法解析时间戳 |
| 视频读取失败 | `True` | 帧索引超出范围或读取异常 |
| 成功提取帧 | `False` | 返回观察，继续交互 |

**关键点**: 使用 `|=` 运算符合并 done 状态
```python
done_list[subidx] |= done
```
- 如果层级 1 已经设置 `done_list[subidx] = True`，则无论工具返回什么，都会保持 `True`
- 如果层级 1 设置 `done_list[subidx] = False`，则由工具执行结果决定最终的 done 值

#### 9.4.3 层级 3：Agent Loop 层

**文件位置**: `verl/workers/agent/parallel_env.py:268-358`

**四个检查点**:

1. **检查点 1**: 生成 action 后的序列长度
   - 条件: `running_states[idx].shape[-1] >= max_total_length`
   - 结果: `active_mask[idx] = False`

2. **检查点 2**: Done 状态或达到最大轮次
   - 条件: `done == True` 或 `step == config.agent.max_turns - 1`
   - 结果: `active_mask[idx] = False`

3. **检查点 3**: 添加 observation 前的长度检查
   - 条件: 添加 observation 后会超过 `max_total_length`
   - 结果: `active_mask[idx] = False`，**不添加该 observation**

4. **检查点 4**: 添加 observation 后的最终检查
   - 条件: 添加 observation 后达到或超过 `max_total_length`
   - 结果: `active_mask[idx] = False`

### 9.5 示例场景

#### 9.5.1 场景 1: 正常的多轮交互

```
Turn 1:
  Action: "choose frames between 100 and 200"
  Tool Result: 返回 8 帧图像
  Done: False
  → 继续下一轮

Turn 2:
  Action: "choose frames between 150 and 170"
  Tool Result: 返回 8 帧图像
  Done: False
  → 继续下一轮

Turn 3:
  Action: "output answer: A"
  Tool Result: 无法解析（不是有效的工具调用）
  Done: True
  → 停止该实例
```

#### 9.5.2 场景 2: 达到最大轮次

```
Turn 5 (max_turns=5, step=4):
  Action: "choose frames between 400 and 500"
  Done: False
  但是 step == max_turns - 1
  → 强制停止该实例
```

#### 9.5.3 场景 3: 无效的帧范围

```
Turn 2:
  Action: "choose frames between 5000 and 6000"
  Tool Result: start_frame (5000) > total_frames (3000)
  Done: True
  → 停止该实例
```

#### 9.5.4 场景 4: 序列长度超限

```
Turn 3:
  Action: "choose frames between 200 and 300"
  添加 action tokens 后: Total Length: 6100 tokens
  检查: 6100 >= max_total_length (6000)?
  → 是，停止该实例（不添加 observation）
```

#### 9.5.5 场景 5: 生成达到长度限制

```
Turn 1:
  Action: "choose frames between 0 and 100 and then I need to analyze..."
  vLLM 生成: 达到 max_tokens=8192
  finish_reason: "length"
  Done: True (在层级 1 设置)
  → 跳过工具调用，直接停止该实例
```

### 9.6 Done 状态判断总结

#### 9.6.1 关键点

1. **三层防御**：vLLM 生成层 → Tool 执行层 → Agent Loop 层
2. **逻辑 OR 合并**：`done_list[idx] |= done` 确保任何一层返回 True 都会停止
3. **序列长度优先**：即使 `done=False`，长度超限也会强制停止
4. **最大轮次保底**：即使一直没有 done，也会在最后一轮强制停止

#### 9.6.2 Done=True 的所有情况

| 层级 | 条件 | Done 值 |
|------|------|---------|
| 层级 1 | `finish_reason == "length"` | `True` |
| 层级 1 | `len(token_ids) == 0` | `True` |
| 层级 2 | 无法解析 action 格式 | `True` |
| 层级 2 | 帧范围超出视频范围 | `True` |
| 层级 2 | 帧范围太小 | `True` |
| 层级 2 | 没有提取到任何帧 | `True` |
| 层级 2 | 时间戳格式错误 | `True` |
| 层级 2 | 视频读取失败 | `True` |
| 层级 3 | `done == True` (来自上层) | `active_mask=False` |
| 层级 3 | `step == max_turns - 1` | `active_mask=False` |
| 层级 3 | 序列长度 >= `max_total_length` | `active_mask=False` |

#### 9.6.3 配置参数

```yaml
actor_rollout_ref:
  rollout:
    response_length: 8192  # max_total_length = prompt_length + response_length
    agent:
      max_turns: 5  # 最大交互轮次
      single_response_max_tokens: 8192  # 单次生成的最大 token 数
```

---

**文档版本**: 1.2
**最后更新**: 2025-01-11
**更新内容**:
- v1.1: 添加了默认 EOS token 停止机制的详细说明
- v1.2: 合并了 Done 状态判断机制的完整说明
**维护者**: FrameThinker Team
