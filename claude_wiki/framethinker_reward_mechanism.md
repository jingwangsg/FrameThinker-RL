# FrameThinker Reward计算机制

## 概述

FrameThinker在RL训练过程中使用基于规则的reward函数，该函数位于 `verl/utils/reward_score/think_with_video_reward.py`。

## Reward组成

Reward由**4个分数**组成（在 `think_with_video_reward.py:19`）：

```python
def v11(predict_str: str, ground_truth: str, extra_info=None):
    format_score = 0.0
    acc_score = 0.0
    other_score = 0.0
    total_score = 0.0
```

### 1. Format Score（格式分数）

**作用**: 检查模型输出是否符合规范格式

**取值**:
- 格式正确: `1.0`
- 格式错误: `0.0`

**检查规则** (必须全部满足):

#### 1.1 基本结构检查
```python
think_contents = re.findall(r'<think>(.*?)</think>', predict_str, re.DOTALL)
action_contents = re.findall(r'<action>(.*?)</action>', predict_str, re.DOTALL)
system_responses = re.findall(r'</action>(.*?)<think>', predict_str, re.DOTALL)
```

- 必须有 `<think>` 和 `<action>` 标签
- `<think>` 和 `<action>` 数量必须相等
- 至少要有一对 `<think>/<action>`

#### 1.2 最终输出格式
```python
last_action = action_contents[-1].strip()
answer_match = re.match(r'output answer:\s*(\S+)', last_action)
```

- 最后一个 `<action>` 必须是 `output answer: <答案>` 格式

#### 1.3 帧选择约束
```python
frame_match = re.match(r'choose frames between (\d+) and (\d+)', action)
```

对于每个 `choose frames between X and Y` 的action:

1. **范围充分性**: `Y - X >= nframes` (默认 nframes=8)
   - 确保选择的范围足够大，能够提取足够的帧

2. **无重复选择**:
   ```python
   if current_pair in action_frame_pairs:
       return 全0
   ```
   - 不能重复选择相同的帧范围
   - 还会检查 `(X+1, Y-1)` 避免过度重叠

3. **Think-Action一致性**:
   ```python
   numbers_in_think = re.findall(r'(?<!:)\b(\d+)\b(?!:)', think_block)
   if (num_a, num_b) != current_pair:
       return 全0
   ```
   - `<think>` 块中最后两个数字必须与 `<action>` 中的范围一致

#### 1.4 时间戳查询约束
```python
time_match = re.match(r'get frame number at time\s+(\d{1,3}:\d{2})', action)
```

对于每个 `get frame number at time HH:MM` 的action:

1. **无重复查询**: 不能查询相同的时间戳

2. **后续帧选择一致性**:
   ```python
   system_response = system_responses[i].strip()
   response_match = re.search(r'is:\s*(\d+)', system_response)
   expected_frame_in_next_action = int(response_match.group(1))
   ```
   - 如果查询了时间戳，下一个 `choose frames` action的范围必须包含返回的帧号
   - 即: `start_f <= expected_frame <= end_f`

#### 1.5 末尾检查
- 在最后的 `output answer` 之前，不能有未处理的时间戳查询
- 所有的时间戳查询都必须对应后续的帧选择

### 2. Accuracy Score（准确率分数）

**前置条件**: 只有在 `format_score = 1.0` 时才计算

**计算逻辑**:
```python
model_answer = answer_match.group(1)
if model_answer == ground_truth:
    acc_score = 1.0
else:
    acc_score = 0.0
```

**取值**:
- 答案正确: `1.0`
- 答案错误: `0.0`

### 3. Other Score（其他奖励分数）

**前置条件**: 只有在 `acc_score = 1.0` 时才给予额外奖励

**奖励组成**:

#### 3.1 时间查询奖励 (+0.5)
```python
if time_reward:
    time_pattern = re.compile(r'\d+:\d+(:\d+)?')
    if re.search(time_pattern, question):
        other_score += 0.5
```

**条件**:
- 使用了 `get frame number at time` 功能 (time_reward=True)
- 且问题中包含时间模式（如 "1:23"、"12:34:56"）

**目的**: 鼓励模型在问题涉及时间时主动使用时间查询工具

#### 3.2 多步推理奖励 (+0.02)
```python
if len(action_frame_pairs) > 1:
    other_score += 0.02
```

**条件**: 进行了多次帧选择操作

**目的**: 鼓励模型进行多步骤的视频分析，而非一次性选择

### 4. Total Score（总分）

```python
total_score = acc_score + other_score
```

**取值范围**:
- 最低: `0.0` (答案错误或格式错误)
- 标准: `1.0` (答案正确，无额外奖励)
- 最高: `1.52` (答案正确 + 时间查询奖励 + 多步推理奖励)

## Parse失败处理

在rollout过程中，如果样本无法正确parse，会**立即返回全0的reward**:

```python
return (0.0, 0.0, 0.0, 0.0)  # (total_score, acc_score, format_score, other_score)
```

### 触发全0 Reward的情况

1. **正则表达式解析失败**
   ```python
   try:
       think_contents = re.findall(...)
       action_contents = re.findall(...)
   except Exception as e:
       return (0.0, 0.0, 0.0, 0.0)
   ```

2. **缺少必要标签**
   ```python
   if not think_contents or not action_contents:
       return (0.0, 0.0, 0.0, 0.0)
   ```

3. **标签数量不匹配**
   ```python
   if len(think_contents) != len(action_contents):
       return (0.0, 0.0, 0.0, 0.0)
   ```

4. **最终输出格式错误**
   ```python
   if not answer_match:
       return (0.0, 0.0, 0.0, 0.0)
   ```

5. **帧选择违规**
   - 重复选择相同帧范围
   - 帧范围太小 (`num1 >= num2 - nframes`)
   - Think-Action不一致

6. **时间戳查询违规**
   - 重复查询相同时间
   - 时间戳查询后的帧选择不包含返回的帧号

7. **未完成的时间戳查询**
   - 在最后的 `output answer` 前还有未处理的时间戳查询

## 代码位置

### 主要函数
- **计算入口**: `verl/utils/reward_score/think_with_video_reward.py:11` - `compute_score()`
- **核心逻辑**: `verl/utils/reward_score/think_with_video_reward.py:19` - `v11()`

### 调用路径
1. **训练入口**: `verl/trainer/main_ppo.py:181` - 使用 `get_custom_reward_fn()` 或默认函数
2. **Reward Manager**: `verl/workers/reward_manager/naive.py:85` - 调用 `compute_score()`
3. **批量处理**: `verl/workers/reward_manager/batch.py:29` - `verify()` 方法

### 数据源识别
在 `verl/utils/reward_score/__init__.py:65-66`:
```python
elif data_source in ['vstar', 'vl_agent', 'chart']:
    from . import think_with_video_reward
    return think_with_video_reward.compute_score(solution_str, ground_truth, extra_info)
```

## 设计理念

### 严格的格式要求
- 通过 `format_score` 强制模型学习正确的输出格式
- 即使答案可能正确，格式错误也得不到任何奖励
- 确保输出可被下游系统可靠解析

### 鼓励正确的推理模式
- **时间查询奖励**: 引导模型在涉及时间的问题中使用工具
- **多步推理奖励**: 鼓励渐进式的视频分析，而非盲目猜测

### 零容忍的错误处理
- 任何违反规则的行为都导致全0 reward
- 强化学习信号明确：要么完全正确，要么完全失败
- 避免模型学习到不完整或不规范的行为模式

## 训练配置

在 `examples/agent/train_frame_thinker.sh` 中，FrameThinker使用：
- **Reward Manager**: `naive` (默认，从配置文件 `verl/trainer/config/ppo_trainer.yaml` 中设置)
- **算法**: GRPO (`algorithm.adv_estimator=grpo`)
- **KL惩罚**: 关闭 (`algorithm.kl_ctrl.kl_coef=0.0`)

## 注意事项

1. **extra_info依赖**: `other_score` 的时间奖励需要 `extra_info['question']`，确保数据集提供此字段

2. **nframes参数**: 默认为8，可能需要根据视频帧率和问题类型调整

3. **答案格式**: ground_truth应该是单个token或短语，匹配使用精确字符串比较

4. **调试输出**: `compute_score()` 会打印详细信息到控制台，便于调试:
   ```python
   print("========================start of text========================")
   print_answer(predict_str, ground_truth)
   print("scores:", a, b, c, d)
   print("========================end of text========================")
   ```
