# Custom Reward Function 机制详解

## 概述

FrameThinker-RL 训练框架提供了两种配置 reward function 的方式：
1. **基于 `data_source` 的自动路由机制**（隐式配置）
2. **显式指定自定义 reward function**（显式配置）

本文档详细说明这两种机制的工作原理、配置方法和参数差异。

---

## 一、两种配置方式对比

### 1.1 方式一：基于 `data_source` 自动路由（train_frame_thinker.sh）

#### 配置特点
- **不需要**在命令行显式指定 `custom_reward_function`
- 依赖数据文件（.parquet）中的 `data_source` 字段自动选择 reward function
- 使用函数内置的默认参数，**无法传入自定义参数**

#### 配置示例
```bash
# train_frame_thinker.sh 中没有 custom_reward_function 相关配置
python3 -m verl.trainer.main_ppo \
    data.train_batch_size=32 \
    ... \
    trainer.total_epochs=10 $@
```

#### 默认配置来源
配置文件：`verl/trainer/config/ppo_trainer.yaml:198-201`
```yaml
custom_reward_function:
  path: null                  # null 表示使用自动路由
  name: compute_score
  reward_kwargs: null         # 无法传入自定义参数
```

---

### 1.2 方式二：显式指定自定义 reward function（train_frame_thinker_vhonly.sh）

#### 配置特点
- **显式指定** reward function 的文件路径和函数名
- 可以通过 `reward_kwargs` 传入**自定义参数**
- 提供更大的灵活性，可以精细调整 reward 计算逻辑

#### 配置示例
```bash
# train_frame_thinker_vhonly.sh 第 68-72 行
python3 -m verl.trainer.main_ppo \
    ... \
    custom_reward_function.path=verl/utils/reward_score/think_with_video_reward.py \
    custom_reward_function.name=compute_score \
    +custom_reward_function.reward_kwargs.nframes=8 \
    +custom_reward_function.reward_kwargs.lambda_gfn=0.2 \
    +custom_reward_function.reward_kwargs.lambda_cf=0.0 \
    $@
```

---

## 二、自动路由机制详解

### 2.1 工作流程

```
数据加载 → 读取 data_source 字段 → _default_compute_score 路由 → 调用对应 reward function
```

### 2.2 关键代码位置

#### (1) 配置 reward_fn_key
文件：`verl/trainer/config/ppo_trainer.yaml:7`
```yaml
data:
  reward_fn_key: data_source  # 指定从数据的哪个字段读取 reward function 标识符
```

#### (2) Reward Manager 读取 data_source
文件：`verl/workers/reward_manager/naive.py:76-87`
```python
ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
data_source = data_item.non_tensor_batch[self.reward_fn_key]  # 读取 data_source
extra_info = data_item.non_tensor_batch.get("extra_info", None)

score, acc_score, format_score, other_score = self.compute_score(
    data_source=data_source,      # 传入 data_source
    solution_str=response_str,
    ground_truth=ground_truth,
    extra_info=extra_info,
)
```

#### (3) 自动路由逻辑
文件：`verl/utils/reward_score/__init__.py:17-76`
```python
def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == "openai/gsm8k":
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)

    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
        from . import math
        res = math.compute_score(solution_str, ground_truth)

    # ... 其他 data_source ...

    elif data_source in ['vstar', 'vl_agent', 'chart', 'TencentARC/Video-Holmes']:
        from . import think_with_video_reward
        return think_with_video_reward.compute_score(solution_str, ground_truth, extra_info)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")
```

**关键点**：
- 当 `data_source` 为 `'TencentARC/Video-Holmes'`、`'vstar'`、`'vl_agent'` 或 `'chart'` 时，自动调用 `think_with_video_reward.compute_score`
- **无法传入额外参数**，使用函数内置的默认值

---

### 2.3 显式配置加载逻辑

文件：`verl/trainer/main_ppo.py:26-58`
```python
def get_custom_reward_fn(config):
    import importlib.util
    import sys

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")

    if not file_path:
        return None  # 返回 None，使用 _default_compute_score

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    # 动态加载模块
    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    function_name = reward_fn_config.get("name")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))  # 读取自定义参数

    # 包装函数，注入 reward_kwargs
    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn
```

---

## 三、think_with_video_reward 参数详解

### 3.1 函数签名与默认参数

文件：`verl/utils/reward_score/think_with_video_reward.py:14-23`
```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    nframes = kwargs.get('nframes', 8)        # 默认: 8
    lambda_gfn = kwargs.get('lambda_gfn', 0.5)  # 默认: 0.5
    lambda_cf = kwargs.get('lambda_cf', 0.02)   # 默认: 0.02

    a, b, c, d = v11(solution_str, ground_truth, extra_info, nframes, lambda_gfn, lambda_cf)
    print("========================start of text========================")
    print_answer(solution_str, ground_truth)
    print("scores:", a, b, c, d)
    print("========================end of text========================")
    return a, b, c, d
```

### 3.2 参数详细说明

#### 参数 1: `nframes` (默认: 8)

**作用**：最小帧间隔要求

**使用位置**：`verl/utils/reward_score/think_with_video_reward.py:83-84`
```python
if num1 >= num2 - nframes:
    return total_score, acc_score, format_score, other_score
```

**含义**：
- 每次选择帧范围时，要求 `end_frame - start_frame >= nframes`
- 确保模型选择的帧范围至少包含 `nframes` 帧
- 如果选择的帧范围太窄（< nframes），reward 为 0

**示例**：
- ✅ `choose frames between 10 and 20` （范围 10 帧，≥ 8）
- ❌ `choose frames between 10 and 15` （范围 5 帧，< 8）

---

#### 参数 2: `lambda_gfn` (默认: 0.5)

**作用**："Get Frame Number" 时间戳查询奖励系数

**使用位置**：`verl/utils/reward_score/think_with_video_reward.py:124-127`
```python
if model_answer == ground_truth:
    acc_score = 1.0
    if time_reward:
        time_pattern = re.compile(r'\d+:\d+(:\d+)?')
        if re.search(time_pattern, question):
            other_score += lambda_gfn  # 奖励时间戳查询
```

**触发条件**（需同时满足）：
1. 模型答案正确（`model_answer == ground_truth`）
2. 模型使用了 `get frame number at time MM:SS` 查询
3. 问题文本中包含时间信息（如 "1:30", "0:45:30" 等）

**含义**：
- 鼓励模型在遇到**时间相关问题**时，主动使用时间戳查询工具
- 奖励值会加到 `other_score` 和 `total_score` 中

**示例**：
```
Question: "What happened at 1:30 in the video?"
Model action: get frame number at time 1:30
→ 如果答对，获得 +0.5 奖励（默认 lambda_gfn=0.5）
```

---

#### 参数 3: `lambda_cf` (默认: 0.02)

**作用**："Choose Frames" 多轮帧选择奖励系数

**使用位置**：`verl/utils/reward_score/think_with_video_reward.py:128-129`
```python
if len(action_frame_pairs) > 1:
    other_score += lambda_cf  # 奖励多次帧选择
```

**触发条件**（需同时满足）：
1. 模型答案正确（`model_answer == ground_truth`）
2. 模型进行了**多次**帧选择（`action_frame_pairs` 长度 > 1）

**含义**：
- 初始时，`action_frame_pairs = [(0, total_frames-1)]`（初始全帧范围）
- 每次成功 `choose frames between X and Y`，添加一个 pair
- 如果 `len(action_frame_pairs) > 1`，说明模型进行了主动的帧选择
- 鼓励模型进行**迭代式帧选择**，而不是一开始就用全部帧回答

**示例**：
```
Turn 1: choose frames between 100 and 200  → action_frame_pairs = [(0,299), (100,200), (101,199)]
Turn 2: choose frames between 140 and 160  → action_frame_pairs 增加
Final: output answer: A
→ 如果答对，获得 +0.02 奖励（默认 lambda_cf=0.02）
```

---

### 3.3 返回值说明

```python
return total_score, acc_score, format_score, other_score
```

| 返回值 | 含义 | 计算公式 |
|--------|------|----------|
| `total_score` | 总分 | `acc_score + other_score` |
| `acc_score` | 答案准确性 | 答对：1.0，答错：0.0 |
| `format_score` | 格式正确性 | 格式正确：1.0，格式错误：0.0 |
| `other_score` | 额外奖励 | `lambda_gfn`（如适用）+ `lambda_cf`（如适用） |

**格式要求**：
- 必须有配对的 `<think>...</think>` 和 `<action>...</action>`
- 最后一个 action 必须是 `output answer: X` 格式
- 帧选择必须符合逻辑（不重复、数字匹配等）

---

## 四、两个脚本的参数对比

### 4.1 参数对比表

| 参数 | train_frame_thinker.sh<br/>（自动路由，默认值） | train_frame_thinker_vhonly.sh<br/>（显式指定） | 差异 |
|------|-----------------------------------------------|-----------------------------------------------|------|
| `nframes` | **8** | **8** | 相同 |
| `lambda_gfn` | **0.5** | **0.2** | ↓ 降低 60% |
| `lambda_cf` | **0.02** | **0.0** | ↓ 完全禁用 |

### 4.2 设计意图分析

#### train_frame_thinker.sh（原始版本）
- `lambda_gfn = 0.5`：**强烈鼓励**时间戳查询行为
- `lambda_cf = 0.02`：**轻微鼓励**多轮帧选择

**适用场景**：
- 训练模型学习主动使用工具（时间戳查询 + 多轮帧选择）
- 希望模型在时间相关问题上使用 "get frame number" 工具
- 鼓励迭代式帧选择策略

---

#### train_frame_thinker_vhonly.sh（调整版本）
- `lambda_gfn = 0.2`：**适度鼓励**时间戳查询
- `lambda_cf = 0.0`：**不鼓励**多轮帧选择

**适用场景**：
- 更关注**答案准确性**（`acc_score`）
- 降低对复杂交互行为的奖励
- 可能用于 Video-Holmes 特定数据集，该数据集可能不需要复杂的帧选择策略
- "vhonly" 可能代表 "Video-Holmes only"

---

## 五、如何选择配置方式

### 5.1 使用自动路由（方式一）的场景

✅ **适合**：
- 数据集中 `data_source` 字段已正确设置
- 使用默认参数即可满足需求
- 不需要精细调整 reward 权重
- 快速实验和原型开发

❌ **不适合**：
- 需要自定义 reward 参数
- 需要针对特定任务调整奖励机制
- 需要消融实验（ablation study）比较不同参数的影响

---

### 5.2 使用显式配置（方式二）的场景

✅ **适合**：
- 需要自定义 `nframes`、`lambda_gfn`、`lambda_cf` 等参数
- 进行消融实验，比较不同奖励策略的效果
- 针对特定数据集或任务优化 reward function
- 需要完全控制 reward 计算逻辑

❌ **不适合**：
- 快速实验，不想手动指定参数
- 对默认参数满意

---

## 六、配置示例

### 6.1 使用默认参数（自动路由）

```bash
# train_frame_thinker.sh
python3 -m verl.trainer.main_ppo \
    "data.train_files=[${TRAIN_FILES}]" \
    "data.val_files=[${VAL_FILES}]" \
    data.train_batch_size=32 \
    # ... 其他配置 ...
    trainer.total_epochs=10 $@

# 使用默认值：nframes=8, lambda_gfn=0.5, lambda_cf=0.02
```

---

### 6.2 自定义参数（显式配置）

```bash
# train_frame_thinker_vhonly.sh
python3 -m verl.trainer.main_ppo \
    "data.train_files=[${TRAIN_FILES}]" \
    "data.val_files=[${VAL_FILES}]" \
    data.train_batch_size=32 \
    # ... 其他配置 ...
    custom_reward_function.path=verl/utils/reward_score/think_with_video_reward.py \
    custom_reward_function.name=compute_score \
    +custom_reward_function.reward_kwargs.nframes=8 \
    +custom_reward_function.reward_kwargs.lambda_gfn=0.2 \
    +custom_reward_function.reward_kwargs.lambda_cf=0.0 \
    $@
```

---

### 6.3 其他自定义示例

#### 示例 1：完全禁用额外奖励，只看准确率
```bash
custom_reward_function.path=verl/utils/reward_score/think_with_video_reward.py \
custom_reward_function.name=compute_score \
+custom_reward_function.reward_kwargs.nframes=8 \
+custom_reward_function.reward_kwargs.lambda_gfn=0.0 \
+custom_reward_function.reward_kwargs.lambda_cf=0.0
```

#### 示例 2：强烈鼓励多轮帧选择
```bash
custom_reward_function.path=verl/utils/reward_score/think_with_video_reward.py \
custom_reward_function.name=compute_score \
+custom_reward_function.reward_kwargs.nframes=8 \
+custom_reward_function.reward_kwargs.lambda_gfn=0.3 \
+custom_reward_function.reward_kwargs.lambda_cf=0.1  # 提高到 0.1
```

#### 示例 3：调整最小帧间隔
```bash
custom_reward_function.path=verl/utils/reward_score/think_with_video_reward.py \
custom_reward_function.name=compute_score \
+custom_reward_function.reward_kwargs.nframes=16  # 要求更大的帧范围
+custom_reward_function.reward_kwargs.lambda_gfn=0.5 \
+custom_reward_function.reward_kwargs.lambda_cf=0.02
```

---

## 七、数据格式要求

### 7.1 Parquet 文件必需字段

使用自动路由时，数据文件（.parquet）需要包含：

```python
{
    "prompt": [...],                    # 对话历史
    "images": [...],                    # 初始帧
    "metadata": {
        "answer_key": "A",              # 正确答案
        "options": [...],
        "video_id": "xxx"
    },
    "ground_truth": "A",                # 用于 reward 计算
    "data_source": "TencentARC/Video-Holmes",  # 用于路由到正确的 reward function
    "extra_info": {
        "question": "...",              # 问题文本
        "total_frames": 300,            # 视频总帧数
        "fps": 30,                      # 帧率
        "video_path": "...",            # 视频路径
        "height": 360,
        "width": 640
    }
}
```

**关键点**：
- `data_source` 必须是 `['vstar', 'vl_agent', 'chart', 'TencentARC/Video-Holmes']` 之一
- `extra_info` 中的 `question` 和 `total_frames` 是 reward 计算所必需的

---

## 八、调试建议

### 8.1 检查 reward function 是否正确加载

查看训练日志中的打印：
```
using customized reward function 'compute_score' from 'verl/utils/reward_score/think_with_video_reward.py'
```

如果没有这行，说明使用了 `_default_compute_score` 自动路由。

---

### 8.2 检查 reward 计算结果

训练过程中会打印：
```
========================start of text========================
predict_str: <think>...</think><action>output answer: A</action>
ground_truth: A
scores: 1.02 1.0 1.0 0.02
========================end of text========================
```

- `scores: (total_score, acc_score, format_score, other_score)`
- 检查 `other_score` 是否符合预期（是否包含 `lambda_gfn` 和 `lambda_cf` 的奖励）

---

### 8.3 消融实验建议

比较不同配置的影响：

| 实验组 | lambda_gfn | lambda_cf | 目的 |
|--------|-----------|----------|------|
| Baseline | 0.5 | 0.02 | 原始配置 |
| No tool reward | 0.0 | 0.0 | 只看准确率 |
| High CF reward | 0.5 | 0.1 | 强烈鼓励多轮帧选择 |
| VH-only | 0.2 | 0.0 | Video-Holmes 专用配置 |

---

## 九、总结

### 核心要点

1. **两种配置方式**：
   - 自动路由：简单、快速，使用默认参数
   - 显式配置：灵活、可调，支持自定义参数

2. **默认参数**：
   - `nframes = 8`：最小帧间隔
   - `lambda_gfn = 0.5`：时间戳查询奖励
   - `lambda_cf = 0.02`：多轮帧选择奖励

3. **VH-only 配置差异**：
   - 降低时间戳查询奖励（0.5 → 0.2）
   - 禁用多轮帧选择奖励（0.02 → 0.0）
   - 更关注答案准确性

4. **选择建议**：
   - 快速实验 → 使用自动路由
   - 精细调优 → 使用显式配置
   - 消融实验 → 必须使用显式配置

---

## 十、参考文件

| 文件路径 | 作用 |
|---------|------|
| `verl/trainer/config/ppo_trainer.yaml` | 默认配置 |
| `verl/trainer/main_ppo.py` | 加载自定义 reward function |
| `verl/utils/reward_score/__init__.py` | 自动路由逻辑 |
| `verl/utils/reward_score/think_with_video_reward.py` | Video reward 计算实现 |
| `verl/workers/reward_manager/naive.py` | Reward manager 实现 |
| `examples/agent/train_frame_thinker.sh` | 自动路由示例 |
| `examples/agent/train_frame_thinker_vhonly.sh` | 显式配置示例 |

---

**文档版本**: v1.0
**最后更新**: 2025-11-10
**维护者**: FrameThinker-RL Team
