# Self-Verification Turn 实施指南

## 目录
- [1. 概述](#1-概述)
- [2. 架构说明](#2-架构说明)
- [3. 实施步骤](#3-实施步骤)
  - [3.1 修改推理脚本](#31-修改推理脚本-inferpy)
  - [3.2 修改训练循环](#32-修改训练循环-parallel_envpy)
  - [3.3 扩展 Tool 定义](#33-扩展-tool-定义-think_with_videopy)
  - [3.4 更新奖励函数](#34-更新奖励函数-think_with_video_rewardpy)
  - [3.5 更新系统提示词](#35-更新系统提示词)
- [4. 测试验证](#4-测试验证)
- [5. 常见问题](#5-常见问题)

---

## 1. 概述

### 功能需求
在模型生成答案后，添加一个 self-verification turn，让模型：
1. 回顾自己的推理过程
2. 检查证据是否支持结论
3. 确认答案（不修改）

### 设计决策
- **使用场景**: 训练和推理都启用
- **验证行为**: 只能确认原答案，不能修改
- **输出格式**: 沿用 `<think>...</think><action>verify answer: X</action>` 格式
- **奖励机制**: 答案正确且执行验证时给予额外奖励

### 影响范围
需要修改以下 5 个文件：
1. `examples/agent/infer.py` - 推理脚本
2. `verl/workers/agent/parallel_env.py` - 训练循环
3. `verl/workers/agent/envs/visual_agent/think_with_video.py` - Tool 执行
4. `verl/utils/reward_score/think_with_video_reward.py` - 奖励函数
5. `scripts/convert_video_holmes_to_rl_parquet_v2.py` - 系统提示词

---

## 2. 架构说明

### 2.1 当前工作流程

```
用户问题 + 初始帧
    ↓
[Turn 1] Model: <think>...</think><action>choose frames 100 to 200</action>
    ↓
[Turn 1] System: 返回新帧
    ↓
[Turn 2] Model: <think>...</think><action>zoom in frame 150</action>
    ↓
[Turn 2] System: 返回高清帧
    ↓
[Turn 3] Model: <think>...</think><action>output answer: B</action>
    ↓
结束 → 计算奖励
```

### 2.2 添加 Verification 后的流程

```
用户问题 + 初始帧
    ↓
[Turn 1-N] 正常的工具调用和推理
    ↓
[Turn N] Model: <think>...</think><action>output answer: B</action>
    ↓
[Turn N+1] System: 插入验证提示 ← **新增**
    ↓
[Turn N+1] Model: <think>review reasoning...</think><action>verify answer: B</action> ← **新增**
    ↓
结束 → 计算奖励（包含验证 bonus）
```

### 2.3 关键概念

**Action Masking**:
- `action_mask = 1`: 模型生成的 tokens（用于 RL 训练）
- `action_mask = 0`: 系统返回的 tokens（不参与训练）
- **验证 tokens 必须设为 `action_mask = 1`**，这样模型才能学习如何验证

**Done Flag**:
- 当前：检测到 `output answer:` 就设置 `done = True`
- 修改后：检测到 `output answer:` 后添加验证 turn，然后才设置 `done = True`

---

## 3. 实施步骤

## 3.1 修改推理脚本 (infer.py)

### 文件位置
`examples/agent/infer.py`

### 需要修改的函数
`process_single_problem()` 函数中的主循环部分（第 362-378 行附近）

### 当前代码
```python
for i in range(CONFIG["MAX_ITERATIONS"]):
    print(f"\n{'='*80}\nIteration {i + 1}/{CONFIG['MAX_ITERATIONS']}\n{'='*80}")

    # Generate model response
    model_response_str = run_inference(engine, conversation_history)
    conversation_history.append({
        "role": "assistant",
        "content": model_response_str
    })

    # Parse response
    thinking, action = parse_model_response(model_response_str)

    # Check if answer is provided
    if action.startswith("output answer:"):
        final_answer = action.replace("output answer:", "").strip()
        print(f"\nFinal Answer: {final_answer}")
        break

    # Execute tool action
    observation = execute_action(action, metadata)
    conversation_history.append({
        "role": "user",
        "content": observation
    })
```

### 修改后的代码

```python
for i in range(CONFIG["MAX_ITERATIONS"]):
    print(f"\n{'='*80}\nIteration {i + 1}/{CONFIG['MAX_ITERATIONS']}\n{'='*80}")

    # Generate model response
    model_response_str = run_inference(engine, conversation_history)
    conversation_history.append({
        "role": "assistant",
        "content": model_response_str
    })

    # Parse response
    thinking, action = parse_model_response(model_response_str)

    # Check if answer is provided
    if action.startswith("output answer:"):
        final_answer = action.replace("output answer:", "").strip()
        print(f"\nFinal Answer (before verification): {final_answer}")

        # ========== 新增：Self-Verification Turn ==========
        # 构建验证提示
        verification_prompt = build_verification_prompt(
            original_answer=final_answer,
            question=metadata.get("question", "")
        )

        # 添加验证提示到对话历史
        conversation_history.append({
            "role": "user",
            "content": verification_prompt
        })

        print(f"\n{'='*80}\nVerification Turn\n{'='*80}")

        # 生成验证响应
        verification_response = run_inference(engine, conversation_history)
        conversation_history.append({
            "role": "assistant",
            "content": verification_response
        })

        # 解析验证响应
        verify_thinking, verify_action = parse_model_response(verification_response)

        # 检查验证格式和一致性
        if verify_action.startswith("verify answer:"):
            verified_answer = verify_action.replace("verify answer:", "").strip()

            if verified_answer == final_answer:
                print(f"✓ Verification passed: {verified_answer}")
                has_verification = True
            else:
                print(f"✗ Verification inconsistent: {verified_answer} != {final_answer}")
                has_verification = False
        else:
            print(f"✗ Invalid verification format")
            has_verification = False

        # 在 metadata 中记录验证信息（用于后续分析）
        metadata["has_verification"] = has_verification
        metadata["verification_response"] = verification_response
        # ========== 验证 Turn 结束 ==========

        break

    # Execute tool action
    observation = execute_action(action, metadata)
    conversation_history.append({
        "role": "user",
        "content": observation
    })
```

### 需要添加的辅助函数

在 `infer.py` 文件顶部添加以下函数：

```python
def build_verification_prompt(original_answer: str, question: str) -> str:
    """
    构建验证提示，要求模型检查推理过程

    Args:
        original_answer: 模型给出的原始答案（如 "B"）
        question: 原始问题文本

    Returns:
        验证提示字符串
    """
    prompt = f"""You have provided the answer: {original_answer}

Now, please verify your answer by carefully reviewing:
1. The video frames and evidence you examined
2. Whether your reasoning process is logically sound
3. Whether the evidence truly supports your conclusion

Output your verification in the format:
<think>
[Your review of the reasoning process]
</think>
<action>verify answer: {original_answer}</action>

Note: You should confirm the same answer. Focus on verifying the correctness of your reasoning."""

    return prompt
```

### 修改说明

1. **检测答案输出**: 保持原有的 `if action.startswith("output answer:")` 逻辑
2. **构建验证提示**: 使用 `build_verification_prompt()` 创建验证请求
3. **生成验证响应**: 调用 `run_inference()` 生成验证 turn
4. **解析验证结果**: 检查格式是否为 `verify answer: X` 且答案一致
5. **记录验证信息**: 保存到 `metadata` 中，便于后续分析

---

## 3.2 修改训练循环 (parallel_env.py)

### 文件位置
`verl/workers/agent/parallel_env.py`

### 需要修改的函数
`agent_rollout_loop()` 函数（第 126-428 行）

### 关键修改点

#### 修改 1: 在循环前添加验证状态追踪

在函数开始处（约第 160 行），添加验证状态追踪变量：

```python
def agent_rollout_loop(config, vllm_engine, vllm_inputs, prompts, multi_modal_inputs, sampling_params):
    # ... 现有代码 ...

    # 初始化状态变量
    active_mask = torch.ones(batch_size, dtype=torch.bool, device='cpu')
    running_states = [[] for _ in range(batch_size)]
    running_action_masks = [[] for _ in range(batch_size)]
    # ... 其他初始化 ...

    # ========== 新增：验证状态追踪 ==========
    needs_verification = [False] * batch_size  # 标记哪些样本需要验证
    original_answers = [None] * batch_size      # 保存原始答案
    # ========================================
```

#### 修改 2: 修改 done 检测逻辑

找到检测 `output answer:` 的代码（约第 275-278 行），修改如下：

**当前代码**:
```python
if done or step == config.agent.max_turns - 1:
    active_mask[idx] = False
    continue
```

**修改后的代码**:
```python
# ========== 修改：不立即结束，而是标记需要验证 ==========
if done and not needs_verification[idx]:
    # 检查是否是 output answer
    action_text = extract_action_from_response(actions[active_idx])

    if action_text and action_text.startswith("output answer:"):
        # 提取答案
        answer = action_text.replace("output answer:", "").strip()
        original_answers[idx] = answer

        # 标记需要验证，不设置 done
        needs_verification[idx] = True

        # 构建验证提示
        verification_prompt = build_verification_prompt_for_training(
            answer=answer,
            question=info[idx].get("question", "")
        )

        # 将验证提示作为新的 observation 添加
        obs_results_list[active_idx] = {
            "prompt": verification_prompt,
            "multi_modal_data": None
        }
        dones[active_idx] = False  # 确保不结束

        print(f"Sample {idx}: Requesting verification for answer '{answer}'")
        continue
    else:
        # 其他类型的 done（如达到 max_turns）
        active_mask[idx] = False
        continue

elif done and needs_verification[idx]:
    # 已经完成验证，现在真正结束
    active_mask[idx] = False

    # 验证答案一致性
    action_text = extract_action_from_response(actions[active_idx])
    if action_text and action_text.startswith("verify answer:"):
        verified_answer = action_text.replace("verify answer:", "").strip()
        if verified_answer == original_answers[idx]:
            info[idx]["verification_passed"] = True
        else:
            info[idx]["verification_passed"] = False
            print(f"Sample {idx}: Verification mismatch! Original: {original_answers[idx]}, Verified: {verified_answer}")
    else:
        info[idx]["verification_passed"] = False
        print(f"Sample {idx}: Invalid verification format")

    continue

elif step == config.agent.max_turns - 1:
    # 达到最大轮次
    active_mask[idx] = False
    continue
# ========================================================
```

#### 修改 3: 添加辅助函数

在 `parallel_env.py` 文件中添加以下辅助函数：

```python
def extract_action_from_response(response_text: str) -> str:
    """
    从模型响应中提取 action 部分

    Args:
        response_text: 完整的模型响应

    Returns:
        action 文本，如 "output answer: B" 或 "verify answer: B"
    """
    import re

    # 匹配 <action>...</action>
    action_match = re.search(r'<action>(.*?)</action>', response_text, re.DOTALL)
    if action_match:
        return action_match.group(1).strip()
    return ""


def build_verification_prompt_for_training(answer: str, question: str) -> str:
    """
    构建训练时的验证提示

    Args:
        answer: 模型给出的答案
        question: 原始问题

    Returns:
        验证提示字符串
    """
    prompt = f"""You have provided the answer: {answer}

Now, please verify your answer by carefully reviewing:
1. The video frames and evidence you examined
2. Whether your reasoning process is logically sound
3. Whether the evidence truly supports your conclusion

Output your verification in the format:
<think>
[Your review of the reasoning process]
</think>
<action>verify answer: {answer}</action>

Note: You should confirm the same answer. Focus on verifying the correctness of your reasoning."""

    return prompt
```

#### 修改 4: 确保验证 tokens 的 action_mask = 1

在更新 `running_action_masks` 的代码中（约第 320-340 行），确保验证响应被标记为可训练：

```python
# 当前逻辑：模型生成 action_mask=1，observation action_mask=0
# 需要确认：验证响应也应该是 action_mask=1

# 在添加 response tokens 时
response_token_ids = tokenize_response(actions[active_idx])
response_action_mask = torch.ones(len(response_token_ids), dtype=torch.long)  # 模型生成，mask=1

running_states[idx].extend(response_token_ids)
running_action_masks[idx].extend(response_action_mask.tolist())

# 在添加 observation tokens 时
if not needs_verification[idx]:
    # 普通 observation: action_mask = 0
    obs_action_mask = torch.zeros(len(obs_token_ids), dtype=torch.long)
else:
    # 验证提示：这是 system prompt，通常设为 0
    # 但验证响应（上面的 response）已经是 1 了
    obs_action_mask = torch.zeros(len(obs_token_ids), dtype=torch.long)

running_states[idx].extend(obs_token_ids)
running_action_masks[idx].extend(obs_action_mask.tolist())
```

### 修改说明

1. **两阶段 done 检测**:
   - 第一次检测到 `output answer:` → 标记 `needs_verification=True`，不结束
   - 第二次检测到 `verify answer:` → 真正结束

2. **验证提示注入**:
   - 将验证提示作为 observation 添加到对话中
   - 确保格式与其他 observations 一致

3. **一致性检查**:
   - 比较 `original_answers[idx]` 和验证答案
   - 将结果保存到 `info[idx]["verification_passed"]`

4. **Action Masking**:
   - 验证响应的 tokens 设置 `action_mask=1`（可训练）
   - 验证提示的 tokens 设置 `action_mask=0`（不训练）

---

## 3.3 扩展 Tool 定义 (think_with_video.py)

### 文件位置
`verl/workers/agent/envs/visual_agent/think_with_video.py`

### 需要修改的方法
`execute()` 方法（第 47-147 行）

### 当前代码结构
```python
def execute(self, action_str: str, extra_info: dict = None) -> Tuple[Union[str, dict], float, bool, dict]:
    """
    执行 action 并返回 observation

    Returns:
        (observation, reward, done, info)
    """

    # 解析 action
    action_block = extract_action_block(action_str)

    # 匹配各种 action 类型
    if "get frame number at time" in action_block:
        # ... 处理逻辑 ...
        return observation, 0.0, False, {}

    elif "choose frames between" in action_block:
        # ... 处理逻辑 ...
        return observation_dict, 0.0, False, {}

    elif "zoom in frame" in action_block:
        # ... 处理逻辑 ...
        return observation_dict, 0.0, False, {}

    elif action_block.startswith("output answer:"):
        answer = action_block.replace("output answer:", "").strip()
        return f"Answer provided: {answer}", 0.0, True, {"answer": answer}

    else:
        return "Invalid action format.", 0.0, False, {}
```

### 修改后的代码

在 `elif action_block.startswith("output answer:")` 之后，添加新的处理分支：

```python
def execute(self, action_str: str, extra_info: dict = None) -> Tuple[Union[str, dict], float, bool, dict]:
    """
    执行 action 并返回 observation

    Returns:
        (observation, reward, done, info)
    """

    # 解析 action
    action_block = extract_action_block(action_str)

    # 匹配各种 action 类型
    if "get frame number at time" in action_block:
        # ... 现有代码 ...
        return observation, 0.0, False, {}

    elif "choose frames between" in action_block:
        # ... 现有代码 ...
        return observation_dict, 0.0, False, {}

    elif "zoom in frame" in action_block:
        # ... 现有代码 ...
        return observation_dict, 0.0, False, {}

    elif action_block.startswith("output answer:"):
        answer = action_block.replace("output answer:", "").strip()
        return f"Answer provided: {answer}", 0.0, True, {"answer": answer}

    # ========== 新增：处理 verify answer action ==========
    elif action_block.startswith("verify answer:"):
        verified_answer = action_block.replace("verify answer:", "").strip()

        observation = f"Verification completed. Confirmed answer: {verified_answer}"

        # 返回信息
        info = {
            "verified_answer": verified_answer,
            "has_verification": True
        }

        # done=True: 验证后结束对话
        return observation, 0.0, True, info
    # ====================================================

    else:
        return "Invalid action format.", 0.0, False, {}
```

### 修改说明

1. **新增 action 类型**: `verify answer: X`
2. **设置 done=True**: 验证完成后结束对话
3. **返回信息**: 在 `info` 中标记 `has_verification=True`
4. **Observation**: 返回确认信息（虽然这通常是最后一个 turn，但保持格式一致性）

### 可选：添加验证逻辑

如果需要在 tool 层面检查答案一致性，可以扩展如下：

```python
elif action_block.startswith("verify answer:"):
    verified_answer = action_block.replace("verify answer:", "").strip()

    # 从 extra_info 中获取原始答案（如果有）
    original_answer = extra_info.get("original_answer", None)

    if original_answer and verified_answer != original_answer:
        # 答案不一致
        observation = f"Verification failed: answer mismatch (original: {original_answer}, verified: {verified_answer})"
        info = {
            "verified_answer": verified_answer,
            "has_verification": False,
            "verification_mismatch": True
        }
    else:
        # 答案一致或无法比较
        observation = f"Verification completed. Confirmed answer: {verified_answer}"
        info = {
            "verified_answer": verified_answer,
            "has_verification": True
        }

    return observation, 0.0, True, info
```

---

## 3.4 更新奖励函数 (think_with_video_reward.py)

### 文件位置
`verl/utils/reward_score/think_with_video_reward.py`

### 当前函数
当前使用的是 `v11` 函数（第 19-119 行）

### 创建新的 v12 函数

在 `v11` 函数后添加新的 `v12` 函数：

```python
def v12(predict_str: str, ground_truth: str, extra_info=None):
    """
    Version 12: 添加 self-verification 奖励

    奖励结构：
    - format_score: 1.0 if 格式正确（包含验证）
    - acc_score: 1.0 if 答案正确
    - other_score: bonus rewards
        - +0.15: 答案正确且有验证
        - +0.02: 使用了工具调用
        - +0.5: 时间问题使用了 time-to-frame

    Args:
        predict_str: 模型的完整输出
        ground_truth: 正确答案
        extra_info: 额外信息（可选）

    Returns:
        (total_score, acc_score, format_score, other_score)
    """
    import re

    format_score = 0.0
    acc_score = 0.0
    other_score = 0.0

    # ========== 1. 解析所有 think 和 action 块 ==========
    think_blocks = re.findall(r'<think>(.*?)</think>', predict_str, re.DOTALL)
    action_blocks = re.findall(r'<action>(.*?)</action>', predict_str, re.DOTALL)

    if len(think_blocks) == 0 or len(action_blocks) == 0:
        # 格式错误：缺少 think 或 action
        return 0.0, 0.0, 0.0, 0.0

    if len(think_blocks) != len(action_blocks):
        # 格式错误：think 和 action 数量不匹配
        return 0.0, 0.0, 0.0, 0.0

    # ========== 2. 检查最后一个 action 是否是 verify answer ==========
    last_action = action_blocks[-1].strip()

    # 检查倒数第二个 action 是否是 output answer
    if len(action_blocks) < 2:
        # 至少需要两个 actions：output answer + verify answer
        return 0.0, 0.0, 0.0, 0.0

    second_last_action = action_blocks[-2].strip()

    # 验证格式：倒数第二个必须是 output answer，最后一个必须是 verify answer
    if not second_last_action.startswith("output answer:"):
        return 0.0, 0.0, 0.0, 0.0

    has_verification = last_action.startswith("verify answer:")

    if not has_verification:
        # 没有验证 turn，格式不完整
        return 0.0, 0.0, 0.0, 0.0

    # ========== 3. 提取答案 ==========
    original_answer = second_last_action.replace("output answer:", "").strip().upper()
    verified_answer = last_action.replace("verify answer:", "").strip().upper()

    # ========== 4. 检查答案一致性 ==========
    if original_answer != verified_answer:
        # 验证答案与原答案不一致，格式错误
        return 0.0, 0.0, 0.0, 0.0

    # 使用 original_answer 作为模型的最终答案
    model_answer = original_answer

    # ========== 5. 检查其他格式约束 ==========
    # （保留 v11 中的其他检查逻辑）

    # 检查 frame ranges
    frame_ranges = set()
    for i in range(len(action_blocks) - 2):  # 排除最后两个（output answer 和 verify answer）
        action = action_blocks[i].strip()
        think = think_blocks[i].strip()

        if "choose frames between" in action:
            # 从 action 中提取 frame range
            match = re.search(r'choose frames between (\d+) and (\d+)', action)
            if match:
                start_frame = int(match.group(1))
                end_frame = int(match.group(2))

                # 检查是否在 think 中提到了这个范围
                think_mentions_range = (
                    f"{start_frame}" in think or
                    f"{end_frame}" in think or
                    "frames" in think.lower()
                )

                if not think_mentions_range:
                    # think 中没有提到 frame range
                    return 0.0, 0.0, 0.0, 0.0

                # 检查范围是否重复
                range_key = (start_frame, end_frame)
                if range_key in frame_ranges:
                    # 重复请求相同的 frames
                    return 0.0, 0.0, 0.0, 0.0
                frame_ranges.add(range_key)

                # 检查范围是否有效
                nframes = 8 if extra_info is None else (8 if extra_info.get("duration", 0) <= 300 else 12)
                if end_frame - start_frame < nframes:
                    # 范围太小
                    return 0.0, 0.0, 0.0, 0.0

    # 检查 time-to-frame 后是否跟着 choose frames
    used_time_to_frame = False
    for i in range(len(action_blocks) - 2):
        action = action_blocks[i].strip()

        if "get frame number at time" in action:
            used_time_to_frame = True

            # 检查下一个 action 是否是 choose frames
            if i + 1 < len(action_blocks) - 2:  # 确保不越界且不是最后两个
                next_action = action_blocks[i + 1].strip()
                if not "choose frames" in next_action:
                    # time-to-frame 后没有 choose frames
                    return 0.0, 0.0, 0.0, 0.0
            else:
                # time-to-frame 是最后一个工具调用，但没有 choose frames
                return 0.0, 0.0, 0.0, 0.0

    # ========== 6. 所有格式检查通过 ==========
    format_score = 1.0

    # ========== 7. 检查答案正确性 ==========
    if model_answer == ground_truth.upper():
        acc_score = 1.0

        # ========== 8. 计算 bonus rewards ==========
        # Bonus 1: 正确答案 + 验证 → +0.15
        other_score += 0.15

        # Bonus 2: 使用了工具调用 → +0.02
        if len(action_blocks) > 2:  # 除了 output answer 和 verify answer 之外还有其他 actions
            other_score += 0.02

        # Bonus 3: 时间问题使用了 time-to-frame → +0.5
        if used_time_to_frame and extra_info and extra_info.get("is_time_based_question", False):
            other_score += 0.5

    # ========== 9. 计算总分 ==========
    total_score = acc_score + other_score

    return total_score, acc_score, format_score, other_score
```

### 修改奖励函数调用

在使用奖励函数的地方（通常是训练脚本或配置文件），将 `v11` 改为 `v12`：

```python
# 示例：在某个配置或调用处
from verl.utils.reward_score.think_with_video_reward import v12 as compute_reward

# 使用
total_score, acc, format_score, other = compute_reward(
    predict_str=model_output,
    ground_truth=correct_answer,
    extra_info=metadata
)
```

### 奖励函数说明

**v12 与 v11 的主要区别**：

1. **要求验证 turn**:
   - 必须有 `output answer:` 和 `verify answer:` 两个 actions
   - 两个答案必须一致

2. **验证奖励**:
   - 答案正确 + 有验证 → `other_score += 0.15`

3. **格式检查更严格**:
   - `len(action_blocks) < 2` → 格式错误
   - `verified_answer != original_answer` → 格式错误

**奖励分配示例**：

| 场景 | acc_score | format_score | other_score | total_score |
|------|-----------|--------------|-------------|-------------|
| 答案正确 + 验证 + 用工具 | 1.0 | 1.0 | 0.17 | 1.17 |
| 答案正确 + 验证 + 无工具 | 1.0 | 1.0 | 0.15 | 1.15 |
| 答案错误 + 验证 | 0.0 | 1.0 | 0.0 | 0.0 |
| 答案正确 + 无验证 | 0.0 | 0.0 | 0.0 | 0.0 |
| 验证答案不一致 | 0.0 | 0.0 | 0.0 | 0.0 |

---

## 3.5 更新系统提示词

### 文件位置
`scripts/convert_video_holmes_to_rl_parquet_v2.py`

### 需要修改的内容
`SYSTEM_PROMPT` 变量（第 27-36 行）

### 当前提示词
```python
SYSTEM_PROMPT = f"""You are an expert AI assistant that answers questions about a video by iteratively analyzing it.
Your task is to output your reasoning within a <think> </think> tag, followed by a specific action within an <action> </action> tag.
Possible actions are:
1. `choose frames between START_FRAME and END_FRAME`: Request a more detailed view of a specific video segment.
2. `get frame number at time MM:SS`: Get the exact frame number for a specific time.
3. `zoom in frame FRAME_INDEX`: Zoom in on a specific frame and return the high-resolution image.
4. `output answer: OPTION`: Provide the final answer (e.g., A, B, C...) when you are confident.

Always use the <think></think> and <action></action> tags for every response."""
```

### 修改后的提示词

```python
SYSTEM_PROMPT = f"""You are an expert AI assistant that answers questions about a video by iteratively analyzing it.
Your task is to output your reasoning within a <think> </think> tag, followed by a specific action within an <action> </action> tag.
Possible actions are:
1. `choose frames between START_FRAME and END_FRAME`: Request a more detailed view of a specific video segment.
2. `get frame number at time MM:SS`: Get the exact frame number for a specific time.
3. `zoom in frame FRAME_INDEX`: Zoom in on a specific frame and return the high-resolution image.
4. `output answer: OPTION`: Provide the final answer (e.g., A, B, C...) when you are confident.
5. `verify answer: OPTION`: Verify your answer by reviewing your reasoning process and confirming the answer.

Always use the <think></think> and <action></action> tags for every response.

After providing your answer with `output answer`, you must verify it in the next turn using `verify answer` with the same answer."""
```

### 修改说明

1. **添加第 5 个 action**: `verify answer: OPTION`
2. **添加验证要求**: 在最后一句说明必须在 `output answer` 后进行验证
3. **保持格式要求**: 强调继续使用 `<think>` 和 `<action>` 标签

### 可选：更详细的验证指导

如果希望提供更详细的验证指导，可以扩展如下：

```python
SYSTEM_PROMPT = f"""You are an expert AI assistant that answers questions about a video by iteratively analyzing it.
Your task is to output your reasoning within a <think> </think> tag, followed by a specific action within an <action> </action> tag.
Possible actions are:
1. `choose frames between START_FRAME and END_FRAME`: Request a more detailed view of a specific video segment.
2. `get frame number at time MM:SS`: Get the exact frame number for a specific time.
3. `zoom in frame FRAME_INDEX`: Zoom in on a specific frame and return the high-resolution image.
4. `output answer: OPTION`: Provide the final answer (e.g., A, B, C...) when you are confident.
5. `verify answer: OPTION`: Verify your answer by reviewing your reasoning process and confirming the answer.

Always use the <think></think> and <action></action> tags for every response.

**Important**: After providing your answer with `output answer`, you MUST verify it in the next turn:
- Review the evidence you gathered from the video frames
- Check if your reasoning is logically sound
- Confirm that the evidence supports your conclusion
- Use `verify answer: OPTION` with the SAME answer to confirm

Example verification turn:
<think>
Let me review my reasoning:
1. I examined frames 100-200 and saw [evidence]
2. This supports option B because [reasoning]
3. My conclusion is logical and well-supported
</think>
<action>verify answer: B</action>"""
```

---

## 4. 测试验证

### 4.1 测试推理脚本

#### 测试命令
```bash
python examples/agent/infer.py \
    --model_path <your_model_path> \
    --test_file <test_data_path> \
    --max_iterations 6  # 增加到 6 以容纳验证 turn
```

#### 预期输出
```
==================================================
Iteration 3/6
==================================================
Model Response:
<think>
Based on the frames I've examined, option B is most likely correct because...
</think>
<action>output answer: B</action>

Final Answer (before verification): B

==================================================
Verification Turn
==================================================
Model Response:
<think>
Let me review my reasoning:
1. I examined frames 100-200 and saw the person performing action X
2. This clearly indicates option B
3. My reasoning is sound and evidence-based
</think>
<action>verify answer: B</action>

✓ Verification passed: B
```

#### 检查点
- [ ] 模型在给出答案后是否收到验证提示？
- [ ] 模型是否生成了验证响应？
- [ ] 验证响应格式是否正确（`<think>` + `<action>verify answer: X</action>`）？
- [ ] 验证答案是否与原答案一致？
- [ ] `metadata` 中是否正确记录了 `has_verification`？

### 4.2 测试训练循环

#### 小规模测试
创建一个小的测试脚本来验证 `parallel_env.py` 的修改：

```python
# test_verification.py
from verl.workers.agent.parallel_env import agent_rollout_loop
# ... 其他 imports ...

# 准备测试数据（1-2 个样本）
test_samples = [...]

# 运行 rollout
results = agent_rollout_loop(
    config=test_config,
    vllm_engine=engine,
    vllm_inputs=test_inputs,
    prompts=test_prompts,
    multi_modal_inputs=test_mm_inputs,
    sampling_params=test_params
)

# 检查结果
for i, result in enumerate(results):
    print(f"\nSample {i}:")
    print(f"  Has verification: {result['info'].get('verification_passed', False)}")
    print(f"  Final answer: {result['info'].get('verified_answer', 'N/A')}")
    print(f"  Reward: {result['reward']}")
```

#### 检查点
- [ ] 检测到 `output answer:` 后是否继续了验证 turn？
- [ ] 验证提示是否正确注入到对话中？
- [ ] 验证响应的 `action_mask` 是否为 1？
- [ ] `info` 中是否正确记录了 `verification_passed`？
- [ ] 答案不一致时是否正确标记为失败？

### 4.3 测试奖励函数

创建单元测试：

```python
# test_reward.py
from verl.utils.reward_score.think_with_video_reward import v12

# 测试用例 1: 正确答案 + 验证
test_output_1 = """
<think>Let me analyze frame 100...</think>
<action>choose frames between 100 and 200</action>
<think>Based on the evidence, option B is correct</think>
<action>output answer: B</action>
<think>Let me verify: I saw clear evidence in frames 100-200 supporting B</think>
<action>verify answer: B</action>
"""
score, acc, fmt, other = v12(test_output_1, "B", None)
print(f"Test 1: score={score}, acc={acc}, fmt={fmt}, other={other}")
assert acc == 1.0, "Should be correct"
assert fmt == 1.0, "Format should be valid"
assert other >= 0.15, "Should have verification bonus"

# 测试用例 2: 正确答案但无验证
test_output_2 = """
<think>Let me analyze...</think>
<action>choose frames between 100 and 200</action>
<think>Option B is correct</think>
<action>output answer: B</action>
"""
score, acc, fmt, other = v12(test_output_2, "B", None)
print(f"Test 2: score={score}, acc={acc}, fmt={fmt}, other={other}")
assert fmt == 0.0, "Format should be invalid (no verification)"
assert score == 0.0, "Total score should be 0"

# 测试用例 3: 验证答案不一致
test_output_3 = """
<think>Analyzing...</think>
<action>choose frames between 100 and 200</action>
<think>Option B is correct</think>
<action>output answer: B</action>
<think>Wait, maybe C?</think>
<action>verify answer: C</action>
"""
score, acc, fmt, other = v12(test_output_3, "B", None)
print(f"Test 3: score={score}, acc={acc}, fmt={fmt}, other={other}")
assert fmt == 0.0, "Format should be invalid (answer mismatch)"
assert score == 0.0, "Total score should be 0"

print("\n✓ All tests passed!")
```

运行测试：
```bash
python test_reward.py
```

### 4.4 端到端测试

#### 完整训练测试
使用少量数据进行一个 epoch 的训练：

```bash
bash examples/agent/train_frame_thinker.sh \
    --data_size 100 \
    --num_epochs 1 \
    --reward_fn v12
```

#### 监控指标
在训练日志中查看：
- `verification_rate`: 有多少比例的样本包含验证
- `verification_pass_rate`: 验证通过的比例（答案一致）
- `avg_reward`: 平均奖励是否提高
- `format_score`: 格式正确率

#### 示例日志
```
Epoch 1, Step 10:
  avg_reward: 0.85
  avg_acc: 0.72
  avg_format: 0.95
  verification_rate: 0.93
  verification_pass_rate: 0.98
```

### 4.5 调试建议

#### 如果验证 turn 没有触发：
1. 检查 `needs_verification` 标志是否正确设置
2. 打印 `action_text` 确认是否检测到 `output answer:`
3. 检查验证提示是否正确添加到 `obs_results_list`

#### 如果验证答案不一致：
1. 打印验证提示，确认格式清晰
2. 检查模型是否理解验证任务
3. 可能需要调整验证提示的措辞

#### 如果奖励为 0：
1. 检查 `format_score` - 是否格式错误？
2. 检查 `acc_score` - 答案是否正确？
3. 打印 `predict_str` 查看完整输出
4. 使用上面的单元测试验证奖励函数逻辑

---

## 5. 常见问题

### Q1: 验证 turn 会增加多少计算成本？

**A**: 每个样本增加 1 个额外的生成 turn，约增加 15-20% 的推理时间。但由于验证可以提高准确率，实际上是值得的投资。

### Q2: 如果模型在验证时修改了答案怎么办？

**A**: 根据当前设计，如果验证答案与原答案不一致，会被判定为格式错误（`format_score = 0`），从而获得 0 奖励。这会惩罚模型在验证时修改答案的行为。

### Q3: 可以让模型在验证时修改答案吗？

**A**: 可以，但需要修改奖励函数逻辑：
- 允许 `verified_answer != original_answer`
- 使用 `verified_answer` 作为最终答案计算 `acc_score`
- 可以设计特殊奖励：如果原答案错误但验证后改对了，给予 bonus

修改示例：
```python
# 在 v12 函数中
if original_answer != verified_answer:
    # 允许修改，使用验证后的答案
    model_answer = verified_answer

    # 如果验证后改对了，给 bonus
    if model_answer == ground_truth.upper() and original_answer != ground_truth.upper():
        other_score += 0.3  # 成功修正错误
else:
    model_answer = original_answer
```

### Q4: max_turns 应该设置为多少？

**A**: 建议从 5 增加到 6：
- 原来 5 turns 可能用于工具调用
- 1 turn 用于 `output answer`
- 1 turn 用于 `verify answer`
- 总共 6-7 turns 比较合适

### Q5: 验证提示的设计有什么最佳实践？

**A**:
1. **明确要求确认相同答案**：避免模型误解为要修改答案
2. **提供检查清单**：如"检查证据"、"检查逻辑"等
3. **保持格式一致**：继续使用 `<think><action>` 格式
4. **简洁明了**：避免过长的提示影响生成质量

### Q6: 如何监控验证功能的效果？

**A**: 添加以下指标：
```python
# 在训练或推理脚本中
verification_stats = {
    "total_samples": 0,
    "has_verification": 0,
    "verification_passed": 0,
    "verification_failed": 0,
    "correct_with_verification": 0,
    "correct_without_verification": 0
}

# 更新统计
for sample in results:
    verification_stats["total_samples"] += 1
    if sample["info"].get("has_verification"):
        verification_stats["has_verification"] += 1
        if sample["info"].get("verification_passed"):
            verification_stats["verification_passed"] += 1
            if sample["acc_score"] == 1.0:
                verification_stats["correct_with_verification"] += 1
        else:
            verification_stats["verification_failed"] += 1

# 计算比率
print(f"Verification rate: {verification_stats['has_verification'] / verification_stats['total_samples']:.2%}")
print(f"Verification pass rate: {verification_stats['verification_passed'] / verification_stats['has_verification']:.2%}")
```

### Q7: 验证功能会影响原有模型的性能吗？

**A**:
- **训练阶段**：由于添加了新的约束（必须验证），初期可能会降低格式正确率，但通过 RL 训练模型会逐渐学会
- **推理阶段**：如果模型没有在验证上训练过，可能会不理解验证提示。建议先在训练中启用，再在推理中使用
- **建议**：逐步引入，先在小规模数据上测试，确认没问题后再全面部署

### Q8: 如果只想在推理时使用验证，不想在训练中使用，应该怎么做？

**A**:
1. **不修改** `parallel_env.py` 和奖励函数（保持使用 `v11`）
2. **只修改** `infer.py` 添加验证 turn
3. **优点**：训练不受影响，推理时增加验证提高可靠性
4. **缺点**：模型没有学习过验证，验证质量可能不高

---

## 附录：完整修改文件清单

### 必须修改的文件
1. ✅ `examples/agent/infer.py`
2. ✅ `verl/workers/agent/parallel_env.py`
3. ✅ `verl/workers/agent/envs/visual_agent/think_with_video.py`
4. ✅ `verl/utils/reward_score/think_with_video_reward.py`
5. ✅ `scripts/convert_video_holmes_to_rl_parquet_v2.py`

### 可能需要修改的配置
- 训练配置中的 `max_turns`: 5 → 6
- 奖励函数版本：`v11` → `v12`
- `max_tokens`: 确保足够（8192 通常够用）

### 建议添加的测试文件
- `tests/test_verification_reward.py` - 奖励函数单元测试
- `tests/test_verification_inference.py` - 推理脚本测试
- `tests/test_verification_training.py` - 训练循环测试

---

## 总结

通过以上 5 个文件的修改，您可以成功地在 FrameThinker 系统中添加 self-verification 功能。关键点包括：

1. **推理脚本**: 检测答案后注入验证提示，解析验证响应
2. **训练循环**: 使用两阶段 done 检测，确保验证 tokens 可训练
3. **Tool 定义**: 添加 `verify answer` action 处理
4. **奖励函数**: 奖励正确答案+验证，惩罚答案不一致
5. **系统提示**: 明确说明验证 action 和要求

遵循本指南逐步实施，并通过测试验证每个步骤，可以确保功能正确集成到系统中。
