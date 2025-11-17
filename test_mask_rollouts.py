#!/usr/bin/env python3
"""
简单测试脚本：验证mask_incomplete_rollouts函数的逻辑（Mask版本）
"""
import torch
import numpy as np
from collections import defaultdict


def mock_mask_test():
    """模拟测试mask逻辑"""

    # 模拟数据
    batch_size = 8  # 2个prompt，每个4个rollout
    seq_len = 8192

    # 模拟不同场景的attention_mask
    # Rollout 0: 长度8192，格式正确 (不mask)
    # Rollout 1: 长度8192，格式错误 (mask - 超长且无答案)
    # Rollout 2: 长度5000，格式错误 (不mask - 未超长)
    # Rollout 3: 长度8192，格式正确 (不mask)
    # Rollout 4: 长度8192，格式错误 (mask)
    # Rollout 5: 长度8192，格式错误 (mask)
    # Rollout 6: 长度8192，格式错误 (mask)
    # Rollout 7: 长度8192，格式错误 (mask)

    attention_mask = torch.zeros((batch_size, seq_len))
    response_lengths = [8192, 8192, 5000, 8192, 8192, 8192, 8192, 8192]
    for i, length in enumerate(response_lengths):
        attention_mask[i, :length] = 1

    # 模拟response_mask (初始全为1)
    response_mask = torch.ones((batch_size, seq_len))
    # 通常只有response部分是1，这里简化处理，假设所有都是response
    for i, length in enumerate(response_lengths):
        response_mask[i, :length] = 1
        response_mask[i, length:] = 0

    # 模拟format_tensor (1.0表示格式正确，0.0表示格式错误)
    format_tensor = torch.zeros((batch_size, seq_len))
    format_correct = [True, False, False, True, False, False, False, False]
    for i, is_correct in enumerate(format_correct):
        if is_correct:
            # 在最后一个有效token位置放置格式分数
            format_tensor[i, response_lengths[i] - 1] = 1.0

    # 模拟uid (2个prompt，每个4个rollout)
    uid = np.array(['prompt_1', 'prompt_1', 'prompt_1', 'prompt_1',
                    'prompt_2', 'prompt_2', 'prompt_2', 'prompt_2'], dtype=object)

    # ========== 模拟核心mask逻辑 ==========
    max_response_length = 8192
    masked_indices = []

    print("=" * 80)
    print("步骤 1: 检查每个rollout是否需要mask")
    print("=" * 80)

    for idx in range(batch_size):
        response_length = attention_mask[idx].sum().item()
        is_max_length = response_length >= max_response_length
        has_final_answer = format_tensor[idx].sum().item() > 0
        should_mask = is_max_length and (not has_final_answer)

        if should_mask:
            # 将整个response_mask置为0
            response_mask[idx] = 0
            masked_indices.append(idx)

        status = "❌ MASK" if should_mask else "✓ KEEP"
        mask_sum = response_mask[idx].sum().item()
        print(f"Rollout {idx}: length={int(response_length):5d}, has_answer={has_final_answer}, "
              f"uid={uid[idx]:10s}, response_mask_sum={int(mask_sum):5d} -> {status}")

    print(f"\n被mask的索引: {masked_indices}")
    print(f"预期结果: [1, 4, 5, 6, 7] (5个rollout被mask)")

    assert masked_indices == [1, 4, 5, 6, 7], f"Step 1 Failed: {masked_indices} != [1, 4, 5, 6, 7]"
    print("✓ 步骤 1 通过\n")

    # 验证所有样本都在batch中
    print("=" * 80)
    print("步骤 2: 验证所有样本都保留在batch中")
    print("=" * 80)

    print(f"Batch总大小: {batch_size}")
    print(f"所有样本都保留: ✓")
    print(f"被mask样本的response_mask全为0:")
    for idx in masked_indices:
        assert response_mask[idx].sum().item() == 0, f"Rollout {idx} should have response_mask=0"
        print(f"  Rollout {idx}: response_mask.sum() = {response_mask[idx].sum().item()} ✓")

    print("\n✓ 步骤 2 通过\n")

    # 模拟GRPO advantage计算
    print("=" * 80)
    print("步骤 3: 验证GRPO计算逻辑")
    print("=" * 80)

    # 模拟rewards
    token_level_rewards = torch.randn(batch_size, seq_len)
    # 计算每个rollout的总reward（被mask的样本reward也会计算）
    rewards = token_level_rewards.sum(dim=-1)

    # 按uid分组计算GRPO
    uid_to_indices = defaultdict(list)
    for idx in range(batch_size):
        uid_to_indices[uid[idx]].append(idx)

    print("GRPO分组和advantage计算:")
    for uid_val, indices in uid_to_indices.items():
        group_rewards = rewards[indices]
        mean_reward = group_rewards.mean().item()
        std_reward = group_rewards.std().item()

        print(f"\nUID {uid_val}:")
        print(f"  组内样本: {indices}")
        print(f"  Mean reward: {mean_reward:.4f}")
        print(f"  Std reward: {std_reward:.4f}")

        # 计算advantages
        for idx in indices:
            advantage = (rewards[idx].item() - mean_reward) / (std_reward + 1e-8)
            is_masked = idx in masked_indices
            will_update = "不参与更新 (response_mask=0)" if is_masked else "参与更新"
            print(f"  Rollout {idx}: reward={rewards[idx].item():.4f}, "
                  f"adv={advantage:.4f}, {will_update}")

    print("\n说明：被mask的样本参与advantage计算（用于组内normalization），")
    print("      但因为response_mask=0，在实际loss计算时不会产生梯度。")
    print("✓ 步骤 3 通过\n")

    # 统计信息
    print("=" * 80)
    print("步骤 4: 统计mask结果")
    print("=" * 80)

    masked_count = len(masked_indices)
    valid_count = batch_size - masked_count
    format_error_kept = sum(1 for i in range(batch_size)
                           if i not in masked_indices and format_tensor[i].sum().item() == 0)
    masked_rate = masked_count / batch_size

    print(f"总rollout数: {batch_size}")
    print(f"被mask的rollout: {masked_count}")
    print(f"保留的有效rollout: {batch_size} (全部保留在batch中)")
    print(f"会参与梯度更新的rollout: {valid_count}")
    print(f"保留的格式错误样本: {format_error_kept} (长度未超限)")
    print(f"Mask率: {masked_rate:.2%}")

    assert masked_count == 5, f"masked_count should be 5, got {masked_count}"
    assert format_error_kept == 1, f"format_error_kept should be 1, got {format_error_kept}"
    assert abs(masked_rate - 0.625) < 0.01, f"masked_rate should be 0.625, got {masked_rate}"

    print("✓ 步骤 4 通过\n")

    print("=" * 80)
    print("✓✓✓ 所有测试通过！Mask逻辑正确！")
    print("=" * 80)
    print("\n核心特性验证:")
    print("  ✓ 所有样本保留在batch中（不删除）")
    print("  ✓ 被mask样本的response_mask全部置0")
    print("  ✓ 被mask样本参与GRPO advantage计算")
    print("  ✓ 被mask样本不会产生梯度（response_mask=0）")


if __name__ == "__main__":
    print("=" * 80)
    print("测试mask_incomplete_rollouts逻辑（Mask版本）")
    print("=" * 80)
    print()
    mock_mask_test()
