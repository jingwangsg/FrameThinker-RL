# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Metrics related to the PPO trainer.
"""

from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, List

import numpy as np
import torch

from verl import DataProto


def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    if 'action_mask' in batch.batch:
        action_mask = batch.batch['action_mask'][:, -batch.batch['responses'].shape[-1]:]
        obs_mask = response_mask * (1 - action_mask)
        obs_length = obs_mask.sum(-1).float()
    else:
        obs_length = torch.zeros_like(response_length)
    response_length -= obs_length

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
        obs_length=obs_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    # TODO: add response length
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)
    acc_tensor = batch.batch["acc_tensor"].sum(-1)
    format_tensor = batch.batch["format_tensor"].sum(-1)
    other_tensor = batch.batch["other_tensor"].sum(-1)
    # print(f"acc_tensor: {acc_tensor}")
    # print(f"acc_tensor: {acc_tensor.shape}")
    # print(f"format_tensor: {format_tensor}")
    # print(f"format_tensor: {format_tensor.shape}")
    # print(batch.non_tensor_batch)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]
    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    action_or_attn_mask = batch.batch['action_mask'] if 'action_mask' in batch.batch else batch.batch['attention_mask']
    response_mask = action_or_attn_mask[:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]
    obs_length = response_info["obs_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # acc
        "critic/acc/mean": torch.mean(acc_tensor).detach().item(),
        #format
        "critic/format/mean": torch.mean(format_tensor).detach().item(),
        #other
        "critic/other/mean": torch.mean(other_tensor).detach().item(),
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length

        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),

        # obs length
        'obs_length/mean': torch.mean(obs_length).detach().item(),
        'obs_length/min': torch.min(obs_length).detach().item(),
        'obs_length/max': torch.max(obs_length).detach().item(),

        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # Compute positive/negative token loss statistics
    if 'pg_losses' in batch.batch.keys() and advantages is not None and response_mask is not None:
        pg_losses = batch.batch.pop('pg_losses')
        clipped_mask = batch.batch.pop('clipped_mask', None)

        # Get action_mask to exclude tool returns
        if 'action_mask' in batch.batch:
            action_mask = batch.batch['action_mask'][:, -batch.batch['responses'].shape[-1]:]
            valid_mask = response_mask.bool() & action_mask.bool()
        else:
            valid_mask = response_mask.bool()

        # Separate positive and negative advantages
        positive_mask = (advantages > 0) & valid_mask
        negative_mask = (advantages <= 0) & valid_mask

        # Exclude clipped tokens
        if clipped_mask is not None:
            positive_unclipped_mask = positive_mask & (~clipped_mask.bool())
            negative_unclipped_mask = negative_mask & (~clipped_mask.bool())
        else:
            positive_unclipped_mask = positive_mask
            negative_unclipped_mask = negative_mask

        # Count tokens
        total_tokens = valid_mask.sum().item()
        positive_tokens = positive_mask.sum().item()
        negative_tokens = negative_mask.sum().item()

        # Extract losses
        positive_losses = torch.masked_select(pg_losses, positive_unclipped_mask)
        negative_losses = torch.masked_select(pg_losses, negative_unclipped_mask)

        # Compute metrics
        metrics.update({
            "actor/positive_tokens_ratio": positive_tokens / total_tokens * 100 if total_tokens > 0 else 0.0,
            "actor/negative_tokens_ratio": negative_tokens / total_tokens * 100 if total_tokens > 0 else 0.0,
            "actor/positive_loss_mean": torch.mean(positive_losses).item() if len(positive_losses) > 0 else 0.0,
            "actor/negative_loss_mean": torch.mean(negative_losses).item() if len(negative_losses) > 0 else 0.0,
        })

    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def compute_agent_metrics(batch: DataProto):
    if 'tool_cnt' not in batch.batch.keys():
        return {}

    metrics = {}

    # Overall tool calls
    tool_cnt_tensor = batch.batch.pop('tool_cnt').detach().cpu()
    metrics.update({
        "agent/tool_call_mean": torch.mean(tool_cnt_tensor).item(),
        "agent/tool_call_max": torch.max(tool_cnt_tensor).item(),
        "agent/tool_call_min": torch.min(tool_cnt_tensor).item(),
    })

    # Count responses with/without tool calls
    responses_with_tools = (tool_cnt_tensor > 0).sum().item()
    responses_without_tools = (tool_cnt_tensor == 0).sum().item()
    total_responses = len(tool_cnt_tensor)

    metrics.update({
        "agent/responses_with_tools": responses_with_tools / total_responses if total_responses > 0 else 0.0,
        "agent/responses_without_tools": responses_without_tools / total_responses if total_responses > 0 else 0.0,
    })

    # Track correct answers with/without tool calls
    if 'acc_tensor' in batch.batch.keys():
        acc_tensor = batch.batch['acc_tensor']  # Don't pop, other metrics need it
        # Sum over sequence length dimension to get per-response accuracy
        acc_per_response = acc_tensor.sum(-1, keepdims=True).detach().cpu()

        # Boolean masks for correct answers
        correct_mask = acc_per_response > 0  # 1.0 for correct, 0.0 for incorrect

        # Combine accuracy with tool usage
        correct_with_tools = torch.logical_and(correct_mask, tool_cnt_tensor > 0).sum().item()
        correct_without_tools = torch.logical_and(correct_mask, tool_cnt_tensor == 0).sum().item()

        # Also compute incorrect counts for completeness
        incorrect_with_tools = torch.logical_and(~correct_mask, tool_cnt_tensor > 0).sum().item()
        incorrect_without_tools = torch.logical_and(~correct_mask, tool_cnt_tensor == 0).sum().item()

        metrics.update({
            "agent/correct_with_tools_ratio": correct_with_tools / total_responses if total_responses > 0 else 0.0,
            "agent/correct_without_tools_ratio": correct_without_tools / total_responses if total_responses > 0 else 0.0,
            "agent/incorrect_with_tools_ratio": incorrect_with_tools / total_responses if total_responses > 0 else 0.0,
            "agent/incorrect_without_tools_ratio": incorrect_without_tools / total_responses if total_responses > 0 else 0.0,
        })

    # Per-tool-type metrics (using hierarchical naming for grouped visualization)
    if 'choose_frames_cnt' in batch.batch.keys():
        choose_frames_tensor = batch.batch.pop('choose_frames_cnt').detach().cpu()
        metrics.update({
            "agent/tool_type/choose_frames_mean": torch.mean(choose_frames_tensor).item(),
            "agent/tool_type/choose_frames_max": torch.max(choose_frames_tensor).item(),
        })

    if 'get_time_cnt' in batch.batch.keys():
        get_time_tensor = batch.batch.pop('get_time_cnt').detach().cpu()
        metrics.update({
            "agent/tool_type/get_time_mean": torch.mean(get_time_tensor).item(),
            "agent/tool_type/get_time_max": torch.max(get_time_tensor).item(),
        })

    if 'zoom_cnt' in batch.batch.keys():
        zoom_tensor = batch.batch.pop('zoom_cnt').detach().cpu()
        metrics.update({
            "agent/tool_type/zoom_mean": torch.mean(zoom_tensor).item(),
            "agent/tool_type/zoom_max": torch.max(zoom_tensor).item(),
        })

    return metrics


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate the majority voting metric
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(
    data_sources: list[str], sample_inputs: list[str], infos_dict: dict[str, list[Any]], seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]:
    """Process validation metrics into a structured format.

    Args:
        data_sources: Array of data source identifiers for each sample
        sample_inputs: List of input prompts
        infos_dict: variable name -> list of values for each sample

    Returns:
        dict[str, dict[str, dict[str, float]]]: data source -> variable name -> metric value
    """
    # Group metrics by data source, prompt and variable
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue
                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)
                metric[f"std@{n_resps}"] = np.std(var_vals)

                ns = []
                n = 2
                while n < n_resps:
                    ns.append(n)
                    n *= 2
                ns.append(n_resps)

                for n in ns:
                    # Best/Worst-of-N
                    [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(
                        data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed
                    )
                    metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                    metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                    # Majority voting
                    if var2vals.get("pred", None) is not None:
                        vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"])]
                        [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                            data=vote_data,
                            subset_size=n,
                            reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                            seed=seed,
                        )
                        metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

    return data_src2var2metric2val
