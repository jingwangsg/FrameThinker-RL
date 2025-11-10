# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FrameThinker is a reinforcement learning framework for long-video reasoning that uses multi-turn frame spotlighting. The system actively interrogates video content through iterative frame selection guided by a Cognitive Consistency Verification (CCV) module. It's built on top of the verl (volcengine RL) library and uses vision-language models (Qwen2.5-VL-7B-Instruct) for video understanding.

## Environment Setup

```bash
# Create and activate virtual environment
pip install uv
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
uv pip uninstall pynvml
uv pip install nvidia-ml-py

# Install ffmpeg
sudo apt update
sudo apt install ffmpeg -y

# Or run the setup script
bash setup.sh
```

## Common Commands

### Training
```bash
# Train FrameThinker model (requires 8xH100 GPUs)
# Must run from project root directory
bash examples/agent/train_frame_thinker.sh

# The training script uses PPO/GRPO algorithm with these key parameters:
# - data.train_batch_size=32
# - actor_rollout_ref.rollout.n=8 (number of rollout samples)
# - actor_rollout_ref.rollout.agent.max_turns=5 (max reasoning iterations)
# - trainer.total_epochs=10
```

### Inference & Evaluation
```bash
# Run inference on test dataset
python examples/agent/infer.py

# With custom model path and tensor parallelism
python examples/agent/infer.py --model /path/to/model --tp_size 4
```

### Checkpoint Management
```bash
# Merge distributed FSDP checkpoints to HuggingFace format
python merge_script.py \
    --backend fsdp \
    --hf_model_path /path/to/original/hf-model \
    --local_dir /path/to/your/checkpoints \
    --target_dir /path/to/save/merged_hf_model
```

### Data Preprocessing
```bash
# Transform video paths in dataset
python data_map.py --local_dir data/video_reason
```

### Testing
```bash
# Run tests with pytest
pytest tests/

# Run specific test files
pytest tests/ray/test_worker_group_basics.py
pytest tests/rollout/test_vllm_spmd.py
```

## Architecture

### Core Components

**verl/trainer/** - Training orchestration
- `main_ppo.py`: Main PPO/GRPO training entry point using Hydra config
- Uses Ray for distributed execution across workers
- Supports custom reward functions via dynamic module loading

**verl/workers/** - Distributed worker system
- `actor/`: Policy model training workers (FSDP-based)
- `rollout/`: Trajectory generation using vllm inference
- `agent/`: Multi-turn frame selection environment
- `reward_manager/`: Reward computation and scoring

**verl/models/** - Model registry and loaders
- Registry system for different model architectures
- Weight loader registry for checkpoint management
- Support for vision-language models (Qwen2.5-VL)

**verl/utils/reward_score/** - Custom reward functions
- `think_with_video_reward.py`: Implements Cognitive Consistency Verification
- Validates reasoning format: `<think>` and `<action>` tags
- Checks logical consistency of frame selection

**examples/agent/** - FrameThinker implementation
- `train_frame_thinker.sh`: Training script with full configuration
- `infer.py`: Inference script with iterative frame selection logic

### Training Flow

1. **Data Loading**: Load `.parquet` files with video paths, questions, and metadata
2. **Rollout Generation**: vllm generates responses with `<think>` and `<action>` tags
3. **Agent Execution**: Parse actions to request frames or timestamps, provide feedback
4. **Reward Computation**: Custom reward function validates reasoning consistency and answer correctness
5. **Policy Update**: PPO updates using GRPO advantage estimation
6. **Checkpoint Saving**: FSDP checkpoints saved periodically

### Agent System (Multi-turn Frame Selection)

The agent operates in a loop:
1. Model sees initial uniformly sampled frames
2. Model outputs `<think>reasoning</think><action>choose frames between X and Y</action>`
3. System extracts and provides requested frames
4. Repeat for up to `max_turns` iterations
5. Model outputs final answer: `<action>output answer: A</action>`

Actions supported:
- `choose frames between START_FRAME and END_FRAME`: Request specific video segment
- `get frame number at time MM:SS`: Convert timestamp to frame number
- `output answer: OPTION`: Provide final answer (A, B, C, etc.)

### Configuration System (Hydra)

Training uses Hydra with these config groups:
- `data`: Batch size, sequence lengths, file paths
- `algorithm`: RL algorithm (PPO/GRPO), KL control, advantage estimator
- `actor_rollout_ref`: Actor/rollout/reference model configs
  - `actor`: Optimizer, FSDP settings, PPO hyperparameters
  - `rollout`: vllm settings, agent config, generation params
  - `ref`: Reference model for KL divergence
- `trainer`: Distributed settings (GPUs, nodes, logging, checkpoints)

Override configs via command line:
```bash
python -m verl.trainer.main_ppo \
    data.train_batch_size=64 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    trainer.total_epochs=20
```

### Distributed Training Strategy

- **FSDP (Fully Sharded Data Parallel)**: Shards model parameters/gradients across GPUs
- **Ray**: Manages distributed workers and resource allocation
- **vllm**: High-performance inference engine for rollout generation
- **Tensor Parallelism**: Supported via `tensor_model_parallel_size` parameter

### Reward Function Structure

Custom rewards validate:
1. **Format**: Correct `<think>...</think><action>...</action>` structure
2. **Logical Consistency**: Frame selections don't overlap, requested timestamps are used
3. **Reasoning Quality**: Numbers in `<think>` match frame ranges in `<action>`
4. **Answer Correctness**: Final answer matches ground truth

Returns tuple: `(total_score, acc_score, format_score, other_score)`

## Key Files to Understand

- `verl/trainer/main_ppo.py`: Training entry point, config management
- `verl/utils/reward_score/think_with_video_reward.py`: Reward logic for CCV
- `examples/agent/infer.py`: Complete inference pipeline with frame extraction
- `verl/workers/agent/parallel_env.py`: Agent environment orchestration
- `examples/agent/train_frame_thinker.sh`: Full training configuration example

## Data Format

Training data (`.parquet` files) contains:
- `video_path`: Path to video file
- `prompt`: Question text
- `images`: Initial frames (for cold start)
- `metadata`: Contains `answer_key`, `options`, `video_id`
- `ground_truth`: Correct answer
- `env_name`: Environment type (e.g., "think_with_video")
- `reward_model`: Reward function name

## Important Notes

- Training requires 8xH100 GPUs (or adjust script for available resources)
- Always run training from project root directory
- Frame extraction uses aspect-ratio-preserving scaling (max 640x360 for short videos, 448x252 for long)
- SwanLab API key required for logging: `swanlab login --api-key <api_key>`
- Videos longer than 300 seconds use 12 frames per sample instead of 8
