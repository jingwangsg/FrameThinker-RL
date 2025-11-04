# Architecture Details

## Core Components

### 1. Training Pipeline (verl/trainer/)
- **main_ppo.py**: Main entry point for PPO/GRPO training
  - Uses Hydra for configuration
  - Initializes Ray for distributed execution
  - Supports custom reward functions via dynamic module loading
  - Key functions: `main()`, `run_ppo()`, `get_custom_reward_fn()`
  - TaskRunner class executes training via Ray remote actors

- **RayPPOTrainer**: Core trainer class (in verl/trainer/ppo/)
  - Manages distributed PPO training
  - Coordinates actor, rollout, and reference model workers

### 2. Model Components (verl/models/)
- Model registry for different architectures
- Weight loader registry for checkpoint management
- Support for various model types: CausalLM, Vision2Seq, etc.

### 3. Workers (verl/workers/)
- Distributed worker classes for parallel execution
- Actor workers: handle policy updates
- Rollout workers: generate trajectories
- Reference workers: compute reference log probabilities

### 4. Checkpoint Management
- **FSDPCheckpointManager**: Handles FSDP checkpoint save/load
- **model_merger.py**: Converts distributed checkpoints to HuggingFace format
- Supports both FSDP and Megatron-LM backends

### 5. Agent System (examples/agent/)
- **infer.py**: Multi-turn frame selection and reasoning
  - Iteratively selects frames based on model output
  - Parses <think> and <action> tags from model responses
  - Supports adaptive frame sampling (8 or 12 frames)
  - Implements frame extraction with aspect ratio preservation
  - Uses Cognitive Consistency Verification (CCV)

## Training Configuration (Hydra)
Key configuration groups:
- `data`: training data settings (batch size, sequence lengths, file paths)
- `algorithm`: RL algorithm settings (advantage estimator, KL control)
- `actor_rollout_ref`: actor, rollout, and reference model configs
  - `actor`: optimizer settings, FSDP config, PPO hyperparams
  - `rollout`: vllm settings, generation params, agent config
  - `ref`: reference model settings
- `trainer`: distributed training settings (GPUs, epochs, logging)

## Agent Configuration
Frame selection agent config (in rollout config):
- `activate_agent`: Enable agent mode
- `tool_name_key`: Key for environment/tool selection
- `single_response_max_tokens`: Max tokens per turn
- `max_turns`: Maximum reasoning iterations
- `max_vllm_images`: Maximum images per batch
- `concurrent_workers`: Parallel worker count

## Distributed Training Strategy
1. **FSDP (Fully Sharded Data Parallel)**:
   - Shards model parameters across GPUs
   - Supports parameter and optimizer offloading
   - Mixed precision training (bfloat16)

2. **Ray**:
   - Manages distributed workers
   - Coordinates actor/rollout/ref model execution
   - Handles resource allocation

3. **vllm**:
   - High-performance inference engine
   - Used for rollout generation
   - Supports tensor model parallelism
   - Configurable memory utilization and batching

## Data Flow
1. Load training data from .parquet files
2. Process prompts with video frames
3. Generate rollouts using vllm
4. Compute advantages (GRPO: group relative policy optimization)
5. Update actor model via PPO
6. Log metrics to wandb/swanlab/tensorboard
7. Save checkpoints periodically

## Key Algorithms
- **PPO (Proximal Policy Optimization)**: Base RL algorithm
- **GRPO (Group Relative Policy Optimization)**: Used as advantage estimator
- **REINFORCE++**: Alternative trainer available
- **REMAX**: Another RL algorithm option

## Video Processing Pipeline
1. Load video with decord.VideoReader
2. Extract metadata (FPS, frame count)
3. Sample frames based on model's <action> output
4. Scale frames preserving aspect ratio (max 640x360 or 448x252)
5. Save frames as JPEG
6. Load frames into model via Qwen2.5-VL processor
7. Generate response with <think> and <action> tags
8. Parse response and determine next frames to sample
9. Repeat until max iterations or task completion