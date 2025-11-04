# FrameThinker-RL Project Overview

## Purpose
FrameThinker is a novel framework for long-video reasoning that uses multi-turn, iterative frame spotlighting to actively interrogate video content. The project implements reinforcement learning (specifically PPO/GRPO) to train video understanding models that can intelligently select and analyze relevant frames from long videos.

Key features:
- Multi-turn iterative process for video reasoning
- Cognitive Consistency Verification (CCV) module
- Average +10.4% accuracy improvement over baselines
- Achieves state-of-the-art on LongVideo-Reason benchmark using only 20.6 frames on average

## Tech Stack
- **Language**: Python 3.10
- **Core Framework**: Built on top of `verl` (VolcEngine RL library) and `DeepEyes`
- **Deep Learning**: PyTorch with FSDP (Fully Sharded Data Parallel)
- **Model Serving**: vllm 0.9.1 for inference
- **Transformers**: Hugging Face transformers 4.52.4
- **Video Processing**: decord, ffmpeg, opencv (cv2)
- **Vision Models**: Qwen2.5-VL models
- **Distributed Training**: Ray, torch.distributed
- **Logging**: wandb, swanlab, tensorboard
- **Other**: hydra-core (config management), flash-attn, liger-kernel

## Project Structure
```
FrameThinker-RL/
├── verl/                  # Core RL training library (from volcengine/verl)
│   ├── models/           # Model registry and weight loaders
│   ├── trainer/          # PPO/GRPO trainers
│   ├── workers/          # Distributed workers
│   ├── single_controller/ # Training controllers
│   ├── third_party/      # Third-party integrations
│   └── utils/            # Utilities (checkpointing, distributed, logging, etc.)
├── examples/             # Training and inference examples
│   ├── agent/           # Frame thinker agent scripts (train & infer)
│   ├── ppo_trainer/     # PPO training examples
│   ├── grpo_trainer/    # GRPO training examples
│   ├── sft/             # Supervised fine-tuning examples
│   ├── checkpoint/      # Checkpoint conversion scripts
│   └── data_preprocess/ # Data preprocessing scripts
├── recipe/              # Training recipes (dapo, drgrpo, prime, r1)
├── tests/               # Test suites
├── scripts/             # Utility scripts (model_merger.py, install_deepeyes.sh)
├── patches/             # Code patches
└── assets/              # Images and documentation assets
```

## Acknowledgements
Built on top of:
- **verl**: https://github.com/volcengine/verl
- **DeepEyes**: https://github.com/Visual-Agent/DeepEyes

Training data from:
- LongVideoReason, Video-R1, Video-Holmes, CG-Bench