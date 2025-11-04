# Suggested Commands for FrameThinker-RL Development

## Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install DeepEyes (additional video understanding dependencies)
bash scripts/install_deepeyes.sh
```

## Training
```bash
# Train FrameThinker agent
bash examples/agent/train_frame_thinker.sh

# Note: Edit the script to set:
# - BASE_DATA_DIR: directory containing .parquet training files
# - PROJECT_NAME: wandb/swanlab project name
# - EXPERIMENT_NAME: experiment identifier
# - SAVE_CHECKPOINT_DIR: where to save checkpoints
# - REF_MODEL_PATH: path to base model (e.g., Qwen2.5-VL)
```

## Inference & Evaluation
```bash
# Run inference with trained model
python examples/agent/infer.py

# Note: Edit CONFIG dict in infer.py to set:
# - MODEL_PATH: path to trained/merged model
# - BASE_VIDEO_DIR: directory containing video files
# - TARGET_JSON_PATH: output path for results
```

## Checkpoint Management
```bash
# Merge FSDP checkpoints into HuggingFace format
python scripts/model_merger.py \
    --backend fsdp \
    --hf_model_path /path/to/original/hf-model \
    --local_dir /path/to/your/checkpoints \
    --target_dir /path/to/save/merged_hf_model

# For Megatron checkpoints
python scripts/model_merger.py \
    --backend megatron \
    --hf_model_path /path/to/original/hf-model \
    --local_dir /path/to/your/checkpoints \
    --target_dir /path/to/save/merged_hf_model
```

## Testing
```bash
# Run specific test file
pytest tests/checkpoint/test_fsdp_ckpt.py

# Run all tests in a directory
pytest tests/checkpoint/

# Run all tests
pytest tests/
```

## Utility Commands (Linux System)
```bash
# File search (use fd instead of find - faster)
fd PATTERN              # find files matching pattern
fd -t f "*.py"          # find Python files
fd -t d                 # find directories only

# Content search (use ag instead of grep - faster)
ag PATTERN              # search for pattern in files
ag -l PATTERN           # list files containing pattern
ag "class.*Model"       # regex search

# Standard utilities
ls -la                  # list files with details
cd /path                # change directory
pwd                     # print working directory
cat file.txt            # display file contents
git status              # check git status
git log --oneline       # view commit history
```

## Configuration
- Training configs are managed by Hydra in `verl/trainer/config/`
- Override config values via command line:
  ```bash
  python -m verl.trainer.main_ppo \
      data.train_batch_size=32 \
      actor_rollout_ref.model.path=/path/to/model \
      trainer.total_epochs=10
  ```

## Common Development Tasks
```bash
# Start distributed training with Ray
# (Ray is initialized automatically in main_ppo.py)

# Monitor training
# - Check wandb/swanlab dashboard
# - Check tensorboard logs in: ${SAVE_CHECKPOINT_DIR}/logs/tensorboard

# Debug mode
# Set +debug=True in training command to enable debug mode
```