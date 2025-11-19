set CKPT_FULL /mnt/amlfs-02/shared/checkpoints/jingwang/video_reason/sft/qwen2_5vl_7b_full_framethinker_sft
set DATA_DIR /mnt/amlfs-02/shared/datasets/s3/video_reason/
set LVR10K $DATA_DIR/LongVideo-Reason/train_10k.parquet
set LVR3K $DATA_DIR/LongVideo-Reason/train_3k.parquet
set VH $DATA_DIR/Video-Holmes/train.parquet

# Benchmarks
set VHTEST $DATA_DIR/Video-Holmes/test.parquet
set LVRTEST $DATA_DIR/LongVideo-Reason/test.parquet
set VIDEOMMMU_MCQ $DATA_DIR/BENCHMARKS/videommmu_mcq.parquet
set MMVU $DATA_DIR/BENCHMARKS/mmvu.parquet
set VSIBENCH_MCQ $DATA_DIR/BENCHMARKS/vsibench_mcq.parquet


EXP=grpo_nf8_baseline_vh_lr5e-7_1117 \
ray_job_submit \
  --skip-exists \
  --submission-id $EXP\
  --runtime-env ../runtime_env.yaml \
  --no-wait \
  -- bash -c "
    EXP_NAME=$EXP \
    bash examples/agent/train_frame_thinker_vhonly.sh \
    data.message_template=default \
    actor_rollout_ref.rollout.agent.activate_agent=False \
    data.media_reading_kwargs.num_frames=8 \
    actor_rollout_ref.actor.optim.lr=5e-7
  "

EXP=grpo_nf8_baseline_vh_lr1e-6_1117 \
ray_job_submit \
  --skip-exists \
  --submission-id $EXP\
  --runtime-env ../runtime_env.yaml \
  --no-wait \
  -- bash -c "
    EXP_NAME=$EXP \
    bash examples/agent/train_frame_thinker_vhonly.sh \
    data.message_template=default \
    actor_rollout_ref.rollout.agent.activate_agent=False \
    data.media_reading_kwargs.num_frames=8 \
    actor_rollout_ref.actor.optim.lr=5e-7
  "

EXP=grpo_nf16_baseline_vh_lr1e-6_1117 \
ray_job_submit \
  --skip-exists \
  --submission-id $EXP\
  --runtime-env ../runtime_env.yaml \
  --no-wait \
  -- bash -c "
    EXP_NAME=$EXP \
    bash examples/agent/train_frame_thinker_vhonly.sh \
    data.message_template=default \
    actor_rollout_ref.rollout.agent.activate_agent=False \
    data.media_reading_kwargs.num_frames=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4
  "

# EXP=grpo_nf32_baseline_vh_1117 \
# ray_job_submit \
#   --skip-exists \
#   --submission-id $EXP\
#   --runtime-env ../runtime_env.yaml \
#   --no-wait \
#   -- bash -c "
#     EXP_NAME=$EXP \
#     bash examples/agent/train_frame_thinker_vhonly.sh \
#     data.message_template=default \
#     actor_rollout_ref.rollout.agent.activate_agent=False \
#     data.media_reading_kwargs.num_frames=32 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=8
#   "

EXP=grpo_nf8_baseline_vh+lvr10k_lr1e-6_1117 \
ray_job_submit \
  --skip-exists \
  --submission-id $EXP\
  --runtime-env ../runtime_env.yaml \
  --no-wait \
  -- bash -c "
    EXP_NAME=$EXP \
    bash examples/agent/train_frame_thinker.sh \
    data.message_template=default \
    actor_rollout_ref.rollout.agent.activate_agent=False \
    data.media_reading_kwargs.num_frames=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.optim.lr=1e-6
  "

EXP=grpo_nf8_baseline_vh+lvr10k_lr3e-6_1117 \
ray_job_submit \
  --skip-exists \
  --submission-id $EXP\
  --runtime-env ../runtime_env.yaml \
  --no-wait \
  -- bash -c "
    EXP_NAME=$EXP \
    bash examples/agent/train_frame_thinker.sh \
    data.message_template=default \
    actor_rollout_ref.rollout.agent.activate_agent=False \
    data.media_reading_kwargs.num_frames=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.optim.lr=3e-6
  "