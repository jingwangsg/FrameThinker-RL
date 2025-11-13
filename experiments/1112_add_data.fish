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

EXP=framethinker_coldstart_full_vh+lvr10k_1112 \
ray_job_submit --skip-exists --no-wait --submission-id $EXP \
  --runtime-env runtime_env.yaml \
  -- bash -c "
    EXP_NAME=$EXP \
    MODEL_PATH=$CKPT_FULL \
    TRAIN_FILES=$VH,$LVR10K \
    VAL_FILES=$VHTEST,$LVRTEST,$VIDEOMMMU_MCQ,$MMVU,$VSIBENCH_MCQ \
    bash examples/agent/train_frame_thinker.sh \
    data.message_template=framethinker_default \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    trainer.test_freq=50
  "

EXP=framethinker_coldstart_full_vh+lvr3k_1112 \
ray_job_submit --skip-exists --no-wait --submission-id $EXP \
  --runtime-env runtime_env.yaml \
  -- bash -c "
    EXP_NAME=$EXP \
    MODEL_PATH=$CKPT_FULL \
    TRAIN_FILES=$VH,$LVR3K \
    VAL_FILES=$VHTEST,$LVRTEST,$VIDEOMMMU_MCQ,$MMVU,$VSIBENCH_MCQ \
    bash examples/agent/train_frame_thinker.sh \
    data.message_template=framethinker_default \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    trainer.test_freq=20
  "


# framethinker_coldstart_full_toolv2
EXP=framethinker_coldstart_full_toolv2_vh+lvr10k_1112 \
ray_job_submit --skip-exists --no-wait --submission-id $EXP \
  --runtime-env runtime_env.yaml \
  -- bash -c "
    EXP_NAME=$EXP \
    MODEL_PATH=$CKPT_FULL \
    TRAIN_FILES=$VH,$LVR10K \
    VAL_FILES=$VHTEST,$LVRTEST,$VIDEOMMMU_MCQ,$MMVU,$VSIBENCH_MCQ \
    bash examples/agent/train_frame_thinker.sh \
    data.message_template=framethinker_add_zoomin \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    trainer.test_freq=20
  "

# EXP=framethinker_coldstart_full_rpseq16k_vh+lvr10k_1112 \
# ray_job_submit --skip-exists --no-wait --submission-id $EXP \
#   --runtime-env runtime_env.yaml \
#   -- bash -c "
#     EXP_NAME=$EXP \
#     MODEL_PATH=$CKPT_FULL \
#     TRAIN_FILES=$VH,$LVR10K \
#     VAL_FILES=$VHTEST,$LVRTEST,$VIDEOMMMU_MCQ,$MMVU,$VSIBENCH_MCQ \
#     bash examples/agent/train_frame_thinker.sh \
#     data.message_template=framethinker_add_zoomin \
#     data.max_response_length=16384 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
#     trainer.test_freq=20
#   "

# EXP=framethinker_coldstart_full_rpseq16k_toolv2_vh+lvr10k_1112 \
# ray_job_submit --skip-exists --no-wait --submission-id $EXP \
#   --runtime-env runtime_env.yaml \
#   -- bash -c "
#     EXP_NAME=$EXP \
#     MODEL_PATH=$CKPT_FULL \
#     TRAIN_FILES=$VH,$LVR10K \
#     VAL_FILES=$VHTEST,$LVRTEST,$VIDEOMMMU_MCQ,$MMVU,$VSIBENCH_MCQ \
#     bash examples/agent/train_frame_thinker.sh \
#     data.message_template=framethinker_add_zoomin \
#     data.max_response_length=16384 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
#     trainer.test_freq=20
#   "