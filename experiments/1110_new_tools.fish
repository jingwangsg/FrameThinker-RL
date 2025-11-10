# framethinker_coldstart_full_toolv2_rolln16
EXP=framethinker_coldstart_full_toolv2_rolln16 \
ray_job_submit --skip-exists --submission-id $EXP \
  --runtime-env runtime_env.yaml \
  --no-wait \
  -- bash -c "
    EXP_NAME=$EXP \
    MODEL_PATH=/mnt/amlfs-02/shared/checkpoints/jingwang/video_reason/sft/qwen2_5vl_7b_full_framethinker_sft \
    TRAIN_FILES=/mnt/amlfs-03/shared/datasets/s3:/video_reason/Video-Holmes/train.parquet \
    VAL_FILES=/mnt/amlfs-03/shared/datasets/s3:/video_reason/Video-Holmes/test.parquet \
    bash examples/agent/train_frame_thinker_vhonly.sh \
    actor_rollout_ref.rollout.n=16 \
    data.message_template=framethinker_add_zoomin
  "

# framethinker_coldstart_full_toolv2
EXP=framethinker_coldstart_full_toolv2 \
ray_job_submit --skip-exists --submission-id $EXP \
  --runtime-env runtime_env.yaml \
  --no-wait \
  -- bash -c "
    EXP_NAME=$EXP \
    MODEL_PATH=/mnt/amlfs-02/shared/checkpoints/jingwang/video_reason/sft/qwen2_5vl_7b_full_framethinker_sft \
    TRAIN_FILES=/mnt/amlfs-03/shared/datasets/s3:/video_reason/Video-Holmes/train.parquet \
    VAL_FILES=/mnt/amlfs-03/shared/datasets/s3:/video_reason/Video-Holmes/test.parquet \
    bash examples/agent/train_frame_thinker_vhonly.sh \
    data.message_template=framethinker_add_zoomin
  "

