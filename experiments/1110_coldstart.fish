
# framethinker_coldstart_lora
EXP=framethinker_coldstart_lora \
ray_job_submit \
  --skip-exists \
  --runtime-env runtime_env.yaml \
  --submission-id $EXP \
  --no-wait \
  -- bash -c "
    EXP_NAME=$EXP \
    MODEL_PATH=/mnt/amlfs-02/shared/checkpoints/jingwang/video_reason/sft/qwen2_5vl_7b_lorar8_framethinker \
    bash examples/agent/train_frame_thinker_vhonly.sh
  "

EXP=framethinker_coldstart_ep6lora \
ray_job_submit \
  --skip-exists \
  --runtime-env runtime_env.yaml \
  --submission-id $EXP \
  --no-wait \
  -- bash -c "
    EXP_NAME=$EXP \
    MODEL_PATH=/mnt/amlfs-02/shared/checkpoints/jingwang/video_reason/sft/qwen2_5vl_7b_lorar8_ep6_framethinker \
    bash examples/agent/train_frame_thinker_vhonly.sh
  "

EXP=framethinker_coldstart_full \
ray_job_submit \
  --skip-exists \
  --runtime-env runtime_env.yaml \
  --submission-id $EXP \
  --no-wait \
  -- bash -c "
    EXP_NAME=$EXP \
    MODEL_PATH=/mnt/amlfs-02/shared/checkpoints/jingwang/video_reason/sft/qwen2_5vl_7b_full_framethinker_sft \
    bash examples/agent/train_frame_thinker_vhonly.sh
  "

EXP=framethinker_coldstart_full_rolln16 \
ray_job_submit \
  --skip-exists \
  --runtime-env runtime_env.yaml \
  --submission-id $EXP \
  --no-wait \
  -- bash -c "
    EXP_NAME=$EXP \
    MODEL_PATH=/mnt/amlfs-02/shared/checkpoints/jingwang/video_reason/sft/qwen2_5vl_7b_full_framethinker_sft \
    bash examples/agent/train_frame_thinker_vhonly.sh \
    actor_rollout_ref.rollout.n=16
  "

