# framethinker_coldstart_full_toolv2_rolln16
EXP=framethinker_coldstart_full_toolv2_rolln16_1110 \
ray_job_submit --skip-exists --no-wait --submission-id $EXP \
  --runtime-env ../runtime_env.yaml \
  -- bash -c "
    EXP_NAME=$EXP \
    MODEL_PATH=/mnt/amlfs-02/shared/checkpoints/jingwang/video_reason/sft/qwen2_5vl_7b_full_framethinker_sft \
    bash examples/agent/train_frame_thinker_vhonly.sh \
    actor_rollout_ref.rollout.n=16 \
    data.message_template=framethinker_add_zoomin
  "

# framethinker_coldstart_full_toolv2
EXP=framethinker_coldstart_full_toolv2_1110 \
ray_job_submit --skip-exists --no-wait --submission-id $EXP \
  --runtime-env ../runtime_env.yaml \
  -- bash -c "
    EXP_NAME=$EXP \
    MODEL_PATH=/mnt/amlfs-02/shared/checkpoints/jingwang/video_reason/sft/qwen2_5vl_7b_full_framethinker_sft \
    bash examples/agent/train_frame_thinker_vhonly.sh \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    data.message_template=framethinker_add_zoomin
  "

EXP=framethinker_coldstart_full_toolv2_rpseq16k_1110 \
ray_job_submit --skip-exists --no-wait --submission-id $EXP \
  --runtime-env ../runtime_env.yaml \
  -- bash -c "
    EXP_NAME=$EXP \
    MODEL_PATH=/mnt/amlfs-02/shared/checkpoints/jingwang/video_reason/sft/qwen2_5vl_7b_full_framethinker_sft \
    bash examples/agent/train_frame_thinker_vhonly.sh \
    data.message_template=framethinker_add_zoomin \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    data.max_response_length=16384
  "


EXP=framethinker_coldstart_full_toolv2_ep20_1110 \
ray_job_submit --skip-exists --no-wait --submission-id $EXP \
  --runtime-env ../runtime_env.yaml \
  -- bash -c "
    EXP_NAME=$EXP \
    MODEL_PATH=/mnt/amlfs-02/shared/checkpoints/jingwang/video_reason/sft/qwen2_5vl_7b_full_framethinker_sft \
    bash examples/agent/train_frame_thinker_vhonly.sh \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    data.message_template=framethinker_add_zoomin \
    trainer.total_epochs=20
  "



