set CKPT_FULL /mnt/amlfs-02/shared/checkpoints/jingwang/video_reason/sft/qwen2_5vl_7b_full_framethinker_sft

EXP=framethinker_coldstart_full_toolv2_lr1e-6_bsz32_1115 \
ray_job_submit --skip-exists --no-wait --submission-id $EXP \
  --runtime-env ../runtime_env.yaml \
  -- bash -c "
    EXP_NAME=$EXP \
    MODEL_PATH=$CKPT_FULL \
    bash examples/agent/train_frame_thinker_vhonly.sh \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    data.message_template=framethinker_add_zoomin \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32
  "

EXP=framethinker_coldstart_full_toolv2_lr3e-6_bsz32_1115 \
ray_job_submit --skip-exists --no-wait --submission-id $EXP \
  --runtime-env ../runtime_env.yaml \
  -- bash -c "
    EXP_NAME=$EXP \
    MODEL_PATH=$CKPT_FULL \
    bash examples/agent/train_frame_thinker_vhonly.sh \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    data.message_template=framethinker_add_zoomin \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32
  "

EXP=framethinker_coldstart_full_toolv2_lr5e-6_bsz32_1115 \
ray_job_submit --skip-exists --no-wait --submission-id $EXP \
  --runtime-env ../runtime_env.yaml \
  -- bash -c "
    EXP_NAME=$EXP \
    MODEL_PATH=$CKPT_FULL \
    bash examples/agent/train_frame_thinker_vhonly.sh \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    data.message_template=framethinker_add_zoomin \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32
  "

# EXP=framethinker_coldstart_full_lr3e-6_bsz32_1115 \
# ray_job_submit --skip-exists --no-wait --submission-id $EXP \
#   --runtime-env ../runtime_env.yaml \
#   -- bash -c "
#     EXP_NAME=$EXP \
#     MODEL_PATH=$CKPT_FULL \
#     bash examples/agent/train_frame_thinker_vhonly.sh \
#     actor_rollout_ref.actor.optim.lr=3e-6 \
#     actor_rollout_ref.actor.ppo_mini_batch_size=32
#   "

# EXP=framethinker_coldstart_full_lr5e-6_bsz32_1115 \
# ray_job_submit --skip-exists --no-wait --submission-id $EXP \
#   --runtime-env ../runtime_env.yaml \
#   -- bash -c "
#     EXP_NAME=$EXP \
#     MODEL_PATH=$CKPT_FULL \
#     bash examples/agent/train_frame_thinker_vhonly.sh \
#     actor_rollout_ref.actor.optim.lr=5e-6 \
#     actor_rollout_ref.actor.ppo_mini_batch_size=32
#   "

#   EXP=framethinker_coldstart_full_lr3e-6_bsz32_ep20_1115 \
# ray_job_submit --skip-exists --no-wait --submission-id $EXP \
#   --runtime-env ../runtime_env.yaml \
#   -- bash -c "
#     EXP_NAME=$EXP \
#     MODEL_PATH=$CKPT_FULL \
#     bash examples/agent/train_frame_thinker_vhonly.sh \
#     actor_rollout_ref.actor.optim.lr=3e-6 \
#     actor_rollout_ref.actor.ppo_mini_batch_size=32 \
#     trainer.total_epochs=20
#   "