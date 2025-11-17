set CKPT_FULL /mnt/amlfs-02/shared/checkpoints/jingwang/video_reason/sft/qwen2_5vl_7b_full_framethinker_sft
set CKPT_LORA /mnt/amlfs-02/shared/checkpoints/jingwang/video_reason/sft/qwen2_5vl_7b_lorar8_framethinker
set CKPT_LORA_EP6 /mnt/amlfs-02/shared/checkpoints/jingwang/video_reason/sft/qwen2_5vl_7b_lorar8_ep6_framethinker

# framethinker_coldstart_lora
EXP=framethinker_coldstart_lora_1110 \
ray_job_submit --skip-exists --no-wait --submission-id $EXP \
  --runtime-env ../runtime_env.yaml \
  -- bash -c "
    EXP_NAME=$EXP \
    MODEL_PATH=$CKPT_LORA \
    bash examples/agent/train_frame_thinker_vhonly.sh
  "

EXP=framethinker_coldstart_ep6lora_1110 \
ray_job_submit --skip-exists --no-wait --submission-id $EXP \
  --runtime-env ../runtime_env.yaml \
  -- bash -c "
    EXP_NAME=$EXP \
    MODEL_PATH=$CKPT_LORA_EP6 \
    bash examples/agent/train_frame_thinker_vhonly.sh"

EXP=framethinker_coldstart_full_1110 \
ray_job_submit --skip-exists --no-wait --submission-id $EXP \
  --runtime-env ../runtime_env.yaml \
  -- bash -c "
    EXP_NAME=$EXP \
    MODEL_PATH=$CKPT_FULL \
    bash examples/agent/train_frame_thinker_vhonly.sh
  "

EXP=framethinker_coldstart_full_rpseq16k_1110 \
ray_job_submit --skip-exists --no-wait --submission-id $EXP \
  --runtime-env ../runtime_env.yaml \
  -- bash -c "
    EXP_NAME=$EXP \
    MODEL_PATH=$CKPT_FULL \
    bash examples/agent/train_frame_thinker_vhonly.sh \
    data.max_response_length=16384
  "

EXP=framethinker_coldstart_full_ep20_1110 \
ray_job_submit --skip-exists --no-wait --submission-id $EXP \
  --runtime-env ../runtime_env.yaml \
  -- bash -c "
    EXP_NAME=$EXP \
    MODEL_PATH=$CKPT_FULL \
    bash examples/agent/train_frame_thinker_vhonly.sh \
    trainer.total_epochs=20
  "

