#!/bin/bash
# run on 8xH100
# make sure your current working directory is the root of the project
swanlab login --api-key <api_key>

set -x
ulimit -n 65535

PROJECT_DIR="$(pwd)"

BASE_DATA_DIR=$PROJECT_DIR/data/video_reason/Video-Holmes
PROJECT_NAME=video_holmes_rl
EXP_NAME=${EXP_NAME:-framethinker_vonly_cs_3e-6}
SAVE_CHECKPOINT_DIR=$PROJECT_DIR/ckpt/video_reason
MODEL_PATH=${MODEL_PATH:-$PROJECT_DIR/model_weights/ft_coldstart/qwen2_5vl_7b_full_framethinker_sft}
MEDIA_DIA=${MEDIA_DIA:-$PROJECT_DIR/data/video_reason/}
TRAIN_FILES=${TRAIN_FILES:-$PROJECT_DIR/data/video_reason/Video-Holmes/train.parquet}
VAL_FILES=${VAL_FILES:-$PROJECT_DIR/data/video_reason/Video-Holmes/test.parquet}


PYTHONUNBUFFERED=1  python3 -m verl.trainer.main_ppo \
    "data.train_files=[${TRAIN_FILES}]" \
    "data.val_files=[${VAL_FILES}]" \
    data.train_batch_size=32 \
    data.val_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.media_dir=${MEDIA_DIA} \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=False \
    data.dataloader_num_workers=8 \
    data.message_template=framethinker_default \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.tool_name_key=env_name \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=8192 \
    actor_rollout_ref.rollout.agent.max_turns=5 \
    actor_rollout_ref.rollout.agent.concurrent_workers=1 \
    actor_rollout_ref.rollout.agent.show_tqdm=True \
    actor_rollout_ref.rollout.agent.max_vllm_images=128 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.val_before_train=True \
    trainer.test_freq=20 \
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXP_NAME} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    trainer.total_epochs=20 \
    custom_reward_function.path=verl/utils/reward_score/think_with_video_reward.py \
    custom_reward_function.name=compute_score \
    +custom_reward_function.reward_kwargs.nframes=8 \
    +custom_reward_function.reward_kwargs.lambda_gfn=0.2 \
    +custom_reward_function.reward_kwargs.lambda_cf=0.0 \
    $@