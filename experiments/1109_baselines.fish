# framethinker_baseline
EXP=framethinker_baseline \
ray_job_submit \
  --skip-exists \
  --submission-id $EXP\
  --runtime-env runtime_env.yaml \
  --no-wait \
  -- bash -c "
    EXP_NAME=$EXP \
    bash examples/agent/train_frame_thinker_vhonly.sh
  "

# framthinker_baseline_ep20
EXP=framethinker_baseline_ep20 \
ray_job_submit \
  --skip-exists \
  --submission-id $EXP \
  --runtime-env runtime_env.yaml \
  --no-wait \
  -- bash -c "
    EXP_NAME=$EXP \
    bash examples/agent/train_frame_thinker_vhonly.sh \
    trainer.total_epochs=20
  "