#!/usr/bin/env bash
# Mathematics in the paper's main setting: Qwen3-1.7B-Base student, Qwen3-4B-Instruct-2507 teacher,
# batch 256, max response length 16,384, 250 steps.
#
#   bash examples/run_math.sh vanilla   # Vanilla OPD from the base model; its step-W checkpoint is the warmup
#   bash examples/run_math.sh stopd     # STOPD from that checkpoint, steps W+1 .. TOTAL_STEPS
#
# Both methods share the Vanilla warmup (steps 1..W, W = 25). STOPD's first loss denominator is the
# number of tokens Vanilla generated at step W, read from the Vanilla run's metrics file.
# Extra arguments are passed to verl as Hydra overrides.
set -euo pipefail
mode=${1:?usage: run_math.sh vanilla|stopd [Hydra overrides]}
shift
cd "$(dirname "$0")/.."

DATA_DIR=${DATA_DIR:-$HOME/data/stopd_math}
OUTPUT_DIR=${OUTPUT_DIR:-$PWD/outputs}
STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-1.7B-Base}
TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3-4B-Instruct-2507}
BATCH_SIZE=${BATCH_SIZE:-256}
TOTAL_STEPS=${TOTAL_STEPS:-250}
WARMUP_STEPS=${WARMUP_STEPS:-25}
STUDENT_GPUS=${STUDENT_GPUS:-1}
TEACHER_GPUS=${TEACHER_GPUS:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.45}
MAX_PROMPT_LENGTH=1024

export STOPD_MODE=$mode
export STOPD_MAX_LEN=${STOPD_MAX_LEN:-16384} STOPD_BLOCK=${STOPD_BLOCK:-128} STOPD_SEED=${STOPD_SEED:-0}
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"   # loads src/sitecustomize.py in every process

project=stopd-math
vanilla_dir=$OUTPUT_DIR/$project/vanilla_s$STOPD_SEED
resume=()
case "$mode" in
    vanilla)
        run_dir=$vanilla_dir
        loss_mode=k1
        ;;
    stopd)
        run_dir=$OUTPUT_DIR/$project/stopd_s$STOPD_SEED
        loss_mode=stopd_k1
        warmup=$vanilla_dir/checkpoints/global_step_$WARMUP_STEPS
        [[ -d $warmup ]] || { echo "no Vanilla warmup checkpoint at $warmup; run 'run_math.sh vanilla' first" >&2; exit 1; }
        # C_{W+1} = N_W, the exact number of tokens Vanilla generated at step W.
        STOPD_INIT_DENOM=$(python3 - "$vanilla_dir/stopd_metrics.jsonl" "$WARMUP_STEPS" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
print([r for r in rows if r["mode"] == "vanilla" and r["step"] == int(sys.argv[2])][-1]["tokens"])
PY
)
        export STOPD_INIT_DENOM STOPD_FIRST_STEP=$((WARMUP_STEPS + 1)) STOPD_STATE=$run_dir/stopd_nhat.json
        resume=(trainer.resume_mode=resume_path trainer.resume_from_path="$warmup" trainer.del_local_ckpt_after_load=False)
        ;;
    *)
        echo "mode must be vanilla or stopd" >&2
        exit 2
        ;;
esac
mkdir -p "$run_dir"
export STOPD_METRICS=$run_dir/stopd_metrics.jsonl
max_model_len=$((MAX_PROMPT_LENGTH + STOPD_MAX_LEN + 1))
max_token_len=$((MAX_PROMPT_LENGTH + STOPD_MAX_LEN))

# trainer.total_epochs is only an upper bound: verl stops after total_training_steps or total_epochs,
# whichever comes first, and one epoch of this data is 56 steps at batch 256.
# Gradient clipping is verl's default, 1.0.
python3 -m verl.trainer.main_ppo \
    trainer.use_v1=False \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="['$DATA_DIR/train.parquet']" \
    data.val_files="['$DATA_DIR/math500.parquet']" \
    data.train_batch_size="$BATCH_SIZE" \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length="$STOPD_MAX_LEN" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=True \
    data.seed="$STOPD_SEED" \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$STUDENT_MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    "actor_rollout_ref.actor.optim.betas=[0.9,0.999]" \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$BATCH_SIZE" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$max_token_len \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra','hf_model']" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.seed="$STOPD_SEED" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.max_model_len=$max_model_len \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$max_token_len \
    distillation.enabled=True \
    distillation.n_gpus_per_node="$TEACHER_GPUS" \
    distillation.nnodes=1 \
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL" \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.85 \
    distillation.teacher_models.teacher_model.inference.max_model_len=$max_model_len \
    distillation.distillation_loss.loss_mode=$loss_mode \
    distillation.distillation_loss.use_policy_gradient=True \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.distillation_loss_coef=1.0 \
    trainer.balance_batch=True \
    trainer.logger='["console","file"]' \
    trainer.project_name=$project \
    trainer.experiment_name="$(basename "$run_dir")" \
    trainer.n_gpus_per_node="$STUDENT_GPUS" \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq="$WARMUP_STEPS" \
    trainer.total_epochs=100 \
    trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.default_local_dir="$run_dir/checkpoints" \
    ${resume[@]+"${resume[@]}"} "$@"
