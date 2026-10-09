# STOPD in NeMo-RL

`stopd.patch` adds STOPD to [NeMo-RL](https://github.com/NVIDIA-NeMo/RL)'s on-policy distillation (`nemo_rl/algorithms/distillation.py`). It applies cleanly to NeMo-RL `main` at `724344ff` and `c989568` (2026-10-09).

## What the patch changes

| File | Change |
| --- | --- |
| `nemo_rl/algorithms/stopd.py` (new) | `STOPDDistillationLossFn`: NeMo-RL's `DistillationLossFn` with STOPD's weights and denominator, plus the schedule that runs on the driver. |
| `nemo_rl/algorithms/distillation.py` | When the config has a `stopd` block: per-rollout budgets before generation, the step's denominator before the update, the next denominator and `stopd/*` metrics after it, and the STOPD state in each checkpoint. |
| `nemo_rl/experience/rollouts.py` | Passes a batch's per-rollout `max_new_tokens` to generation. |
| `nemo_rl/models/generation/vllm/vllm_worker.py` | Builds one `SamplingParams` per request when per-rollout budgets are present. |
| `examples/configs/distillation_math_stopd.yaml` (new) | The math example with STOPD enabled. |

Without a `stopd` block, NeMo-RL behaves exactly as before.

## Usage

```bash
cd RL                                   # your NeMo-RL checkout
git apply /path/to/STOPD/integrations/nemo_rl/stopd.patch
uv pip install git+https://github.com/YouAreSpecialToMe/STOPD   # the driver's environment
uv run examples/run_distillation.py --config examples/configs/distillation_math_stopd.yaml
```

The policy workers do not need the `stopd` package: the loss they receive depends only on torch and nemo_rl.

Add a `stopd` block to any distillation config:

```yaml
stopd:
  block: 128                 # K; the largest weight is L / K
  warmup_steps: 25           # initial Vanilla OPD steps
  initial_denominator: null  # C of the first STOPD step when warmup_steps is 0
  seed: 42
```

L is `policy.generation.max_new_tokens`.

## Requirements

The loss raises an error at setup if any of these does not hold:

- `policy.train_global_batch_size` equals `num_prompts_per_step * num_generations_per_prompt`, so that each rollout batch makes one optimizer step with one denominator.
- `distillation.max_rollout_turns: 1`.
- `policy.generation.vllm_cfg.async_engine: false`.
- `vllm_cfg.max_model_len >= data.max_input_seq_length + max_new_tokens`. A rollout cut short by the context limit is a truncation that the weights do not account for.

`policy.max_total_sequence_length` (and the teacher's) must hold the longest prompt plus L, as in `distillation_math_stopd.yaml`.

## Testing status

`tests/test_nemo_rl_integration.py` runs the patched loss against NeMo-RL's real `DistillationLossFn` on CPU. It checks that:

- warmup steps match the stock loss;
- STOPD steps equal Σ w·h / C_k;
- the workers receive no schedule;
- resuming restores the state;
- unsupported configurations are rejected.

```bash
NEMO_RL_SRC=/path/to/patched/RL python -m unittest tests.test_nemo_rl_integration
```

The full training loop (Ray, vLLM, GPUs) has not been run with the patch yet.
