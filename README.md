# STOPD: Stochastic Truncated On-Policy Distillation

Code for the paper **Does On-Policy Distillation Need to Complete Every Rollout?**

Zhengxin Zhang\*, Maosen Tang\*, Ning Wang\*, Changhao Li\*, Jingren Zhou, Bo Dai, Sainyam Galhotra, Claire Cardie
(\* equal contribution)

[arXiv](https://arxiv.org/abs/2610.21337)

On-policy distillation (OPD) trains a student on its own rollouts with token-level feedback from a teacher, but generating complete rollouts and scoring every token is expensive. STOPD randomly truncates student rollouts and corrects the resulting sampling bias with blockwise inverse-probability weighting (IPW), which preserves Vanilla OPD's gradient sum in expectation. Across teacher–student pairs, task domains and optimizers, STOPD generates 25–48% fewer tokens with accuracy comparable to Vanilla OPD, for 1.3–2.2× faster training.

## Method

Rollouts have a maximum length of L tokens, split into blocks of K tokens (L = 16,384 and K = 128 in the paper). At each training step k:

1. **Stop at random.** Before generation, each rollout draws a token position A uniformly from {0, …, L−1}. The student generates until the end of A's block, min((⌊A/K⌋ + 1)·K, L) tokens, or until it ends on its own. The teacher scores every generated token.
2. **Reweight.** Token t is generated with probability p_t = (L − ⌊t/K⌋·K) / L, so its loss term is weighted by 1/p_t. The weighted sum is unbiased for the sum over the complete rollout, and the largest weight is L/K.
3. **Normalize with a lagged count.** The weighted token losses are summed and divided by C_k, which is fixed before step k's stops are drawn. After W Vanilla warmup steps, C_{W+1} is the number of tokens Vanilla generated at step W. From then on, C_k = N̂_{k−1}, the sum of 1/p_t over the tokens generated at step k−1, which is an unbiased estimate of that step's full token count.

STOPD keeps the distillation loss itself unchanged, so any per-token loss works, whether a sampled-token reverse KL, a full-vocabulary KL or JSD, or a top-k divergence. STOPD changes only which tokens are generated and how their losses are weighted and summed.

## Frameworks

| Framework | How to use it | Tested |
| --- | --- | --- |
| Any training loop | `stopd.STOPD`, PyTorch only ([below](#any-training-loop)) | CPU unit tests |
| [verl](https://github.com/verl-project/verl) | Import hooks, no change to verl ([below](#verl)) | The code of the paper's experiments. Hooks tested on CPU with stand-in classes; short Vanilla and STOPD runs on two A100s. |
| [TRL](https://github.com/huggingface/trl) | `stopd.integrations.trl.STOPDTrainer` ([below](#trl)) | Real TRL 1.15.0: tests on CPU and on one GPU; training with vLLM colocate on an A100 and an RTX A6000. |
| [NeMo-RL](https://github.com/NVIDIA-NeMo/RL) | A small patch ([integrations/nemo_rl](integrations/nemo_rl/README.md)) | The loss is tested against NeMo-RL's own code; the full loop, with a resume, on two RTX A6000s. |
| OpenRLHF | Not yet supported: OpenRLHF 0.11 has no on-policy distillation. | — |

Install with `pip install git+https://github.com/YouAreSpecialToMe/STOPD`, or `pip install "stopd[trl] @ git+https://github.com/YouAreSpecialToMe/STOPD"` to also install TRL.

## Any training loop

A training loop needs three calls per step: token budgets before generation, the STOPD reduction of the per-token losses, and `end_step` after the update.

```python
from stopd import STOPD

stopd = STOPD(max_len=16384, block=128, warmup_steps=25)
for batch in loader:
    budgets = stopd.budgets(len(batch))                   # max new tokens of each rollout
    rollouts = generate(batch, max_tokens=budgets)        # e.g. one vLLM SamplingParams per prompt
    token_losses = distillation_loss(rollouts)            # (B, T), any per-token loss
    mask = rollouts.response_mask                         # 1 on generated tokens, any padding layout
    loss = stopd.loss(token_losses, mask, n_obs=mask.sum())
    loss.backward(); optimizer.step(); optimizer.zero_grad()
    stopd.end_step(mask)                                  # sets C for the next step
checkpoint["stopd"] = stopd.state_dict()                  # and stopd.load_state_dict(...) on resume
```

- `budgets(n, offset=i)` gives rollouts with global indices i … i+n−1 their stops, so ranks that generate different parts of a batch can each draw their own share. During the warmup, every budget is L.
- `loss` returns Σ w_t h_t / C_k at STOPD steps and the token mean Σ h_t / n_obs at warmup steps. Summed over micro-batches and data-parallel ranks, it gives the step's objective, so undo any averaging that your framework applies across them.
- For a framework that already divides by the batch's token count n_obs, multiply each token loss by `stopd.multiplier(mask, n_obs)` instead.
- `end_step` sums its statistic over the data-parallel group with `torch.distributed` (pass `group=` for model-parallel setups). Its return value is a dict of metrics.

## TRL

```python
from stopd.integrations.trl import STOPDConfig, STOPDTrainer

args = STOPDConfig(output_dir="out", max_completion_length=16384,       # L
                   stopd_block=128, stopd_warmup_steps=25)
trainer = STOPDTrainer(model="Qwen/Qwen3-1.7B-Base", teacher_model="Qwen/Qwen3-4B-Instruct-2507",
                       args=args, train_dataset=dataset)
trainer.train()
```

`examples/trl_math.py` runs it on the data of `examples/prepare_math_data.py`.

`STOPDTrainer` extends TRL's on-policy `DistillationTrainer`:

- **Budgets.** Each training rollout gets its own budget. With vLLM in colocate mode, each prompt gets its own SamplingParams. With transformers' `generate`, each row stops at its budget, but the batch runs until its longest row ends, so most of the saving comes from the shorter loss computation.
- **Loss.** TRL's per-token divergences are weighted before they are summed, and the global sum is divided by C_k.
- **Checkpoints.** The STOPD state is saved as `stopd_state.json` in each checkpoint and restored by `train(resume_from_checkpoint=...)`.

vLLM server mode and tools are not supported.

## NeMo-RL

Apply `integrations/nemo_rl/stopd.patch` to NeMo-RL and add a `stopd` block to the distillation config. See [integrations/nemo_rl/README.md](integrations/nemo_rl/README.md).

## verl

The verl integration is the code used for the paper's experiments. Tested with:

- verl commit `3d36367e83d7130f0b658b6be46db96d85ff9ce7`
- vLLM 0.26.0, PyTorch 2.11.0, Transformers 5.10.4, Ray 2.56.1, Python 3.12

The hooks attach to verl internals by name (`AgentLoopWorker._run_agent_loop`, `vLLMHttpServer.generate`, `RayPPOTrainer._update_actor` and the distillation-loss registry). If a name is missing, the run stops instead of silently running without STOPD. Newer verl marks the legacy `RayPPOTrainer` as deprecated, and the trainer hook relies on it.

### Quick start

The example reproduces the main mathematics setting: a Qwen3-1.7B-Base student, a Qwen3-4B-Instruct-2507 teacher, batch size 256, AdamW at learning rate 1e-6, 250 steps.

```bash
python examples/prepare_math_data.py --out ~/data/stopd_math
bash examples/run_math.sh vanilla   # Vanilla OPD for 250 steps; its step 25 is the shared warmup
bash examples/run_math.sh stopd     # STOPD from Vanilla's step-25 checkpoint, steps 26-250
```

- Each run uses one GPU for the student and one for the teacher.
- `run_math.sh` reads these environment variables: `DATA_DIR`, `OUTPUT_DIR`, `STUDENT_MODEL`, `TEACHER_MODEL`, `BATCH_SIZE`, `TOTAL_STEPS`, `WARMUP_STEPS`, `STUDENT_GPUS`, `TEACHER_GPUS`, `ROLLOUT_GPU_MEM_UTIL`, `STOPD_SEED`, `STOPD_BLOCK` and `STOPD_MAX_LEN`.
- Extra arguments are passed to verl as Hydra overrides.
- Both runs log per-step `stopd/*` metrics (tokens generated, C_k, N̂_k and the largest weight) to verl's logger and to `stopd_metrics.jsonl` in the run directory.

### Your own verl run

Put `src` on `PYTHONPATH`, so that `src/sitecustomize.py` loads in every process, and set:

| Variable | Meaning |
| --- | --- |
| `STOPD_MODE` | `stopd`, or `vanilla` for full rollouts with the same per-rollout seeds and token logging. Leave it unset to disable the hooks. |
| `STOPD_MAX_LEN` | L. Must equal `data.max_response_length`. Default 16384. |
| `STOPD_BLOCK` | K. Default 128. |
| `STOPD_SEED` | Seed of the stops and of each rollout's sampling. Default 0. |
| `STOPD_METRICS` | JSONL file that receives one summary row per step. |
| `STOPD_FIRST_STEP` | W + 1, the first STOPD step (`stopd` mode only). |
| `STOPD_INIT_DENOM` | C_{W+1}: the number of tokens generated at warmup step W (`stopd` mode only). |
| `STOPD_STATE` | JSON file that stores N̂ by step, so that a resumed run reads the exact previous step (`stopd` mode only). |

Then run verl with:

- `distillation.distillation_loss.loss_mode=stopd_k1`, or `k1` in `vanilla` mode
- `distillation.distillation_loss.use_policy_gradient=True` and `use_task_rewards=False`
- `actor_rollout_ref.rollout.n=1`, `actor_rollout_ref.actor.ppo_epochs=1` and `actor_rollout_ref.actor.loss_agg_mode=token-mean`
- `actor_rollout_ref.actor.ppo_mini_batch_size` equal to `data.train_batch_size`
- `trainer.use_v1=False`

To start STOPD without a warmup, set `STOPD_FIRST_STEP=1` and set `STOPD_INIT_DENOM` to an estimate of a full batch's token count.

### How the hooks work

verl has no setting for a per-rollout token budget or for per-token loss weights, so STOPD patches four places. `src/sitecustomize.py` applies each patch right after the corresponding verl module loads, in every process, Ray workers included.

1. **Agent loop.** Each training request is tagged with its step, sample and rollout index. Validation requests are not tagged.
2. **vLLM rollout server.** The tag determines the rollout's stop, which becomes the request's `max_tokens`, together with a per-rollout seed. The length the rollout reaches is returned with the output.
3. **Trainer.** Before the actor update, the lengths become per-sample factors N_obs / C_k, which turn verl's token-mean into a division by C_k under any micro-batch split. After the update, N̂_k is recorded as the next step's C.
4. **Loss.** `stopd_k1` multiplies verl's sampled-token reverse-KL loss by 1/p_t and by that factor.

## Scope

- Every integration assumes single-turn rollouts and one optimizer step per rollout batch. That step has a single denominator C_k.
- The paper pairs Base students with Instruct teachers whose end-of-sequence tokens differ. For that setting, its experiments also aligned the teacher's feedback at the student's stop tokens. This fix is specific to that pairing and is not included here, so results on those model pairs may differ from the numbers in the paper.

## Repository layout

```
src/stopd/core.py                  the framework-agnostic STOPD class
src/stopd/truncation.py            stop sampling, IPW weights, N̂
src/stopd/normalizer.py            the lagged normalizer C_k (verl)
src/stopd/verl_patch.py, loss.py   verl hooks and the `stopd_k1` loss
src/sitecustomize.py               installs the verl hooks in every process
src/stopd/integrations/trl.py      TRL's STOPDTrainer
integrations/nemo_rl/              the NeMo-RL patch
examples/                          verl data preparation and a Vanilla / STOPD run in mathematics
tests/                             unit tests
paper/                             the paper
```

## Tests

```bash
python -m unittest discover -s tests
```

The core tests need only PyTorch and run on CPU. They check that:

- the IPW-weighted gradient sum and N̂ are exactly unbiased;
- the denominator chain and resumes read the right step;
- the verl hooks behave correctly, using stand-in classes.

Two integration suites run only when their framework is present:

- The TRL tests run real TRL with tiny models when `trl` is installed: on the GPU when exactly one is visible, with TRL's Triton kernel, and otherwise on CPU, where a plain PyTorch divergence stands in for that kernel.
- The NeMo-RL tests run when `NEMO_RL_SRC` points to a patched NeMo-RL checkout.

The integrations were also run on GPUs:

- verl, on two A100s: a 4-step Vanilla run, then STOPD from its step-2 checkpoint to step 6. The run checked the denominator chain and the largest weight.
- TRL with vLLM colocate, on one A100 and on one RTX A6000: 5 steps. The run checked that every rollout stopped within its budget, the denominator chain and the saved state. `examples/trl_math.py` also ran for 4 steps.
- NeMo-RL, on two RTX A6000s: 4 steps, then a resume from the step-4 checkpoint to step 6, with the same checks as TRL.

## Citation

```bibtex
@misc{zhang2026stopd,
  title  = {Does On-Policy Distillation Need to Complete Every Rollout?},
  author = {Zhang, Zhengxin and Tang, Maosen and Wang, Ning and Li, Changhao and Zhou, Jingren and Dai, Bo and Galhotra, Sainyam and Cardie, Claire},
  year   = {2026},
  note   = {arXiv link coming soon}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
