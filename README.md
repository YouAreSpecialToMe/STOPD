# STOPD: Stochastic Truncated On-Policy Distillation

Code for the paper **Does On-Policy Distillation Need to Complete Every Rollout?**

Zhengxin Zhang\*, Maosen Tang\*, Ning Wang\*, Changhao Li\*, Jingren Zhou, Bo Dai, Sainyam Galhotra, Claire Cardie
(\* equal contribution)

[Paper (PDF)](paper/stopd.pdf) · arXiv link coming soon

On-policy distillation (OPD) trains a student on its own rollouts with token-level feedback from a teacher, but generating complete rollouts and scoring every token is expensive. STOPD randomly truncates student rollouts and corrects the resulting sampling bias with blockwise inverse-probability weighting (IPW), which preserves Vanilla OPD's gradient sum in expectation. Across teacher–student pairs, task domains and optimizers, STOPD generates 25–48% fewer tokens with accuracy comparable to Vanilla OPD, for 1.3–2.2× faster training.

## Method

Rollouts have a maximum length of L tokens, split into blocks of K tokens (L = 16,384 and K = 128 in the paper). At each training step k:

1. **Stop at random.** Before generation, each rollout draws a token position A uniformly from {0, …, L−1}. The student generates until the end of A's block, min((⌊A/K⌋ + 1)·K, L) tokens, or until it ends on its own. The teacher scores every generated token.
2. **Reweight.** Token t is generated with probability p_t = (L − ⌊t/K⌋·K) / L, so its loss term is weighted by 1/p_t. The weighted sum is unbiased for the sum over the complete rollout, and the largest weight is L/K.
3. **Normalize with a lagged count.** The weighted token losses are summed and divided by C_k, which is fixed before step k's stops are drawn. After W Vanilla warmup steps, C_{W+1} is the number of tokens Vanilla generated at step W. From then on, C_k = N̂_{k−1}, the sum of 1/p_t over the tokens generated at step k−1, which is an unbiased estimate of that step's full token count.

The per-token loss is the sampled-token reverse KL, log p_student(y_t) − log p_teacher(y_t), on the student's own tokens. It is the same loss as Vanilla OPD's, so STOPD changes only which tokens are generated and how they are weighted.

## Repository layout

```
src/stopd/truncation.py   stop sampling, IPW weights, N̂
src/stopd/normalizer.py   the lagged normalizer C_k
src/stopd/verl_patch.py   hooks into verl's agent loop, vLLM rollout server and trainer
src/stopd/loss.py         the `stopd_k1` loss in verl's distillation-loss registry
src/sitecustomize.py      installs the hooks in every process, Ray workers included
examples/                 data preparation and a Vanilla / STOPD run in mathematics
tests/                    CPU tests
paper/                    the paper
```

## Requirements

STOPD runs on [verl](https://github.com/volcengine/verl)'s on-policy distillation trainer. Tested with:

- verl at commit `3d36367e83d7130f0b658b6be46db96d85ff9ce7`
- vLLM 0.26.0, PyTorch 2.11.0, Transformers 5.10.4, Ray 2.56.1, Python 3.12

The hooks attach to verl internals by name (`AgentLoopWorker._run_agent_loop`, `vLLMHttpServer.generate`, `RayPPOTrainer._update_actor` and the distillation-loss registry), so other verl versions may need small adjustments.

## Quick start

The example reproduces the main mathematics setting: a Qwen3-1.7B-Base student, a Qwen3-4B-Instruct-2507 teacher, batch size 256, AdamW at learning rate 1e-6, 250 steps.

```bash
python examples/prepare_math_data.py --out ~/data/stopd_math
bash examples/run_math.sh vanilla   # Vanilla OPD for 250 steps; its step 25 is the shared warmup
bash examples/run_math.sh stopd     # STOPD from Vanilla's step-25 checkpoint, steps 26-250
```

Each run uses one GPU for the student and one for the teacher. `run_math.sh` reads these environment variables: `DATA_DIR`, `OUTPUT_DIR`, `STUDENT_MODEL`, `TEACHER_MODEL`, `BATCH_SIZE`, `TOTAL_STEPS`, `WARMUP_STEPS`, `STUDENT_GPUS`, `TEACHER_GPUS`, `ROLLOUT_GPU_MEM_UTIL`, `STOPD_SEED`, `STOPD_BLOCK` and `STOPD_MAX_LEN`. Extra arguments are passed to verl as Hydra overrides.

Both runs log per-step `stopd/*` metrics (tokens generated, C_k, N̂_k and the largest weight) to verl's logger and to `stopd_metrics.jsonl` in the run directory.

## Using STOPD in your own verl run

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

## How it hooks into verl

verl has no setting for a per-rollout token budget or for per-token loss weights, so STOPD patches four places. `src/sitecustomize.py` applies each patch right after the corresponding verl module loads, in every process. verl itself is not modified.

1. **Agent loop.** Each training request is tagged with its step, sample and rollout index. Validation requests are not tagged.
2. **vLLM rollout server.** The tag determines the rollout's stop, which becomes the request's `max_tokens`, together with a per-rollout seed. The length the rollout reaches is returned with the output.
3. **Trainer.** Before the actor update, the lengths become per-sample factors N_obs / C_k, which turn verl's token-mean into a division by C_k under any micro-batch split. After the update, N̂_k is recorded as the next step's C.
4. **Loss.** `stopd_k1` multiplies verl's sampled-token reverse-KL loss by 1/p_t and by that factor.

## Scope

- Supported: vLLM rollouts, single-turn agent loops, one rollout per prompt, one update per batch, and verl's legacy trainer.
- The paper pairs Base students with Instruct teachers whose end-of-sequence tokens differ. For that setting, its experiments additionally aligned the teacher's feedback at the student's stop tokens. This fix is specific to that pairing and is not included here, so results on those model pairs may differ from the numbers in the paper.

## Tests

```bash
python -m unittest discover -s tests
```

The tests need only PyTorch and run on CPU. They check that:

- the IPW-weighted gradient sum and N̂ are exactly unbiased;
- the normalizer chain and resumes read the right step;
- the hooks behave correctly, using stand-in classes.

Running inside verl needs GPUs and is not covered by these tests.

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
