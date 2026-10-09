"""Hooks that run STOPD inside verl's on-policy distillation trainer.

verl has no setting for a per-rollout token budget or for per-token loss
weights, so STOPD patches four places. src/sitecustomize.py installs them in
every process, Ray workers included, without modifying verl:

1. AgentLoopWorker._run_agent_loop tags each training request with its step,
   sample and rollout index. Validation requests are left untouched.
2. vLLMHttpServer.generate draws the rollout's stop from that tag, sets
   max_tokens to it together with a per-rollout seed, and reports the length
   the rollout reached.
3. RayPPOTrainer._update_actor turns the lengths into per-sample tensors that
   replace verl's token-mean denominator by C_k. After the update it records
   N_hat_k, the next step's C.
4. The `stopd_k1` distillation loss (stopd.loss) multiplies verl's sampled-token
   reverse-KL loss by the IPW weights and by that per-sample factor.

Settings are environment variables, which verl forwards to its workers:

  STOPD_MODE        vanilla | stopd (unset: nothing is patched)
  STOPD_MAX_LEN     L; must equal data.max_response_length (default 16384)
  STOPD_BLOCK       K (default 128)
  STOPD_SEED        seed of the stops and of each rollout's sampling (default 0)
  STOPD_METRICS     JSONL file that receives one summary row per step
  STOPD_FIRST_STEP  W + 1, the first STOPD step after W Vanilla warmup steps (stopd only)
  STOPD_INIT_DENOM  N_W, the number of tokens generated at warmup step W (stopd only)
  STOPD_STATE       JSON file holding N_hat by step (stopd only)

Vanilla mode samples full rollouts with the same per-rollout seeds and logs the
per-step token counts, so a Vanilla run doubles as the warmup STOPD branches from.
"""

import functools
import json
import os
import random
import sys
from pathlib import Path

from stopd import normalizer
from stopd.truncation import estimated_full_tokens, ipw_weights, max_weight, request_seed, stop_position

MODES = ("vanilla", "stopd")
LOSS_MODE = "stopd_k1"     # distillation_loss.loss_mode for STOPD
VANILLA_LOSS_MODE = "k1"   # verl's own sampled-token reverse KL
TAG = "__stopd"            # sampling-params key carrying a request's indices
FIELD = "stopd"            # extra_fields key of a rollout's record (non_tensor_batch key in the trainer)
_MARK = "_stopd_patched"


def mode():
    value = os.environ.get("STOPD_MODE", "")
    if value and value not in MODES:
        raise ValueError(f"unknown STOPD_MODE {value!r}; use vanilla or stopd")
    return value


def enabled():
    return bool(mode())


def max_len():
    return int(os.environ.get("STOPD_MAX_LEN", "16384"))


def block():
    return int(os.environ.get("STOPD_BLOCK", "128"))


def seed():
    return os.environ.get("STOPD_SEED", "0")


def _seed_for(info, purpose):
    return request_seed(seed(), info["step"], info["sample"], info["rollout"], purpose)


# ---------------------------------------------------------------------------- 1. agent loop
def install_agent(cls=None):
    if not enabled():
        return
    if cls is None:
        cls = sys.modules["verl.experimental.agent_loop.agent_loop"].AgentLoopWorker
    original = cls._run_agent_loop
    if getattr(original, _MARK, False):
        return

    @functools.wraps(original)
    async def run(self, sampling_params, trajectory, **kwargs):
        if trajectory["validate"]:
            return await original(self, sampling_params, trajectory, **kwargs)
        if kwargs.get("agent_name") != "single_turn_agent":
            raise ValueError("STOPD supports single-turn rollouts only")
        params = dict(sampling_params)
        params[TAG] = dict(step=int(trajectory["step"]), sample=int(trajectory["sample_index"]),
                           rollout=int(trajectory["rollout_n"]))
        return await original(self, params, trajectory, **kwargs)

    setattr(run, _MARK, True)
    cls._run_agent_loop = run


# ---------------------------------------------------------------------------- 2. rollout server
def prepare_request(config, prompt_ids, sampling_params):
    """vLLM sampling params of one tagged training rollout, and the record kept with its output."""
    params = dict(sampling_params)
    info = params.pop(TAG)
    L = max_len()
    if int(config.response_length) != L or int(config.max_model_len) <= len(prompt_ids) + L:
        raise ValueError("STOPD_MAX_LEN must equal the rollout response length and fit the model context")
    if params.get("max_tokens", L) < L or params.get("ignore_eos", False):
        raise ValueError("training request already capped below STOPD_MAX_LEN, or ignores EOS")
    if mode() == "stopd":
        budget = stop_position(L, block(), random.Random(_seed_for(info, "block")))
    else:
        budget = L
    params["max_tokens"] = budget
    params["seed"] = _seed_for(info, "rollout")
    return params, dict(info, mode=mode(), budget=budget)


def finish_request(out, info):
    """Check that the rollout ended naturally or at its budget, and attach its record."""
    if out.stop_reason != "completed" or not out.token_ids:
        raise RuntimeError("incomplete or empty rollout")
    length = len(out.token_ids)
    if length > info["budget"]:
        raise RuntimeError("rollout exceeded its token budget")
    if info["mode"] == "stopd" and getattr(out, "routed_experts", None) is not None:
        raise ValueError("STOPD does not support routing replay")
    if out.extra_fields is None:
        out.extra_fields = {}
    out.extra_fields[FIELD] = dict(info, length=length)
    return out


def install_server(cls=None):
    if not enabled():
        return
    if cls is None:
        cls = sys.modules["verl.workers.rollout.vllm_rollout.vllm_async_server"].vLLMHttpServer
    original = cls.generate
    if getattr(original, _MARK, False):
        return

    @functools.wraps(original)
    async def generate(self, prompt_ids, sampling_params, request_id, *args, **kwargs):
        if not isinstance(sampling_params, dict) or TAG not in sampling_params:
            return await original(self, prompt_ids, sampling_params, request_id, *args, **kwargs)
        if sampling_params.get("prompt_logprobs") is not None:
            raise ValueError("a teacher scoring request carries a STOPD rollout tag")
        params, info = prepare_request(self.config, prompt_ids, sampling_params)
        out = await original(self, prompt_ids, params, request_id, *args, **kwargs)
        return finish_request(out, info)

    setattr(generate, _MARK, True)
    cls.generate = generate


# ---------------------------------------------------------------------------- 3. trainer
def check_config(trainer, batch):
    actor = trainer.config.actor_rollout_ref.actor
    rollout = trainer.config.actor_rollout_ref.rollout
    loss = trainer.distillation_config.distillation_loss
    expected = LOSS_MODE if mode() == "stopd" else VANILLA_LOSS_MODE
    if loss.loss_mode != expected:
        raise ValueError(f"STOPD_MODE={mode()} needs distillation_loss.loss_mode={expected}")
    if not loss.use_policy_gradient or loss.use_task_rewards or loss.loss_max_clamp is not None:
        raise ValueError("STOPD expects the policy-gradient k1 loss without task rewards or a loss clamp")
    if (actor.ppo_epochs != 1 or rollout.n != 1 or actor.loss_agg_mode != "token-mean"
            or len(batch.batch["response_mask"]) != actor.ppo_mini_batch_size):
        raise ValueError("STOPD expects one rollout per prompt, token-mean, and one update per batch")


def check_records(rows, lengths, step):
    if not rows or len(rows) != len(lengths):
        raise ValueError("rollout records do not match the batch (is the rollout server patched?)")
    for row, length in zip(rows, lengths):
        if row["mode"] != mode() or row["step"] != step or row["length"] != length:
            raise ValueError("a rollout record disagrees with the trainer's batch")


def scale_tensors(lengths, denominator, device):
    """Per-sample tensors for the loss.

    verl's token-mean divides the summed token losses by N_obs = sum(lengths);
    multiplying every token by N_obs / C makes that a division by C instead,
    however the batch is split into micro-batches.
    """
    import torch

    scale = sum(lengths) / denominator
    return (torch.tensor(lengths, dtype=torch.int64, device=device),
            torch.full((len(lengths),), scale, dtype=torch.float32, device=device))


def record_step(step, lengths, denominator, result):
    row = dict(step=int(step), mode=mode(), samples=len(lengths), tokens=int(sum(lengths)), max_len=max_len())
    if mode() == "stopd":
        n_hat = estimated_full_tokens(lengths, max_len(), block())
        normalizer.record(step, n_hat, os.environ["STOPD_STATE"])   # the next step's C
        row.update(block=block(), denominator=denominator, n_hat=n_hat,
                   n_hat_over_denominator=n_hat / denominator, max_weight=max_weight(lengths, max_len(), block()))
    result.meta_info["metrics"].update({f"stopd/{k}": v for k, v in row.items()
                                        if isinstance(v, (int, float)) and k != "step"})
    path = Path(os.environ["STOPD_METRICS"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(json.dumps(row) + "\n")


def install_trainer(cls=None):
    if not enabled():
        return
    if cls is None:
        cls = sys.modules["verl.trainer.ppo.ray_trainer"].RayPPOTrainer
    original = cls._update_actor
    if getattr(original, _MARK, False):
        return

    @functools.wraps(original)
    def update(self, batch):
        check_config(self, batch)
        step = self.global_steps
        rows = list(batch.non_tensor_batch.pop(FIELD, []))
        lengths = [int(n) for n in batch.batch["response_mask"].sum(-1).tolist()]
        check_records(rows, lengths, step)
        denominator = None
        if mode() == "stopd":
            denominator = normalizer.denominator(step, int(os.environ["STOPD_FIRST_STEP"]),
                                                 float(os.environ["STOPD_INIT_DENOM"]), os.environ["STOPD_STATE"])
            batch.batch["stopd_length"], batch.batch["stopd_scale"] = scale_tensors(
                lengths, denominator, batch.batch["response_mask"].device)
        result = original(self, batch)
        record_step(step, lengths, denominator, result)
        return result

    setattr(update, _MARK, True)
    cls._update_actor = update


# ---------------------------------------------------------------------------- 4. loss weights
def weight_losses(losses, mask, data):
    """Token losses times the IPW weights 1 / p_t and the per-sample factor N_obs / C."""
    import torch

    if "stopd_scale" not in data or "stopd_length" not in data:
        raise RuntimeError("the STOPD loss needs the trainer patch; run with STOPD_MODE=stopd")
    scale = data["stopd_scale"].to(losses.device)
    length = data["stopd_length"].to(losses.device)
    if scale.shape != losses.shape[:1] or length.shape != losses.shape[:1]:
        raise ValueError("STOPD batch tensors have the wrong batch shape")
    if not torch.equal(mask.sum(-1), length):
        raise ValueError("the response mask disagrees with the generated lengths")
    w = ipw_weights(losses.shape[1], max_len(), block(), device=losses.device, dtype=torch.float32)
    return losses.float() * mask * w[None, :] * scale[:, None]
