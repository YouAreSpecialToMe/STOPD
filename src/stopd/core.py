"""Framework-agnostic STOPD for any on-policy distillation loop.

A training loop needs three calls per step: token budgets before generation,
the STOPD reduction of the per-token losses, and `end_step` after the update::

    from stopd import STOPD

    stopd = STOPD(max_len=16384, block=128, warmup_steps=25)
    for batch in loader:
        budgets = stopd.budgets(len(batch))          # max new tokens of each rollout
        rollouts = generate(batch, max_tokens=budgets)
        token_losses = distillation_loss(rollouts)   # (B, T), any per-token loss
        loss = stopd.loss(token_losses, rollouts.response_mask, n_obs=rollouts.response_mask.sum())
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        stopd.end_step(rollouts.response_mask)

The first `warmup_steps` steps are Vanilla OPD: full rollouts and the usual
token mean. After them, every rollout is cut at a random block end, each kept
token t is weighted by 1 / p_t, and the weighted sum is divided by C_k, which
is fixed before step k: C_{W+1} is the exact token count of warmup step W and
C_k = N_hat_{k-1} afterwards. Save `stopd.state_dict()` with each checkpoint.

The per-token loss can be anything whose sum over the response is the
objective: the sampled-token reverse KL of verl, a full-vocabulary KL or JSD,
a top-k divergence. STOPD changes which tokens exist and how they are summed.
"""

import random

import torch

from stopd.truncation import request_seed, stop_position

STATE_VERSION = 1


def response_positions(mask):
    """Zero-based position of each token within its response, for a (B, T) mask of generated tokens.

    Works for right- or left-padded response tensors and for prompt+response
    tensors whose mask covers only the generated tokens, provided each row's
    generated tokens are contiguous.
    """
    mask = mask.bool()
    return (mask.long().cumsum(-1) - 1).clamp_min(0)


def token_weights(mask, max_len, block):
    """IPW weights 1 / p_t at the generated tokens of `mask`, 0 elsewhere (float32)."""
    mask = mask.bool()
    pos = response_positions(mask)
    if (mask.sum(-1) > max_len).any():
        raise ValueError("a response is longer than max_len")
    weights = max_len / (max_len - (pos // block) * block).to(torch.float32)
    return weights * mask


def all_reduce_sum(value, group=None):
    """Sum a Python number over the ranks of `group` when torch.distributed is running, else return it."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return float(value)
    device = (torch.device("cuda", torch.cuda.current_device())
              if dist.get_backend(group) == "nccl" else torch.device("cpu"))
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, group=group)
    return tensor.item()


class STOPD:
    """STOPD state and math for one training run.

    Args:
        max_len: L, the maximum response length (the full rollouts' cap).
        block: K, the block width; the largest weight is L / K.
        warmup_steps: W, the number of initial Vanilla OPD steps. Their last
            step's token count gives the first STOPD denominator exactly.
        initial_denominator: C for the first STOPD step when W = 0, e.g. an
            estimate of a full batch's token count.
        seed: seed of the stops; each rollout's stop depends only on
            (seed, step, rollout index).
    """

    def __init__(self, max_len, block=128, warmup_steps=0, initial_denominator=None, seed=0):
        if not 1 <= block <= max_len:
            raise ValueError("require 1 <= block <= max_len")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if warmup_steps == 0 and initial_denominator is None:
            raise ValueError("without a warmup, give initial_denominator (e.g. a full batch's expected token count)")
        if initial_denominator is not None and not initial_denominator > 0:
            raise ValueError("initial_denominator must be positive")
        self.max_len, self.block, self.warmup_steps, self.seed = int(max_len), int(block), int(warmup_steps), seed
        self.step = 0                               # completed training steps
        self.denominator = None if warmup_steps else float(initial_denominator)   # C of the next step

    # ------------------------------------------------------------------ the step about to run
    @property
    def current_step(self):
        """1-based index of the step about to run."""
        return self.step + 1

    @property
    def truncating(self):
        """Whether the step about to run is a STOPD step (False during the Vanilla warmup)."""
        return self.step >= self.warmup_steps

    def budgets(self, n, offset=0):
        """Max new tokens for rollouts offset .. offset + n - 1 of the current step.

        Pass global rollout indices (`offset`) when ranks generate different
        parts of the batch, so that no two rollouts share a stop by construction.
        """
        if not self.truncating:
            return [self.max_len] * n
        return [stop_position(self.max_len, self.block,
                              random.Random(request_seed(self.seed, self.current_step, offset + i, 0, "block")))
                for i in range(n)]

    # ------------------------------------------------------------------ the loss
    def weights(self, mask):
        """Per-token weights of the current step: 1 / p_t (STOPD steps) or the plain mask (warmup)."""
        if not self.truncating:
            return mask.bool().to(torch.float32)
        return token_weights(mask, self.max_len, self.block)

    def loss(self, token_losses, mask, n_obs=None):
        """Reduced loss of the current step.

        STOPD steps: sum_t w_t h_t / C_k over the tokens in `mask`.
        Warmup steps: sum_t h_t / n_obs, the usual token mean, where n_obs is the
        number of generated tokens in the whole batch (all micro-batches and ranks).
        Summing this value over micro-batches and ranks gives the step's objective,
        so undo any averaging the framework applies across them.
        """
        weights = self.weights(mask).to(token_losses.device)
        if self.truncating:
            return (token_losses.float() * weights).sum() / self.denominator
        if n_obs is None:
            raise ValueError("a warmup step needs n_obs, the batch's generated-token count")
        return (token_losses.float() * weights).sum() / float(n_obs)

    def multiplier(self, mask, n_obs):
        """Per-token factor for frameworks that divide the summed token losses by n_obs.

        Multiplying each token loss by w_t * n_obs / C_k turns that token mean into
        STOPD's sum_t w_t h_t / C_k; n_obs must be the count the framework divides by.
        Warmup steps return the plain mask (the framework's own loss is Vanilla OPD).
        """
        if not self.truncating:
            return mask.bool().to(torch.float32)
        return token_weights(mask, self.max_len, self.block) * (float(n_obs) / self.denominator)

    # ------------------------------------------------------------------ after the update
    def step_statistic(self, mask_or_lengths):
        """This rank's contribution to the next denominator: the exact token count at the last
        warmup step, N_hat at STOPD steps (both add up over ranks)."""
        lengths = self._lengths(mask_or_lengths)
        if not self.truncating:
            return float(sum(lengths))
        from stopd.truncation import estimated_full_tokens

        return estimated_full_tokens(lengths, self.max_len, self.block)

    def end_step(self, mask_or_lengths, *, reduce=all_reduce_sum, group=None):
        """Finish the current step: set the next step's denominator and advance.

        `mask_or_lengths` covers this rank's rollouts of the step. The statistic is
        summed over the data-parallel `group` (pass `reduce=None` if you already
        hold the whole batch on every rank). Returns a dict of metrics.
        """
        local = self.step_statistic(mask_or_lengths)
        value = reduce(local, group) if reduce is not None else local
        stats = dict(step=self.current_step, truncating=self.truncating)
        if self.truncating:
            stats.update(denominator=self.denominator, n_hat=value, n_hat_over_denominator=value / self.denominator)
        if self.truncating or self.current_step == self.warmup_steps:
            if not value > 0:
                raise ValueError(f"invalid statistic for step {self.current_step}: {value}")
            self.denominator = float(value)     # C of the next step
        stats["next_denominator"] = self.denominator
        self.step += 1
        return stats

    @staticmethod
    def _lengths(mask_or_lengths):
        if isinstance(mask_or_lengths, torch.Tensor) and mask_or_lengths.dim() == 2:
            return [int(n) for n in mask_or_lengths.bool().sum(-1).tolist()]
        return [int(n) for n in mask_or_lengths]

    # ------------------------------------------------------------------ checkpoints
    def state_dict(self):
        return dict(version=STATE_VERSION, step=self.step, denominator=self.denominator, max_len=self.max_len,
                    block=self.block, warmup_steps=self.warmup_steps, seed=self.seed)

    def load_state_dict(self, state):
        if state.get("version") != STATE_VERSION:
            raise ValueError(f"unsupported STOPD state version {state.get('version')}")
        for key in ("max_len", "block", "warmup_steps", "seed"):
            if state[key] != getattr(self, key):
                raise ValueError(f"checkpoint was made with {key}={state[key]!r}, not {getattr(self, key)!r}")
        self.step, self.denominator = int(state["step"]), state["denominator"]
