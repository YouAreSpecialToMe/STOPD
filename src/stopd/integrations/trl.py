"""STOPD for Hugging Face TRL's on-policy `DistillationTrainer` (tested with trl 1.15.0).

    from stopd.integrations.trl import STOPDConfig, STOPDTrainer

    args = STOPDConfig(output_dir="out", max_completion_length=16384,
                       stopd_block=128, stopd_warmup_steps=25)
    trainer = STOPDTrainer(model="Qwen/Qwen3-1.7B-Base", teacher_model="Qwen/Qwen3-4B-Instruct-2507",
                           args=args, train_dataset=dataset)
    trainer.train()

Compared with `DistillationTrainer`:

* Generation. After the warmup, each training rollout gets its own token budget,
  a random block end of `max_completion_length` (L). With vLLM in colocate mode
  every prompt gets its own SamplingParams; with transformers' `generate` each
  row stops at its budget (the batch still runs until its longest row ends, so
  the savings come mostly from the shorter loss computation). vLLM server mode
  is not supported, because it sends one `max_tokens` per request batch.
* Loss. TRL's per-token divergences are multiplied by the IPW weights before
  they are summed, and the sum over the global batch is divided by C_k instead
  of the batch's token count.
* Checkpoints. The STOPD state is written to `stopd_state.json` in every
  checkpoint and read back by `train(resume_from_checkpoint=...)`.

Unsupported: tools (multi-turn rollouts) and vLLM server mode.
"""

import contextlib
import json
import os
from dataclasses import dataclass, field
from unittest import mock

import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import DistillationConfig, DistillationTrainer

from stopd.core import STOPD, token_weights

STATE_FILE = "stopd_state.json"


@dataclass
class STOPDConfig(DistillationConfig):
    """`DistillationConfig` plus STOPD's settings; `max_completion_length` is L."""

    stopd_block: int = field(default=128, metadata={"help": "Block width K; the largest IPW weight is L / K."})
    stopd_warmup_steps: int = field(
        default=0,
        metadata={"help": "Initial Vanilla OPD steps; the last one's token count is the first STOPD denominator."})
    stopd_initial_denominator: float | None = field(
        default=None,
        metadata={"help": "First STOPD denominator when stopd_warmup_steps is 0, e.g. a full batch's token count."})


class _StopAtBudget(StoppingCriteria):
    """Per-row stop for transformers' generate: row i is done once it has budgets[i] new tokens."""

    def __init__(self, budgets, prompt_width):
        self.budgets = torch.as_tensor(budgets)
        self.prompt_width = prompt_width

    def __call__(self, input_ids, scores, **kwargs):
        return (input_ids.shape[1] - self.prompt_width) >= self.budgets.to(input_ids.device)


class _SaveState(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}", STATE_FILE)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as file:
                json.dump(self.trainer.stopd.state_dict(), file)


class STOPDTrainer(DistillationTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.args
        if not isinstance(config, STOPDConfig):
            raise TypeError("STOPDTrainer needs a STOPDConfig")
        if self.use_vllm and config.vllm_mode == "server":
            raise NotImplementedError("STOPD needs a token budget per rollout, which TRL's vLLM server mode cannot "
                                      "send; use vllm_mode='colocate' or transformers generation")
        if getattr(self, "tools", None):
            raise NotImplementedError("STOPD supports single-turn rollouts only")
        self.stopd = STOPD(max_len=config.max_completion_length, block=config.stopd_block,
                           warmup_steps=config.stopd_warmup_steps,
                           initial_denominator=config.stopd_initial_denominator, seed=config.seed)
        self.add_callback(_SaveState(self))

    # ------------------------------------------------------------------ resume
    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        if resume_from_checkpoint is True:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
        if resume_from_checkpoint:
            path = os.path.join(resume_from_checkpoint, STATE_FILE)
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} is missing; this checkpoint was not written by STOPDTrainer")
            with open(path) as file:
                self.stopd.load_state_dict(json.load(file))
        return super().train(resume_from_checkpoint=resume_from_checkpoint, trial=trial,
                             ignore_keys_for_eval=ignore_keys_for_eval, **kwargs)

    # ------------------------------------------------------------------ generation
    def _rollout_offset(self, n):
        """Global index of this process's first rollout (every process generates n)."""
        return self.accelerator.process_index * n

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields, has_tool_images=False):
        if not (self.model.training and self.stopd.truncating):
            return super()._generate_single_turn(prompt_ids, images, multimodal_fields, has_tool_images)
        if self.stopd.step != self.state.global_step:
            raise RuntimeError(f"STOPD state is at step {self.stopd.step} but the trainer at {self.state.global_step}; "
                               "resume through STOPDTrainer.train(resume_from_checkpoint=...)")
        budgets = self.stopd.budgets(len(prompt_ids), offset=self._rollout_offset(len(prompt_ids)))
        with self._limit_generation(budgets, prompt_ids):
            completions = super()._generate_single_turn(prompt_ids, images, multimodal_fields, has_tool_images)
        # Also drops the pad token that transformers writes after a row stopped at its budget, which TRL would
        # otherwise read as an EOS when pad and EOS share an id.
        return [list(ids)[:b] for ids, b in zip(completions, budgets)]

    @contextlib.contextmanager
    def _limit_generation(self, budgets, prompt_ids):
        if self.use_vllm:
            generation = self.vllm_generation
            llm = generation.llm
            original = llm.generate
            tp = generation.tensor_parallel_size
            # With tensor parallelism, vLLM receives the prompts of every process in the group, in process order.
            first = (self.accelerator.process_index // tp) * tp * len(prompt_ids)

            def generate(prompts, sampling_params=None, **kwargs):
                group_budgets = self.stopd.budgets(len(prompts), offset=first)
                params = []
                for budget in group_budgets:
                    p = sampling_params.clone()
                    p.max_tokens = budget
                    params.append(p)
                return original(prompts, sampling_params=params, **kwargs)

            with mock.patch.object(llm, "generate", generate):
                yield
        else:
            model = self.accelerator.unwrap_model(self.model_wrapped)
            original = model.generate
            width = max(len(p) for p in prompt_ids)   # prompts are left-padded to the longest

            def generate(*args, **kwargs):
                criteria = StoppingCriteriaList(kwargs.pop("stopping_criteria", None) or [])
                criteria.append(_StopAtBudget(budgets, width))
                return original(*args, stopping_criteria=criteria, **kwargs)

            with mock.patch.object(model, "generate", generate):
                yield

    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        if not self.model.training:
            return out
        mask = out["completion_mask"]
        if self.stopd.truncating:
            # Per-sample weights follow their rows through TRL's shuffle and micro-batch split; the denominator of
            # this step replaces the token count (compute_loss divides it by the process count, which DDP's
            # gradient averaging undoes).
            out["stopd_weights"] = token_weights(mask, self.stopd.max_len, self.stopd.block)
            out["num_items_in_batch"] = torch.tensor(self.stopd.denominator, dtype=torch.float32, device=mask.device)
        stats = self.stopd.end_step(mask, reduce=self._sum_over_processes)
        for key, value in stats.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and key != "step":
                self._metrics["train"][f"stopd/{key}"].append(float(value))
        return out

    def _sum_over_processes(self, value, group=None):
        tensor = torch.tensor(float(value), dtype=torch.float64, device=self.accelerator.device)
        return self.accelerator.reduce(tensor, reduction="sum").item()

    # ------------------------------------------------------------------ loss
    def _compute_loss(self, unwrapped_student, inputs, num_items_in_batch):
        weights = inputs.get("stopd_weights")
        if weights is None:
            return super()._compute_loss(unwrapped_student, inputs, num_items_in_batch)
        if "tool_mask" in inputs:
            raise NotImplementedError("STOPD supports single-turn rollouts only")
        from trl.kernels.chunked_divergence import ChunkedDivergenceFunction

        # TRL computes the divergence at the nonzero entries of the completion mask, in row-major order.
        weights = weights.reshape(-1)[inputs["completion_mask"].reshape(-1) != 0]
        original = ChunkedDivergenceFunction.apply

        def apply(*args):
            divergence, entropy = original(*args)
            return divergence * weights.to(divergence.device, divergence.dtype), entropy

        with mock.patch.object(ChunkedDivergenceFunction, "apply", apply):
            return super()._compute_loss(unwrapped_student, inputs, num_items_in_batch)
