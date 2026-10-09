"""The NeMo-RL STOPD loss against NeMo-RL's real DistillationLossFn (needs a patched checkout).

Set NEMO_RL_SRC to a NeMo-RL checkout with integrations/nemo_rl/stopd.patch applied.
The test loads NeMo-RL's own DistillationLossFn and masked_mean source, with
stand-ins only for the enum/protocol imports, plus the patch's
nemo_rl/algorithms/stopd.py, so no GPU, Ray or vLLM is needed.
"""

import ast
import enum
import os
import pickle
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch

SRC = os.environ.get("NEMO_RL_SRC")


def load_nemo_rl(src):
    """Build nemo_rl.algorithms.{utils,loss.loss_functions,stopd} from the checkout's real code."""
    def source_of(path, names):
        tree = ast.parse(Path(path).read_text())
        return "\n\n".join(ast.get_source_segment(Path(path).read_text(), node) for node in tree.body
                           if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names)

    for name in ("nemo_rl", "nemo_rl.algorithms", "nemo_rl.algorithms.loss"):
        sys.modules[name] = types.ModuleType(name)
    utils = types.ModuleType("nemo_rl.algorithms.utils")
    exec("from typing import Optional\nimport torch\n" + source_of(f"{src}/nemo_rl/algorithms/utils.py", {"masked_mean"}),
         utils.__dict__)
    losses = types.ModuleType("nemo_rl.algorithms.loss.loss_functions")
    losses.__dict__.update(torch=torch, Any=object, TypedDict=dict, masked_mean=utils.masked_mean,
                           LossFunction=object,
                           LossType=enum.Enum("LossType", "TOKEN_LEVEL SEQUENCE_LEVEL"),
                           LossInputType=enum.Enum("LossInputType", "DISTILLATION LOGIT LOGPROB"))
    exec("from __future__ import annotations\n" +
         source_of(f"{src}/nemo_rl/algorithms/loss/loss_functions.py", {"DistillationLossFn"}), losses.__dict__)
    sys.modules["nemo_rl.algorithms.utils"] = utils
    sys.modules["nemo_rl.algorithms.loss.loss_functions"] = losses
    stopd = types.ModuleType("nemo_rl.algorithms.stopd")
    stopd.__file__ = f"{src}/nemo_rl/algorithms/stopd.py"
    exec(compile(Path(stopd.__file__).read_text(), stopd.__file__, "exec"), stopd.__dict__)
    sys.modules["nemo_rl.algorithms.stopd"] = stopd
    return losses, stopd


def master_config(**overrides):
    distillation = SimpleNamespace(num_prompts_per_step=2, num_generations_per_prompt=2, max_rollout_turns=1, seed=7)
    policy = {"train_global_batch_size": 4,
              "generation": {"max_new_tokens": 16, "vllm_cfg": {"async_engine": False, "max_model_len": 24}}}
    config = SimpleNamespace(distillation=distillation, policy=policy, data={"max_input_seq_length": 8})
    for key, value in overrides.items():
        target, field = key.split(".", 1)
        node = getattr(config, target)
        *path, last = field.split(".")
        for p in path:
            node = node[p] if isinstance(node, dict) else getattr(node, p)
        if isinstance(node, dict):
            node[last] = value
        else:
            setattr(node, last, value)
    return config


@unittest.skipUnless(SRC and Path(SRC, "nemo_rl/algorithms/stopd.py").exists(),
                     "set NEMO_RL_SRC to a NeMo-RL checkout with integrations/nemo_rl/stopd.patch applied")
class NeMoRLLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.losses, cls.stopd = load_nemo_rl(SRC)

    def setUp(self):
        self.cfg = SimpleNamespace(kl_type="reverse", mixed_kl_weight=0.5, zero_outside_topk=False)
        torch.manual_seed(0)
        B, S, k = 4, 12, 5
        self.student = torch.log_softmax(torch.randn(B, S - 1, k), -1)
        self.teacher = torch.log_softmax(torch.randn(B, S - 1, k), -1)
        lengths, prompt = [8, 3, 5, 1], 3
        self.token_mask = torch.zeros(B, S, dtype=torch.long)       # [prompt][response][pad]
        for i, n in enumerate(lengths):
            self.token_mask[i, prompt:prompt + n] = 1
        self.data = {"token_mask": self.token_mask, "sample_mask": torch.ones(B), "input_ids": torch.zeros(B, S)}
        self.lengths = lengths

    def make(self, **overrides):
        stopd_cfg = {"block": 4, "warmup_steps": 1}
        stopd_cfg.update(overrides.pop("stopd", {}))
        return self.stopd.STOPDDistillationLossFn(self.cfg, stopd_cfg, master_config(**overrides))

    def per_token_kl(self):
        p = self.student.exp()
        return (p * (self.student - self.teacher)).sum(-1)       # NeMo-RL's reverse KL over the top-k

    def test_warmup_step_is_the_stock_loss(self):
        loss_fn = self.make()
        loss_fn.begin_step(0)
        stock = self.losses.DistillationLossFn(self.cfg)
        args = (self.student, self.teacher, None, self.data, torch.tensor(4.0), torch.tensor(17.0))
        torch.testing.assert_close(loss_fn(*args)[0], stock(*args)[0])
        self.assertTrue(torch.equal(loss_fn.budgets(4), torch.full((4,), 16)))

    def test_stopd_step_weights_tokens_and_divides_by_c(self):
        loss_fn = self.make()
        loss_fn.begin_step(0)
        stats = loss_fn.end_step(self.data)
        self.assertEqual(stats["stopd/next_denominator"], float(sum(self.lengths)))   # C_2 = N_1
        loss_fn.begin_step(1)
        self.assertEqual(loss_fn.denominator, 17.0)
        loss, _ = loss_fn(self.student, self.teacher, None, self.data, torch.tensor(4.0), torch.tensor(999.0))
        weights = self.stopd.ipw_weights(self.token_mask, 16, 4)[:, 1:]                 # NeMo-RL's [:, 1:] shift
        expected = (self.per_token_kl() * weights).sum() / 17.0
        torch.testing.assert_close(loss, expected)
        from stopd import token_weights

        torch.testing.assert_close(self.stopd.ipw_weights(self.token_mask, 16, 4),
                                   token_weights(self.token_mask, 16, 4))
        budgets = loss_fn.budgets(64)
        self.assertTrue(all(b % 4 == 0 and 4 <= b <= 16 for b in budgets.tolist()))

    def test_workers_get_no_schedule_and_resume_restores_it(self):
        loss_fn = self.make()
        loss_fn.begin_step(0)
        loss_fn.end_step(self.data)
        loss_fn.begin_step(1)
        sent = pickle.loads(pickle.dumps(loss_fn))
        self.assertIsNone(sent.schedule)
        self.assertEqual((sent.denominator, sent.max_len, sent.block), (17.0, 16, 4))
        resumed = self.stopd.STOPDDistillationLossFn(self.cfg, {"block": 4, "warmup_steps": 1}, master_config(),
                                                    {"stopd": loss_fn.state_dict()})
        resumed.begin_step(1)
        self.assertEqual(resumed.denominator, 17.0)
        with self.assertRaises(RuntimeError):
            resumed.begin_step(5)

    def test_unsupported_configs_are_rejected(self):
        for overrides in ({"policy.train_global_batch_size": 2}, {"distillation.max_rollout_turns": 2},
                          {"policy.generation.vllm_cfg.async_engine": True},
                          {"policy.generation.vllm_cfg.max_model_len": 20}):
            with self.assertRaises(ValueError):
                self.make(**overrides)


if __name__ == "__main__":
    unittest.main()
