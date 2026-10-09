"""STOPDTrainer on real TRL with tiny models (skipped without trl).

The tests run on the GPU when exactly one is visible, with TRL's own Triton
divergence kernel. Otherwise they run on CPU, where that kernel is replaced by
a plain PyTorch implementation with the same interface; everything else (TRL's
generation, batching, loss reduction and checkpointing) is TRL's own code.
"""

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

HAVE_TRL = importlib.util.find_spec("trl") is not None
STUDENT = "trl-internal-testing/tiny-Qwen3ForCausalLM"
TEACHER = "trl-internal-testing/small-Qwen3ForCausalLM"


def install_reference_kernel():
    """Use TRL's kernel on a single visible GPU (it needs Triton), else a reference one (reverse KL, beta = 1)."""
    import torch

    if torch.cuda.device_count() == 1:
        try:
            import trl.kernels.chunked_divergence  # noqa: F401
            return
        except Exception:
            pass

    class ChunkedDivergenceFunction:
        @staticmethod
        def apply(h_s, w_s, b_s, h_t, w_t, b_t, beta, temperature, s_scale, s_softcap, t_scale, t_softcap):
            if beta != 1.0:
                raise NotImplementedError("the reference kernel implements beta = 1 (reverse KL)")

            def log_probs(h, w, b, scale):
                z = h.float() @ w.float().t()
                if b is not None:
                    z = z + b.float()
                return torch.log_softmax(z * scale / temperature, -1)

            s, t = log_probs(h_s, w_s, b_s, s_scale), log_probs(h_t, w_t, b_t, t_scale)
            return (s.exp() * (s - t)).sum(-1), -(s.exp() * s).sum(-1).detach()

    module = types.ModuleType("trl.kernels.chunked_divergence")
    module.ChunkedDivergenceFunction = ChunkedDivergenceFunction
    sys.modules["trl.kernels.chunked_divergence"] = module


@unittest.skipUnless(HAVE_TRL, "trl is not installed")
class TRLIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_reference_kernel()
        from datasets import load_dataset

        cls.dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, max_steps, **overrides):
        from stopd.integrations.trl import STOPDConfig, STOPDTrainer

        import torch

        # A single process: TRL refuses nn.DataParallel, so the tests use a GPU only when exactly one is visible.
        kwargs = dict(output_dir=self.tmp.name, per_device_train_batch_size=4, gradient_accumulation_steps=2,
                      max_completion_length=16, stopd_block=4, stopd_warmup_steps=1, max_steps=max_steps,
                      save_steps=2, learning_rate=1e-3, beta=1.0, temperature=1.0, report_to="none",
                      use_cpu=torch.cuda.device_count() != 1, bf16=False, logging_steps=1, seed=0)
        kwargs.update(overrides)
        return STOPDTrainer(model=STUDENT, teacher_model=TEACHER, args=STOPDConfig(**kwargs),
                            train_dataset=self.dataset)

    def test_rollouts_respect_their_budgets_and_the_denominator_chain(self):
        trainer = self.make(max_steps=3)
        seen = []
        original = trainer._generate_and_score_completions

        def record(inputs):
            step, truncating, denominator = trainer.stopd.current_step, trainer.stopd.truncating, trainer.stopd.denominator
            budgets = trainer.stopd.budgets(8)
            out = original(inputs)
            seen.append((step, truncating, denominator, budgets, out))
            return out

        trainer._generate_and_score_completions = record
        trainer.train()
        self.assertEqual([s[0] for s in seen], [1, 2, 3])
        lengths = {step: out["completion_mask"].sum(1).tolist() for step, _, _, _, out in seen}
        # warmup step: full rollouts, TRL's own token count, no weights
        _, truncating, _, _, out = seen[0]
        self.assertFalse(truncating)
        self.assertNotIn("stopd_weights", out)
        self.assertEqual(out["num_items_in_batch"].item(), sum(lengths[1]))
        # STOPD steps: every rollout within its budget, some actually cut, C_2 = N_1 and C_3 = N_hat_2
        from stopd import estimated_full_tokens

        cut = 0
        for step, truncating, denominator, budgets, out in seen[1:]:
            self.assertTrue(truncating)
            self.assertAlmostEqual(out["num_items_in_batch"].item(), denominator, places=3)
            for n, b in zip(lengths[step], budgets):
                self.assertLessEqual(n, b)
                cut += n == b < 16
        self.assertGreater(cut, 0)
        self.assertEqual(seen[1][2], float(sum(lengths[1])))
        self.assertAlmostEqual(seen[2][2], estimated_full_tokens(lengths[2], 16, 4), places=3)
        # the state of step 2 is saved with checkpoint-2
        state = json.loads(Path(self.tmp.name, "checkpoint-2", "stopd_state.json").read_text())
        self.assertEqual((state["step"], state["denominator"]), (2, seen[2][2]))
        self.assertEqual(trainer.stopd.step, 3)

    def test_weights_reach_the_right_tokens(self):
        import torch

        trainer = self.make(max_steps=1)
        trainer.model.train()
        torch.manual_seed(0)
        B, P, T = 2, 3, 5
        completion_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]])
        device = trainer.accelerator.device
        completion_mask = completion_mask.to(device)
        inputs = dict(prompt_ids=torch.randint(0, 1000, (B, P), device=device),
                      prompt_mask=torch.ones(B, P, dtype=torch.long, device=device),
                      completion_ids=torch.randint(0, 1000, (B, T), device=device), completion_mask=completion_mask)
        student = trainer.accelerator.unwrap_model(trainer.model)
        positions = completion_mask.nonzero().tolist()
        tol = 1e-4 if device.type == "cpu" else 1e-3   # relative; the GPU kernel sums in a different order
        per_token = []
        for i, j in positions:                                       # one-hot weights isolate each token
            w = torch.zeros(B, T, device=device)
            w[i, j] = 1.0
            loss, _, _ = trainer._compute_loss(student, dict(inputs, stopd_weights=w), torch.tensor(1.0))
            per_token.append(loss.item())
        stock, _, _ = trainer._compute_loss(student, inputs, torch.tensor(1.0))
        self.assertAlmostEqual(sum(per_token), stock.item(), delta=tol * max(1.0, abs(stock.item())))
        w = torch.arange(B * T, dtype=torch.float32, device=device).reshape(B, T) + 1
        weighted, _, _ = trainer._compute_loss(student, dict(inputs, stopd_weights=w), torch.tensor(4.0))
        expected = sum(w[i, j].item() * h for (i, j), h in zip(positions, per_token)) / 4.0
        self.assertAlmostEqual(weighted.item(), expected, delta=tol * max(1.0, abs(expected)))

    def test_generation_stops_each_row_at_its_budget(self):
        from unittest import mock

        trainer = self.make(max_steps=1, stopd_warmup_steps=0, stopd_initial_denominator=100.0)
        trainer.model.train()
        model = trainer.accelerator.unwrap_model(trainer.model_wrapped)
        raw, original = [], model.generate

        def spy(*args, **kwargs):
            out = original(*args, **kwargs)
            raw.append(out)
            return out

        prompts = [[9707, 1879, 11], [785, 3974, 13876, 38835, 30], [9707], [17, 488, 220, 17]]
        budgets = [4, 8, 12, 16]
        with mock.patch.object(model, "generate", spy), \
                mock.patch.object(trainer.stopd, "budgets", lambda n, offset=0: budgets):
            completions = trainer._generate_single_turn(prompts, None, {})
        width = max(len(p) for p in prompts)
        generated = raw[0][:, width:]
        pad = trainer.processing_class.pad_token_id
        self.assertGreaterEqual(generated.shape[1], 16)
        for row, budget, ids in zip(generated.tolist(), budgets, completions):
            self.assertLessEqual(len(ids), budget)
            self.assertTrue(all(token == pad for token in row[budget:]), "row kept generating past its budget")

    def test_resume_restores_the_state(self):
        trainer = self.make(max_steps=2)
        trainer.train()
        saved = json.loads(Path(self.tmp.name, "checkpoint-2", "stopd_state.json").read_text())
        resumed = self.make(max_steps=3)
        resumed.train(resume_from_checkpoint=os.path.join(self.tmp.name, "checkpoint-2"))
        self.assertEqual(resumed.stopd.step, 3)
        fresh = self.make(max_steps=3)
        loaded = json.loads(Path(self.tmp.name, "checkpoint-2", "stopd_state.json").read_text())
        self.assertEqual(loaded, saved)
        os.remove(Path(self.tmp.name, "checkpoint-2", "stopd_state.json"))
        with self.assertRaises(FileNotFoundError):
            fresh.train(resume_from_checkpoint=os.path.join(self.tmp.name, "checkpoint-2"))


if __name__ == "__main__":
    unittest.main()
