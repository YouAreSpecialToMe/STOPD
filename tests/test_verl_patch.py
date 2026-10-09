"""The verl patches against stand-in classes. Running inside verl needs GPUs and is not covered here."""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch

from stopd import normalizer
from stopd import verl_patch as vp
from stopd.truncation import estimated_full_tokens, ipw_weights


def make_server():
    class Server:
        config = SimpleNamespace(response_length=16, max_model_len=32)

        async def generate(self, prompt_ids, sampling_params, request_id, **kwargs):
            self.params = sampling_params
            if "seed" not in sampling_params:            # not a STOPD training request: echo it back
                return sampling_params
            return SimpleNamespace(token_ids=list(range(min(11, sampling_params["max_tokens"]))),
                                   stop_reason="completed", extra_fields={})
    vp.install_server(Server)
    return Server()


def tagged(step=26, sample=3, rollout=0, **extra):
    return {"temperature": 1.0, vp.TAG: dict(step=step, sample=sample, rollout=rollout), **extra}


class PatchTest(unittest.TestCase):
    MODE = "stopd"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, STOPD_MODE=self.MODE, STOPD_MAX_LEN="16", STOPD_BLOCK="4",
                              STOPD_SEED="0", STOPD_FIRST_STEP="26", STOPD_INIT_DENOM="400",
                              STOPD_STATE=str(root / "nhat.json"), STOPD_METRICS=str(root / "metrics.jsonl"))
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()


class ServerTests(PatchTest):
    def test_stop_is_a_block_end_and_the_tag_is_consumed(self):
        server = make_server()
        out = asyncio.run(server.generate([2, 3], tagged(), "r"))
        row = out.extra_fields[vp.FIELD]
        self.assertEqual(server.params["max_tokens"], row["budget"])
        self.assertEqual(row["budget"] % 4, 0)
        self.assertEqual(row["length"], min(11, row["budget"]))
        self.assertNotIn(vp.TAG, server.params)
        self.assertIn("seed", server.params)

    def test_stops_vary_across_rollouts_and_repeat_for_the_same_rollout(self):
        config = SimpleNamespace(response_length=16, max_model_len=32)
        draws = [vp.prepare_request(config, [2], tagged(sample=i)) for i in range(16)]
        self.assertGreater(len({info["budget"] for _, info in draws}), 1)
        self.assertEqual(len({params["seed"] for params, _ in draws}), 16)
        self.assertEqual(vp.prepare_request(config, [2], tagged(sample=5)), draws[5])

    def test_vanilla_generates_full_rollouts_with_the_same_seeds(self):
        config = SimpleNamespace(response_length=16, max_model_len=32)
        stopd_params, _ = vp.prepare_request(config, [2], tagged())
        with patch.dict(os.environ, STOPD_MODE="vanilla"):
            vanilla_params, info = vp.prepare_request(config, [2], tagged())
        self.assertEqual(vanilla_params["max_tokens"], 16)
        self.assertEqual(info["budget"], 16)
        self.assertEqual(vanilla_params["seed"], stopd_params["seed"])

    def test_untagged_requests_pass_through(self):
        server = make_server()
        for params in ({"temperature": 0.7, "max_tokens": 16}, {"prompt_logprobs": 0}):
            self.assertIs(asyncio.run(server.generate([2], params, "v")), params)

    def test_tagged_scoring_request_is_rejected(self):
        with self.assertRaises(ValueError):
            asyncio.run(make_server().generate([2], tagged(prompt_logprobs=0), "t"))

    def test_mismatched_lengths_are_rejected(self):
        for config in (SimpleNamespace(response_length=32, max_model_len=64),
                       SimpleNamespace(response_length=16, max_model_len=17)):
            with self.assertRaises(ValueError):
                vp.prepare_request(config, [2], tagged())

    def test_aborted_or_overlong_rollouts_fail(self):
        info = dict(step=26, sample=0, rollout=0, mode="stopd", budget=8)
        for out in (SimpleNamespace(token_ids=[1], stop_reason="aborted", extra_fields={}),
                    SimpleNamespace(token_ids=list(range(9)), stop_reason="completed", extra_fields={})):
            with self.assertRaises(RuntimeError):
                vp.finish_request(out, info)


class AgentTests(PatchTest):
    def test_only_training_requests_are_tagged(self):
        class Worker:
            async def _run_agent_loop(self, sampling_params, trajectory, **kwargs):
                return sampling_params
        vp.install_agent(Worker)
        outs = {}
        for validate in (False, True):
            params = {"temperature": 1.0}
            outs[validate] = asyncio.run(Worker()._run_agent_loop(
                params, dict(step=3, sample_index=10, rollout_n=0, validate=validate), agent_name="single_turn_agent"))
            self.assertNotIn(vp.TAG, params)
        self.assertEqual(outs[False][vp.TAG], dict(step=3, sample=10, rollout=0))
        self.assertNotIn(vp.TAG, outs[True])

    def test_multi_turn_agents_are_rejected(self):
        class Worker:
            async def _run_agent_loop(self, sampling_params, trajectory, **kwargs):
                return sampling_params
        vp.install_agent(Worker)
        with self.assertRaises(ValueError):
            asyncio.run(Worker()._run_agent_loop({}, dict(step=1, sample_index=0, rollout_n=0, validate=False),
                                                 agent_name="tool_agent"))


class TrainerHelpers:
    def trainer(self, loss_mode="stopd_k1", batch_size=3):
        class Trainer:
            def _update_actor(self, batch):
                self.seen = dict(batch.batch)
                return SimpleNamespace(meta_info={"metrics": {}})
        vp.install_trainer(Trainer)
        t = Trainer()
        t.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(ppo_epochs=1, loss_agg_mode="token-mean", ppo_mini_batch_size=batch_size),
            rollout=SimpleNamespace(n=1)))
        t.distillation_config = SimpleNamespace(distillation_loss=SimpleNamespace(
            loss_mode=loss_mode, use_policy_gradient=True, use_task_rewards=False, loss_max_clamp=None))
        return t

    def batch(self, step, lengths, mode="stopd"):
        mask = torch.arange(16)[None, :] < torch.tensor(lengths)[:, None]
        rows = [dict(step=step, sample=i, rollout=0, mode=mode, budget=16, length=n) for i, n in enumerate(lengths)]
        return SimpleNamespace(batch={"response_mask": mask}, non_tensor_batch={vp.FIELD: rows})


class TrainerTests(TrainerHelpers, PatchTest):
    def test_lagged_denominator_chain(self):
        t = self.trainer()
        lengths = [16, 5, 9]
        t.global_steps = 26
        result = t._update_actor(self.batch(26, lengths))
        torch.testing.assert_close(t.seen["stopd_scale"], torch.full((3,), sum(lengths) / 400))   # C_26 = N_25
        self.assertEqual(t.seen["stopd_length"].tolist(), lengths)
        n_hat = estimated_full_tokens(lengths, 16, 4)
        self.assertAlmostEqual(normalizer.history(os.environ["STOPD_STATE"])[26], n_hat)
        self.assertEqual(result.meta_info["metrics"]["stopd/denominator"], 400)
        t.global_steps = 27
        t._update_actor(self.batch(27, [4, 4, 4]))
        torch.testing.assert_close(t.seen["stopd_scale"], torch.full((3,), 12 / n_hat))         # C_27 = N_hat_26
        rows = [json.loads(line) for line in Path(os.environ["STOPD_METRICS"]).read_text().splitlines()]
        self.assertEqual([r["step"] for r in rows], [26, 27])
        self.assertEqual(rows[0]["tokens"], 30)

    def test_records_must_match_the_batch(self):
        t = self.trainer()
        t.global_steps = 26
        bad_step = self.batch(26, [3, 3, 3])
        bad_step.non_tensor_batch[vp.FIELD][1]["step"] = 25
        bad_length = self.batch(26, [3, 3, 3])
        bad_length.non_tensor_batch[vp.FIELD][2]["length"] = 4
        missing = self.batch(26, [3, 3, 3])
        missing.non_tensor_batch.pop(vp.FIELD)
        for batch in (bad_step, bad_length, missing):
            with self.assertRaises(ValueError):
                t._update_actor(batch)

    def test_config_is_checked(self):
        t = self.trainer(loss_mode="k1")
        t.global_steps = 26
        with self.assertRaises(ValueError):
            t._update_actor(self.batch(26, [3, 3, 3]))
        t = self.trainer(batch_size=6)
        t.global_steps = 26
        with self.assertRaises(ValueError):
            t._update_actor(self.batch(26, [3, 3, 3]))


class VanillaTrainerTests(TrainerHelpers, PatchTest):
    MODE = "vanilla"

    def test_logs_the_token_count_for_the_first_stopd_denominator(self):
        t = self.trainer(loss_mode="k1")
        t.global_steps = 25
        result = t._update_actor(self.batch(25, [16, 5, 9], mode="vanilla"))
        self.assertNotIn("stopd_scale", t.seen)
        self.assertEqual(result.meta_info["metrics"]["stopd/tokens"], 30)
        row = json.loads(Path(os.environ["STOPD_METRICS"]).read_text().splitlines()[0])
        self.assertEqual((row["mode"], row["step"], row["tokens"]), ("vanilla", 25, 30))   # N_25 for C_26

    def test_config_is_checked(self):
        t = self.trainer(loss_mode="stopd_k1")
        t.global_steps = 1
        with self.assertRaises(ValueError):
            t._update_actor(self.batch(1, [3, 3, 3], mode="vanilla"))


class LossWeightTests(PatchTest):
    def test_weighted_loss_survives_micro_batch_splits(self):
        L, K, C = 16, 4, 37.0
        lengths = [1, 16, 6]
        mask = torch.arange(L)[None, :] < torch.tensor(lengths)[:, None]
        length, scale = vp.scale_tensors(lengths, C, "cpu")
        data = dict(stopd_length=length, stopd_scale=scale)
        x = torch.arange(1, 3 * L + 1, dtype=torch.float32).reshape(3, L).requires_grad_()
        expected = (x.square() * mask * ipw_weights(L, L, K)).sum() / C
        # verl's token-mean: each micro-batch's loss sum is divided by the whole batch's N_obs
        full = vp.weight_losses(x.square(), mask, data).sum() / sum(lengths)
        split = sum(vp.weight_losses(x[i:j].square(), mask[i:j], {k: v[i:j] for k, v in data.items()}).sum()
                    for i, j in [(0, 1), (1, 3)]) / sum(lengths)
        for actual in (full, split):
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(torch.autograd.grad(actual, x, retain_graph=True)[0],
                                       torch.autograd.grad(expected, x, retain_graph=True)[0])

    def test_missing_or_mismatched_metadata_is_rejected(self):
        mask = torch.arange(16)[None, :] < torch.tensor([3, 5])[:, None]
        losses = torch.ones(2, 16)
        with self.assertRaises(RuntimeError):
            vp.weight_losses(losses, mask, {})
        length, scale = vp.scale_tensors([3, 6], 10.0, "cpu")
        with self.assertRaises(ValueError):
            vp.weight_losses(losses, mask, dict(stopd_length=length, stopd_scale=scale))


class DisabledTests(unittest.TestCase):
    def test_nothing_is_patched_without_stopd_mode(self):
        class Server:
            async def generate(self, *args, **kwargs):
                return "stock"
        original = Server.generate
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STOPD_MODE", None)
            vp.install_server(Server)
        self.assertIs(Server.generate, original)
        with patch.dict(os.environ, STOPD_MODE="ipw"):
            with self.assertRaises(ValueError):
                vp.enabled()


if __name__ == "__main__":
    unittest.main()
