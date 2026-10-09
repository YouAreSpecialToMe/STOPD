"""The framework-agnostic STOPD class."""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch

from stopd import STOPD, estimated_full_tokens, ipw_weights, response_positions, token_weights


def right_padded(lengths, width):
    return torch.arange(width)[None, :] < torch.tensor(lengths)[:, None]


def left_padded_with_prompt(lengths, prompt, width):
    # [pad ... pad][prompt][response] with the mask on the response only
    rows = []
    for n in lengths:
        pad = width - prompt - n
        rows.append([False] * pad + [False] * prompt + [True] * n)
    return torch.tensor(rows)


class WeightTests(unittest.TestCase):
    def test_positions_and_weights_do_not_depend_on_the_layout(self):
        L, K, lengths = 16, 4, [16, 5, 9, 1]
        right = right_padded(lengths, 16)
        left = left_padded_with_prompt(lengths, prompt=3, width=24)
        w = ipw_weights(16, L, K)
        for mask in (right, left):
            got = token_weights(mask, L, K)
            for row, n in enumerate(lengths):
                torch.testing.assert_close(got[row][mask[row]], w[:n])
                self.assertEqual(got[row][~mask[row]].abs().sum().item(), 0)
                self.assertEqual(response_positions(mask)[row][mask[row]].tolist(), list(range(n)))

    def test_too_long_response_rejected(self):
        with self.assertRaises(ValueError):
            token_weights(right_padded([9], 9), 8, 4)


class ScheduleTests(unittest.TestCase):
    def test_warmup_is_vanilla_then_stops_are_block_ends(self):
        s = STOPD(max_len=16, block=4, warmup_steps=2)
        self.assertEqual(s.budgets(5), [16] * 5)
        mask = right_padded([16, 7, 3], 16)
        torch.testing.assert_close(s.weights(mask), mask.float())
        s.end_step(mask, reduce=None)
        self.assertIsNone(s.denominator)
        s.end_step(mask, reduce=None)                     # last warmup step: C_3 = its exact token count
        self.assertEqual(s.denominator, 26.0)
        self.assertTrue(s.truncating)
        budgets = s.budgets(64)
        self.assertTrue(all(b % 4 == 0 and 4 <= b <= 16 for b in budgets))
        self.assertGreater(len(set(budgets)), 1)
        self.assertEqual(s.budgets(8, offset=10), budgets[10:18])   # global indices, any split across ranks

    def test_budgets_depend_on_seed_and_step(self):
        a, b = STOPD(64, 4, 0, 100.0, seed=0), STOPD(64, 4, 0, 100.0, seed=1)
        self.assertNotEqual(a.budgets(32), b.budgets(32))
        first = a.budgets(32)
        a.end_step([10] * 4, reduce=None)
        self.assertNotEqual(a.budgets(32), first)

    def test_denominator_chain(self):
        s = STOPD(max_len=16, block=4, warmup_steps=0, initial_denominator=400.0)
        lengths = [16, 5, 9]
        stats = s.end_step(lengths, reduce=None)
        n_hat = estimated_full_tokens(lengths, 16, 4)
        self.assertEqual(stats["denominator"], 400.0)
        self.assertAlmostEqual(s.denominator, n_hat)
        self.assertAlmostEqual(stats["n_hat_over_denominator"], n_hat / 400.0)

    def test_reduce_sums_over_ranks(self):
        s = STOPD(max_len=16, block=4, warmup_steps=1)
        calls = []
        s.end_step([3, 4], reduce=lambda value, group: calls.append(value) or value * 2)
        self.assertEqual(calls, [7.0])
        self.assertEqual(s.denominator, 14.0)

    def test_state_round_trip_and_config_check(self):
        s = STOPD(max_len=16, block=4, warmup_steps=1)
        s.end_step([5, 6], reduce=None)
        s.end_step([5, 6], reduce=None)
        t = STOPD(max_len=16, block=4, warmup_steps=1)
        t.load_state_dict(s.state_dict())
        self.assertEqual((t.step, t.denominator, t.budgets(10)), (s.step, s.denominator, s.budgets(10)))
        with self.assertRaises(ValueError):
            STOPD(max_len=16, block=8, warmup_steps=1).load_state_dict(s.state_dict())

    def test_invalid_configs_rejected(self):
        for kwargs in (dict(max_len=8, block=9, warmup_steps=1), dict(max_len=8, block=4),
                       dict(max_len=8, block=4, warmup_steps=-1), dict(max_len=8, block=4, initial_denominator=0)):
            with self.assertRaises(ValueError):
                STOPD(**kwargs)


class LossTests(unittest.TestCase):
    def test_loss_is_unbiased_for_the_vanilla_sum(self):
        # E over the stops of grad(sum_t w_t h_t) equals grad(sum over the complete rollout), for every length.
        L, K = 8, 4
        s = STOPD(max_len=L, block=K, warmup_steps=0, initial_denominator=1.0)
        for T in range(1, L + 1):
            x = torch.arange(1, L + 1, dtype=torch.float64, requires_grad=True)
            h = (x.square() * torch.tensor([(-1) ** t for t in range(L)], dtype=torch.float64))[None, :]
            total = 0
            for anchor in range(L):
                G = min(T, min((anchor // K + 1) * K, L))
                total = total + s.loss(h, right_padded([G], L)) / L
            torch.testing.assert_close(torch.autograd.grad(total, x, retain_graph=True)[0],
                                       torch.autograd.grad(h[0, :T].sum(), x)[0], rtol=1e-5, atol=1e-5)

    def test_loss_adds_up_over_micro_batches_and_matches_the_multiplier(self):
        s = STOPD(max_len=16, block=4, warmup_steps=0, initial_denominator=37.0)
        lengths = [1, 16, 6, 11]
        mask = right_padded(lengths, 16)
        h = torch.randn(4, 16)
        full = s.loss(h, mask)
        parts = s.loss(h[:1], mask[:1]) + s.loss(h[1:], mask[1:])
        torch.testing.assert_close(full, parts)
        expected = (h * ipw_weights(16, 16, 4)[None, :] * mask).sum() / 37.0
        torch.testing.assert_close(full, expected)
        n_obs = sum(lengths)                                   # a framework dividing by the global token count
        torch.testing.assert_close((h * s.multiplier(mask, n_obs)).sum() / n_obs, expected)

    def test_warmup_loss_is_the_token_mean(self):
        s = STOPD(max_len=16, block=4, warmup_steps=3)
        mask = right_padded([4, 8], 16)
        h = torch.randn(2, 16)
        torch.testing.assert_close(s.loss(h, mask, n_obs=12), (h * mask).sum() / 12)
        torch.testing.assert_close(s.multiplier(mask, 12), mask.float())
        with self.assertRaises(ValueError):
            s.loss(h, mask)

    def test_simulated_run_matches_the_full_gradient_on_average(self):
        # Monte Carlo over many steps' stops: the STOPD gradient averages to the Vanilla one.
        rng = random.Random(0)
        L, K = 32, 8
        T = [rng.randrange(1, L + 1) for _ in range(6)]
        x = torch.randn(len(T), L, dtype=torch.float64, requires_grad=True)
        full_mask = right_padded(T, L)
        vanilla = torch.autograd.grad((x.sin() * full_mask).sum() / sum(T), x)[0]
        s = STOPD(max_len=L, block=K, warmup_steps=0, initial_denominator=float(sum(T)), seed=3)
        acc, n = torch.zeros_like(x), 3000
        for _ in range(n):
            s.denominator = float(sum(T))          # hold C fixed to isolate the weighting
            mask = right_padded([min(t, b) for t, b in zip(T, s.budgets(len(T)))], L)
            acc += torch.autograd.grad(s.loss(x.sin(), mask), x)[0].double()
            s.end_step(mask, reduce=None)
        torch.testing.assert_close(acc / n, vanilla, atol=0.02, rtol=0.1)


if __name__ == "__main__":
    unittest.main()
