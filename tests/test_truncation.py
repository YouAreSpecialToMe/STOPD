"""Exact checks of the truncation distribution, the IPW weights and the lagged normalizer."""

import random
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch

from stopd import normalizer
from stopd.truncation import (estimated_full_tokens, ipw_weights, keep_probability, max_weight,
                              request_seed, stop_position)

SHAPES = [(7, 3), (8, 4), (16, 4), (5, 5), (6, 1)]


def budget(anchor, L, K):
    return min((anchor // K + 1) * K, L)


class StopDistributionTests(unittest.TestCase):
    def test_stop_is_the_end_of_a_uniform_tokens_block(self):
        for L, K in SHAPES:
            for s in range(50):
                rng, ref = random.Random(s), random.Random(s)
                self.assertEqual(stop_position(L, K, rng), budget(ref.randrange(L), L, K))

    def test_keep_probability_matches_the_stop_distribution(self):
        for L, K in SHAPES:
            for t in range(L):
                kept = sum(budget(a, L, K) > t for a in range(L))
                self.assertEqual(Fraction(kept, L), Fraction(L - (t // K) * K, L))
                self.assertAlmostEqual(keep_probability(t, L, K), kept / L)

    def test_weights_are_inverse_keep_probabilities_bounded_by_L_over_K(self):
        L, K = 16384, 128
        w = ipw_weights(L, L, K, dtype=torch.float64)
        self.assertEqual(w[0].item(), 1.0)
        self.assertEqual(w.max().item(), L / K)
        for t in (0, 127, 128, 9000, L - 1):
            self.assertAlmostEqual(w[t].item(), 1 / keep_probability(t, L, K))

    def test_invalid_shapes_rejected(self):
        for args in [(0, 4, 2), (5, 4, 2), (3, 4, 0), (3, 4, 5)]:
            with self.assertRaises(ValueError):
                ipw_weights(*args)
        with self.assertRaises(ValueError):
            stop_position(4, 5, random.Random(0))


class UnbiasednessTests(unittest.TestCase):
    def test_weighted_gradient_sum_is_exactly_unbiased(self):
        # E over the stop of grad sum_{t < G} h_t / p_t  ==  grad sum_{t < T} h_t, for every full length T.
        for L, K in SHAPES:
            for T in range(1, L + 1):
                x = torch.arange(1, L + 1, dtype=torch.float64, requires_grad=True)
                h = x.square() * torch.tensor([(-1) ** t for t in range(L)], dtype=torch.float64)
                total = 0
                for anchor in range(L):
                    G = min(T, budget(anchor, L, K))
                    total = total + (h[:G] * ipw_weights(G, L, K, dtype=x.dtype)).sum() / L
                actual = torch.autograd.grad(total, x, retain_graph=True)[0]
                expected = torch.autograd.grad(h[:T].sum(), x)[0]
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_n_hat_is_unbiased_for_the_full_token_count(self):
        for L, K in SHAPES:
            for T in range(1, L + 1):
                total = Fraction(0)
                for anchor in range(L):
                    G = min(T, budget(anchor, L, K))
                    total += sum(Fraction(L, L - (t // K) * K) for t in range(G))
                self.assertEqual(total / L, T)

    def test_n_hat_and_max_weight_agree_with_the_weights(self):
        L, K, lengths = 16, 4, [16, 5, 9, 1]
        w = ipw_weights(L, L, K, dtype=torch.float64)
        self.assertAlmostEqual(estimated_full_tokens(lengths, L, K), float(sum(w[:g].sum() for g in lengths)))
        self.assertEqual(max_weight(lengths, L, K), w[15].item())
        self.assertEqual(max_weight([5], L, K), w[4].item())


class SeedTests(unittest.TestCase):
    def test_seeds_are_deterministic_and_separate_by_use(self):
        a = request_seed("0", 26, 3, 0, "block")
        self.assertEqual(a, request_seed("0", 26, 3, 0, "block"))
        self.assertNotEqual(a, request_seed("0", 26, 3, 0, "rollout"))
        self.assertNotEqual(a, request_seed("0", 26, 4, 0, "block"))
        self.assertNotEqual(a, request_seed("1", 26, 3, 0, "block"))
        self.assertLess(a, 2 ** 32)


class NormalizerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name) / "nhat.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_first_step_uses_the_warmup_count_then_the_previous_n_hat(self):
        self.assertEqual(normalizer.denominator(26, 26, 247011, self.state), 247011)
        normalizer.record(26, 300000.0, self.state)
        self.assertEqual(normalizer.denominator(27, 26, 247011, self.state), 300000.0)
        with self.assertRaises(RuntimeError):
            normalizer.denominator(28, 26, 247011, self.state)   # step 27 never recorded: refuse to guess
        with self.assertRaises(ValueError):
            normalizer.denominator(25, 26, 247011, self.state)

    def test_resume_reads_the_exact_previous_step(self):
        for step, value in [(26, 1e5), (27, 2e5), (28, 3e5), (29, 4e5)]:
            normalizer.record(step, value, self.state)
        self.assertEqual(normalizer.denominator(28, 26, 1.0, self.state), 2e5)   # resumed from step 27
        normalizer.record(28, 3.5e5, self.state)                                   # rerun of step 28
        self.assertEqual(normalizer.denominator(29, 26, 1.0, self.state), 3.5e5)

    def test_invalid_values_rejected(self):
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                normalizer.denominator(26, 26, value, self.state)
            with self.assertRaises(ValueError):
                normalizer.record(26, value, self.state)


if __name__ == "__main__":
    unittest.main()
