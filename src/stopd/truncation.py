"""Random rollout truncation with blockwise inverse-probability weighting (IPW).

Positions are zero-based. Rollouts have a maximum length of L tokens, split
into blocks of width K. Before generation, STOPD draws a token position A
uniformly from {0, ..., L-1} and lets the student generate until the end of
A's block, min((A // K + 1) * K, L) tokens, or until it stops on its own.

Token t therefore survives with probability

    p_t = (L - (t // K) * K) / L,

and weighting every generated token by 1 / p_t makes the weighted sum of
token terms unbiased for the sum over the complete rollout (tests check this
exactly). Rounding the stop up to a block end bounds the largest weight by L / K.

This module needs no ML runtime except torch for `ipw_weights`.
"""

import hashlib


def stop_position(max_len, block, rng):
    """Token budget of one rollout: the end of the block holding a uniform token position."""
    if not 1 <= block <= max_len:
        raise ValueError("require 1 <= block width <= max response length")
    anchor = rng.randrange(max_len)
    return min((anchor // block + 1) * block, max_len)


def keep_probability(t, max_len, block):
    """Probability that zero-based token position t is generated before the random stop."""
    if not 0 <= t < max_len:
        raise ValueError("position outside the max response length")
    return (max_len - (t // block) * block) / max_len


def ipw_weights(size, max_len, block, *, device=None, dtype=None):
    """1 / p_t for positions 0..size-1, as a torch tensor."""
    import torch

    if not 0 < size <= max_len or not 0 < block <= max_len:
        raise ValueError("invalid response size, max response length, or block width")
    pos = torch.arange(size, device=device)
    return max_len / (max_len - (pos // block) * block).to(dtype or torch.float32)


def estimated_full_tokens(lengths, max_len, block):
    """N_hat: the IPW-weighted count of the generated tokens.

    Unbiased for the number of tokens the complete rollouts would have had; the
    lagged normalizer uses a step's N_hat as the next step's denominator.
    """
    total = 0.0
    for length in lengths:
        for start in range(0, int(length), block):
            total += min(block, int(length) - start) * max_len / (max_len - start)
    return total


def max_weight(lengths, max_len, block):
    """Largest IPW weight among the generated tokens."""
    longest = max(int(n) for n in lengths)
    return max_len / (max_len - ((longest - 1) // block) * block)


def request_seed(seed, step, sample, rollout, purpose):
    """Deterministic 32-bit seed for one rollout and one use (the stop, or the rollout's own sampling).

    Vanilla and STOPD runs with the same seed sample every rollout with the same
    vLLM seed, so the two runs differ only by where each rollout is cut.
    """
    value = ":".join(map(str, (seed, step, sample, rollout, purpose)))
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "little")
