"""Lagged loss normalizer C_k.

Vanilla OPD divides a step's summed token losses by the number of generated
tokens. Under random truncation that count depends on the step's own stops,
so STOPD divides by C_k, fixed before step k's stops are drawn:

    C_{W+1} = N_W, the exact token count of the last Vanilla warmup step W;
    C_k     = N_hat_{k-1}, the previous step's IPW estimate of its full token count.

The estimates are kept in a small JSON file keyed by step, so a resumed run
reads the exact previous step and a rerun step overwrites its stale entry.
"""

import json
import math
import os
from pathlib import Path


def _check(value, what):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid {what}: {value}")
    return float(value)


def history(state_path):
    """Recorded N_hat by training step."""
    path = Path(state_path)
    if not path.exists():
        return {}
    return {int(k): float(v) for k, v in json.loads(path.read_text())["n_hat"].items()}


def denominator(step, first_step, initial, state_path):
    """C for training step `step` (first_step = W + 1, initial = N_W)."""
    if step < first_step:
        raise ValueError(f"step {step} precedes the first STOPD step {first_step}")
    if step == first_step:
        return _check(initial, "initial denominator")
    past = history(state_path)
    if step - 1 not in past:
        raise RuntimeError(f"no recorded N_hat for step {step - 1}; refusing to guess C for step {step}")
    return _check(past[step - 1], f"recorded N_hat for step {step - 1}")


def record(step, n_hat, state_path):
    """Store N_hat for `step` (the denominator of step + 1), atomically."""
    _check(n_hat, f"N_hat for step {step}")
    path = Path(state_path)
    state = json.loads(path.read_text()) if path.exists() else dict(schema=1, n_hat={})
    state["n_hat"][str(int(step))] = float(n_hat)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=1) + "\n")
    os.replace(tmp, path)
