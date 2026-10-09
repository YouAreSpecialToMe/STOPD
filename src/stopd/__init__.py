"""STOPD: Stochastic Truncated On-Policy Distillation."""

from stopd.core import STOPD, all_reduce_sum, response_positions, token_weights
from stopd.truncation import estimated_full_tokens, ipw_weights, keep_probability, stop_position

__all__ = ["STOPD", "all_reduce_sum", "estimated_full_tokens", "ipw_weights", "keep_probability",
           "response_positions", "stop_position", "token_weights"]
__version__ = "0.2.0"
