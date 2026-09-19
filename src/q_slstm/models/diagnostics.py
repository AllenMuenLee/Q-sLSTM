# Analysis-only write-proportion diagnostic shared by the QLSTM baseline and the Q-sLSTM cell.
#
# alpha_t is the fraction of post-update (normalized) memory mass that belongs to the current write:
#     n_after_t = f_t * n_before_t + i_t
#     alpha_t   = i_t / clamp_min(n_after_t, epsilon)
# It is elementwise (batch item x hidden coordinate); reduce it only afterwards. It never feeds back
# into a model's state, output, loss, or gradients: callers pass detached tensors.

import torch

DEFAULT_ALPHA_EPSILON = 1e-6


def write_proportion(i_t, f_t, n_before, epsilon=DEFAULT_ALPHA_EPSILON):
    """Return (alpha_t, n_after_t) computed elementwise from detached inputs."""
    i_t, f_t, n_before = i_t.detach(), f_t.detach(), n_before.detach()
    floor = max(epsilon, torch.finfo(n_before.dtype).eps)
    n_after = f_t * n_before + i_t
    alpha = i_t / torch.clamp_min(n_after, floor)
    return alpha, n_after
