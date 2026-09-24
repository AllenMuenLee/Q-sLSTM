# Stable, platform-independent seed utilities shared by the experiment pipelines.

from __future__ import annotations

import numpy as np

MAX_SEED = 2**31 - 1


def stable_seed(*parts):
    """31-bit seed derived from the parts via SeedSequence (stable across runs and platforms)."""
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0] & 0x7FFFFFFF)


def draw_seeds(n, master_seed):
    """`n` distinct run seeds drawn reproducibly from `master_seed`.

    Seeds are generated one at a time, so the list for n is a prefix of the list for any larger n.
    """
    rng = np.random.default_rng(np.random.SeedSequence([int(master_seed), 7]))
    seeds = []
    while len(seeds) < n:
        candidate = int(rng.integers(0, MAX_SEED))
        if candidate not in seeds:
            seeds.append(candidate)
    return seeds


def epoch_permutation(loader_seed, epoch, n):
    """Deterministic training order for one epoch, independent of global RNG state."""
    rng = np.random.default_rng(np.random.SeedSequence([int(loader_seed), int(epoch)]))
    return rng.permutation(n)
