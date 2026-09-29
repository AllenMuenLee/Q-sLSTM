# Synthetic 1-D input / 1-D output sequence tasks for probing general recurrent-memory abilities.
#
# Every task maps a scalar input stream x[t] to a scalar target stream y[t] (sequence-to-sequence,
# one prediction per token). Each task isolates one ability that LSTM-type models are commonly
# tested on:
#
#     delay        y[t] = x[t - k]                          exact short-term memory (fixed lag)
#     ema          y[t] = d * y[t-1] + (1 - d) * x[t]       leaky integration / smoothing
#     running_max  y[t] = max(x[1..t])                      nonlinear memory revision (1-D nearest-neighbor analogue)
#     flip_flop    y[t] = last nonzero pulse in x[1..t]     latching: hold a bit through long silences
#     narma        NARMA-n system response to x              nonlinear system identification (reservoir benchmark)
#     sine_next    y[t] = x[t + 1], x a random-frequency sine  in-context frequency inference / forecasting
#
# Inputs lie in [-1, 1] (narma: [0, 0.5] rescaled to [-1, 1] for the model); only `inputs` may be fed
# to a model. `loss_mask` excludes the warm-up steps where the target is undefined or not yet
# inferable from the inputs seen so far (e.g. the first k steps of `delay`).
#
# Generation is deterministic, uses only local numpy generators, and materializes all tensors once.

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

TASKS = ("delay", "ema", "running_max", "flip_flop", "narma", "sine_next")
INPUT_SIZE = 1
OUTPUT_SIZE = 1

# Independent random streams so training, held-out, and extrapolation data never share draws.
STREAM_IDS = {"train_pool": 0, "test": 1, "extrapolation": 2}
STREAM_ID_OFFSETS = {"train_pool": 0, "test": 1_000_000, "extrapolation": 2_000_000}
MAX_STREAM_SEQUENCES = 1_000_000
NARMA_WASHOUT = 200    # hidden burn-in so every NARMA sequence starts from the system's stationary regime
NARMA_MAX_RETRIES = 100


@dataclass(frozen=True)
class ScalarTaskConfig:
    """Per-task generation parameters; saved with every run's metadata."""

    delay: int = 4                        # delay: lag k
    ema_decay: float = 0.8                # ema: decay d (effective memory ~ 1 / (1 - d) steps)
    flip_probability: float = 0.1         # flip_flop: probability a token is a +-1 pulse
    narma_order: int = 10                 # narma: order n (NARMA10 is the standard benchmark)
    sine_frequency_range: tuple = (0.1, 0.6)  # sine_next: angular frequency per step (rad)

    def validate(self, task, sequence_length=None):
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
        if self.delay < 1:
            raise ValueError(f"delay must be at least 1, got {self.delay}")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError(f"ema_decay must lie in (0, 1), got {self.ema_decay}")
        if not 0.0 < self.flip_probability <= 1.0:
            raise ValueError(f"flip_probability must lie in (0, 1], got {self.flip_probability}")
        if self.narma_order < 2:
            raise ValueError(f"narma_order must be at least 2, got {self.narma_order}")
        lo, hi = self.sine_frequency_range
        if not 0.0 < lo <= hi < np.pi:
            raise ValueError(f"sine_frequency_range must satisfy 0 < lo <= hi < pi, got {self.sine_frequency_range}")
        if sequence_length is not None and sequence_length <= warmup_steps(task, self):
            raise ValueError(
                f"sequence_length {sequence_length} leaves no supervised step for task {task!r} "
                f"(warm-up {warmup_steps(task, self)})")

    def to_dict(self):
        out = asdict(self)
        out["sine_frequency_range"] = list(self.sine_frequency_range)
        return out


def warmup_steps(task, config):
    """Number of leading tokens excluded from the loss (target undefined or not yet inferable)."""
    return {
        "delay": config.delay,
        "ema": 0,
        "running_max": 0,
        "flip_flop": 0,
        "narma": config.narma_order,
        "sine_next": 2,  # a frequency is identifiable only after two observed samples
    }[task]


# ---------------------------------------------------------------------------------------------
# Task generators: (rng, count, length, config) -> inputs [N, L], targets [N, L] (float64)
# ---------------------------------------------------------------------------------------------

def _delay(rng, count, length, config):
    x = rng.uniform(-1.0, 1.0, size=(count, length))
    y = np.zeros_like(x)
    y[:, config.delay:] = x[:, :-config.delay]
    return x, y


def _ema(rng, count, length, config):
    x = rng.uniform(-1.0, 1.0, size=(count, length))
    y = np.zeros_like(x)
    prev = np.zeros(count)
    for t in range(length):
        prev = config.ema_decay * prev + (1.0 - config.ema_decay) * x[:, t]
        y[:, t] = prev
    # Rescale so the stationary target std matches the input std (sqrt((1-d)/(1+d)) shrinkage).
    return x, y * np.sqrt((1.0 + config.ema_decay) / (1.0 - config.ema_decay))


def _running_max(rng, count, length, config):
    x = rng.uniform(-1.0, 1.0, size=(count, length))
    return x, np.maximum.accumulate(x, axis=1)


def _flip_flop(rng, count, length, config):
    pulse = rng.random(size=(count, length)) < config.flip_probability
    x = np.where(pulse, rng.choice([-1.0, 1.0], size=(count, length)), 0.0)
    y = np.zeros_like(x)
    state = np.zeros(count)
    for t in range(length):
        state = np.where(pulse[:, t], x[:, t], state)
        y[:, t] = state
    return x, y


def _narma_sequence(rng, length, order):
    """One stable NARMA-n trajectory (Atiya & Parlos form), input u ~ U(0, 0.5)."""
    total = NARMA_WASHOUT + length
    for _ in range(NARMA_MAX_RETRIES):
        u = rng.uniform(0.0, 0.5, size=total)
        y = np.zeros(total)
        for t in range(order, total):
            y[t] = (0.3 * y[t - 1] + 0.05 * y[t - 1] * y[t - order:t].sum()
                    + 1.5 * u[t - order] * u[t - 1] + 0.1)
            if not np.isfinite(y[t]) or abs(y[t]) > 1e3:
                break
        else:
            return u[NARMA_WASHOUT:], y[NARMA_WASHOUT:]
    raise RuntimeError(f"NARMA-{order} diverged {NARMA_MAX_RETRIES} times in a row")


def _narma(rng, count, length, config):
    pairs = [_narma_sequence(rng, length, config.narma_order) for _ in range(count)]
    u = np.stack([p[0] for p in pairs])
    y = np.stack([p[1] for p in pairs])
    return 4.0 * u - 1.0, y  # model sees u rescaled to [-1, 1]; target in NARMA's native scale


def _sine_next(rng, count, length, config):
    omega = rng.uniform(*config.sine_frequency_range, size=(count, 1))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(count, 1))
    t = np.arange(length + 1)[None, :]
    wave = np.sin(omega * t + phase)
    return wave[:, :-1], wave[:, 1:]


GENERATORS = {
    "delay": _delay, "ema": _ema, "running_max": _running_max,
    "flip_flop": _flip_flop, "narma": _narma, "sine_next": _sine_next,
}


# ---------------------------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------------------------

TENSOR_FIELDS = ("inputs", "targets", "loss_mask")


def _digest(*chunks):
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk)
    return h.hexdigest()


class ScalarTaskDataset(Dataset):
    """Materialized sequences; items are dicts with `inputs`, `targets`, `loss_mask`, `sequence_id`."""

    def __init__(self, tensors, sequence_ids, task, metadata=None):
        self.tensors = tensors
        self.sequence_ids = torch.as_tensor(sequence_ids, dtype=torch.long)
        self.task = task
        self.metadata = dict(metadata or {})
        n = self.sequence_ids.shape[0]
        if any(v.shape[0] != n for v in tensors.values()):
            raise ValueError("all dataset fields must have the same number of sequences")

    @property
    def sequence_length(self):
        return self.tensors["inputs"].shape[1]

    def __len__(self):
        return self.sequence_ids.shape[0]

    def __getitem__(self, index):
        item = {name: self.tensors[name][index] for name in TENSOR_FIELDS}
        item["sequence_id"] = int(self.sequence_ids[index])
        return item

    def select(self, indices):
        indices = torch.as_tensor(np.asarray(indices), dtype=torch.long)
        return ScalarTaskDataset({k: v[indices] for k, v in self.tensors.items()},
                                 self.sequence_ids[indices], self.task, self.metadata)

    def checksums(self):
        """Stable sha256 digests of the materialized tensors and ids."""
        out = {}
        for name in TENSOR_FIELDS:
            t = self.tensors[name].contiguous().cpu()
            out[name] = _digest(name.encode(), str(t.dtype).encode(), str(tuple(t.shape)).encode(),
                                t.numpy().tobytes())
        out["sequence_ids"] = _digest(self.sequence_ids.numpy().astype("<i8").tobytes())
        out["combined"] = _digest(*(out[k].encode() for k in sorted(out)))
        return out


def generate_dataset(task, count, sequence_length, seed, stream="train_pool", config=None):
    """Deterministic dataset of `count` sequences of one task.

    Each (seed, stream, task) triple owns an independent numpy generator, so streams never overlap
    and no global RNG state is touched. Sequence ids start at the stream's offset.
    """
    config = config or ScalarTaskConfig()
    config.validate(task, sequence_length)
    if stream not in STREAM_IDS:
        raise ValueError(f"stream must be one of {tuple(STREAM_IDS)}, got {stream!r}")
    if not 0 < count < MAX_STREAM_SEQUENCES:
        raise ValueError(f"count must lie in (0, {MAX_STREAM_SEQUENCES}), got {count}")

    rng = np.random.default_rng(np.random.SeedSequence([int(seed), STREAM_IDS[stream], TASKS.index(task)]))
    x, y = GENERATORS[task](rng, count, sequence_length, config)
    mask = np.zeros((count, sequence_length), dtype=bool)
    mask[:, warmup_steps(task, config):] = True

    tensors = {
        "inputs": torch.from_numpy(x.astype(np.float32))[..., None],
        "targets": torch.from_numpy(y.astype(np.float32))[..., None],
        "loss_mask": torch.from_numpy(mask)[..., None],
    }
    for name in ("inputs", "targets"):
        if not torch.isfinite(tensors[name]).all():
            raise ValueError(f"non-finite {name} generated for task {task!r}")
    metadata = {"task": task, "stream": stream, "seed": int(seed), "sequence_length": sequence_length,
                "warmup_steps": warmup_steps(task, config), "config": config.to_dict()}
    ids = STREAM_ID_OFFSETS[stream] + np.arange(count)
    return ScalarTaskDataset(tensors, ids, task, metadata)


def split_train_val(dataset, val_size, seed):
    """Deterministic split of the training pool; returns (train_ds, val_ds, train_idx, val_idx)."""
    n = len(dataset)
    if not 0 < val_size < n:
        raise ValueError(f"val_size must lie in (0, {n}), got {val_size}")
    perm = np.random.default_rng(np.random.SeedSequence([int(seed), 99])).permutation(n)
    val_idx, train_idx = np.sort(perm[:val_size]), np.sort(perm[val_size:])
    return dataset.select(train_idx), dataset.select(val_idx), train_idx, val_idx
