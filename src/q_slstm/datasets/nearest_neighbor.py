# Synthetic nearest-neighbor memory-revision task (adapted from the xLSTM nearest-neighbor search
# experiment).
#
# A sequence is one 2-D unit reference vector followed by candidate unit vectors carrying scalar
# values in [0, 1]. At every candidate the target is the value attached to the earliest candidate
# with the largest similarity to the reference so far, so a model must revise its memory when a
# strictly better candidate arrives and retain it through later, worse candidates.
#
# Token layout for a sequence of L total tokens (L - 1 candidates):
#     x[0] = [r_x, r_y, -1.0]   reference token (the -1 sentinel is outside the value range)
#     x[t] = [u_x, u_y, v]      candidate t = 1, ..., L - 1
# Only `inputs` may be fed to a model; every other field is for supervision masks and analysis.
#
# Generation is deterministic, uses only local numpy generators, and materializes all tensors once.

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

CASE_TYPES = ("iid", "late", "early", "near_best")
INPUT_SIZE = 3
OUTPUT_SIZE = 1
REFERENCE_SENTINEL = -1.0
MIN_STRESS_SEQUENCE_LENGTH = 5

# Independent random streams so training, held-out, and extrapolation data never share draws.
STREAM_IDS = {"train_pool": 0, "test": 1, "extrapolation": 2}
STREAM_ID_OFFSETS = {"train_pool": 0, "test": 1_000_000, "extrapolation": 2_000_000}
MAX_STREAM_SEQUENCES = 1_000_000
BAND_TOLERANCE = 1e-4  # float32 slack when deciding band membership of generated similarities


@dataclass(frozen=True)
class NearestNeighborConfig:
    """Generation thresholds; saved with every run's metadata."""

    record_margin: float = 0.02       # non-best candidates stay at least this far below the best
    near_best_delta: float = 0.05     # near-best band is [s_best - delta, s_best - margin]
    value_separation: float = 0.4     # forced events / near-best distractors differ from the target
    early_fraction: float = 0.25      # early range = first fraction of candidates
    late_fraction: float = 0.25       # late range = last fraction of candidates
    best_similarity_range: tuple = (0.90, 0.995)  # global-best similarity of the stress cases

    def validate(self, sequence_length=None):
        if not self.record_margin > 0:
            raise ValueError(f"record_margin must be positive, got {self.record_margin}")
        if not self.near_best_delta > self.record_margin:
            raise ValueError(
                f"near_best_delta ({self.near_best_delta}) must exceed record_margin "
                f"({self.record_margin}) so the near-best band is non-empty"
            )
        if not 0.0 <= self.value_separation <= 0.5:
            raise ValueError(
                f"value_separation must lie in [0, 0.5] so a separated value exists for every target "
                f"in [0, 1], got {self.value_separation}"
            )
        for name in ("early_fraction", "late_fraction"):
            if not 0.0 < getattr(self, name) <= 0.5:
                raise ValueError(f"{name} must lie in (0, 0.5], got {getattr(self, name)}")
        lo, hi = self.best_similarity_range
        if not (-1.0 < lo <= hi < 1.0 + 1e-12) or lo - self.near_best_delta <= -1.0:
            raise ValueError(f"best_similarity_range {self.best_similarity_range} is infeasible with the margins")
        if sequence_length is not None and sequence_length < 3:
            raise ValueError(f"sequence_length must be at least 3 (reference + 2 candidates), got {sequence_length}")

    def to_dict(self):
        out = asdict(self)
        out["best_similarity_range"] = list(self.best_similarity_range)
        return out


# ---------------------------------------------------------------------------------------------
# Geometry and labels
# ---------------------------------------------------------------------------------------------

def unit_vector(theta):
    return np.stack([np.cos(theta), np.sin(theta)], axis=-1)


def vector_from_similarity(reference, similarity, sign):
    """Unit vector u with dot(reference, u) == similarity: u = s*r + sign*sqrt(1-s^2)*r_perp."""
    reference = np.asarray(reference, dtype=np.float64)
    similarity = np.asarray(similarity, dtype=np.float64)
    perp = np.stack([-reference[..., 1], reference[..., 0]], axis=-1)
    u = similarity[..., None] * reference + (
        np.asarray(sign, dtype=np.float64) * np.sqrt(np.maximum(0.0, 1.0 - similarity**2))
    )[..., None] * perp
    return u / np.linalg.norm(u, axis=-1, keepdims=True)  # only removes floating-point drift


def compute_labels(reference, candidates, values):
    """The single target-generation function; every dataset label comes from here.

    reference [N, 2], candidates [N, n, 2], values [N, n]. Returns numpy arrays (float64
    similarities) for candidates 1..n:
      similarities   dot(r, u_t)
      running_best   running maximum similarity through candidate t
      events         candidate t is a strict new record (the first candidate always is)
      best_index     candidate number (1-based) of the earliest candidate attaining the running maximum
      targets        value attached to that candidate
    A later candidate with equal similarity is not a record, so ties resolve to the earliest one.
    """
    reference = np.asarray(reference, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    sims = np.einsum("nd,nkd->nk", reference, candidates)
    running_best = np.maximum.accumulate(sims, axis=1)
    events = np.ones_like(sims, dtype=bool)
    events[:, 1:] = sims[:, 1:] > running_best[:, :-1]
    positions = np.arange(1, sims.shape[1] + 1)[None, :]
    best_index = np.maximum.accumulate(np.where(events, positions, 0), axis=1)
    targets = np.take_along_axis(values, best_index - 1, axis=1)
    return {
        "similarities": sims,
        "running_best": running_best,
        "events": events,
        "best_index": best_index,
        "targets": targets,
    }


def early_range(n_candidates, config):
    """Inclusive candidate-index range (1-based) of the first quarter."""
    return 1, max(1, math.ceil(config.early_fraction * n_candidates))


def late_range(n_candidates, config):
    """Inclusive candidate-index range (1-based) of the final quarter."""
    return n_candidates - max(1, math.ceil(config.late_fraction * n_candidates)) + 1, n_candidates


def _late_best_range(n_candidates, config):
    """Late range with the final token excluded (leaving a later distractor) whenever possible."""
    lo, hi = late_range(n_candidates, config)
    return (lo, hi - 1) if hi - 1 >= lo else (lo, hi)


# ---------------------------------------------------------------------------------------------
# Sequence generators
# ---------------------------------------------------------------------------------------------

def _sample_below(rng, cap, size):
    """Similarities <= cap with uniformly distributed angle: bounded conditional sampling."""
    a = math.acos(min(1.0, max(-1.0, cap)))
    return np.cos(rng.uniform(a, 2.0 * math.pi - a, size=size))


def _sample_separated(rng, previous, separation):
    """Uniform value in [0, 1] with |value - previous| >= separation (direct, no retries)."""
    below = max(previous - separation, 0.0)
    above = max(1.0 - (previous + separation), 0.0)
    total = below + above
    if total <= 0.0:
        raise ValueError(f"no value in [0, 1] is {separation} away from target {previous}")
    x = rng.uniform(0.0, total)
    return x if x < below else (previous + separation) + (x - below)


def _generate_iid(rng, count, sequence_length):
    n = sequence_length - 1
    r = unit_vector(rng.uniform(0.0, 2.0 * math.pi, size=count))
    u = unit_vector(rng.uniform(0.0, 2.0 * math.pi, size=(count, n)))
    v = rng.uniform(0.0, 1.0, size=(count, n))
    return r, u, v


def _generate_stress_sequence(rng, case, sequence_length, config):
    n = sequence_length - 1
    r = unit_vector(rng.uniform(0.0, 2.0 * math.pi))
    lo, hi = _late_best_range(n, config) if case == "late" else early_range(n, config)
    best = int(rng.integers(lo, hi + 1))
    s_best = float(rng.uniform(*config.best_similarity_range))
    cap = s_best - config.record_margin

    sims = _sample_below(rng, cap, n)
    sims[best - 1] = s_best
    if case == "near_best":
        later = np.arange(best + 1, n + 1)
        if len(later):
            chosen = rng.choice(later, size=math.ceil(len(later) / 2), replace=False)
            sims[chosen - 1] = rng.uniform(s_best - config.near_best_delta, cap, size=len(chosen))

    signs = rng.choice([-1.0, 1.0], size=n)
    u = vector_from_similarity(np.broadcast_to(r, (n, 2)), sims, signs)
    # Decide values from exactly the float32 geometry that the dataset stores.
    r32, u32 = r.astype(np.float32).astype(np.float64), u.astype(np.float32).astype(np.float64)
    sims32 = u32 @ r32

    # Any later candidate inside the near-best band (forced or by chance) gets a separated value.
    near_band = np.zeros(n, dtype=bool)
    if case == "near_best":
        near_band = ((sims32 >= s_best - config.near_best_delta - BAND_TOLERANCE)
                     & (sims32 <= cap + BAND_TOLERANCE) & (np.arange(n) >= best))

    values = rng.uniform(0.0, 1.0, size=n)
    best_sim, target = -np.inf, None
    for t in range(n):
        is_record = sims32[t] > best_sim
        if (is_record and t > 0) or near_band[t]:
            values[t] = _sample_separated(rng, target, config.value_separation)
        if is_record:
            best_sim, target = sims32[t], values[t]
    return r, u, values


def generate_case(rng, case, count, sequence_length, config):
    """`count` sequences of one case type as float64 arrays (reference, candidates, values)."""
    if case not in CASE_TYPES:
        raise ValueError(f"unknown case type {case!r}; expected one of {CASE_TYPES}")
    if count == 0:
        n = sequence_length - 1
        return np.zeros((0, 2)), np.zeros((0, n, 2)), np.zeros((0, n))
    if case == "iid":
        return _generate_iid(rng, count, sequence_length)
    if sequence_length < MIN_STRESS_SEQUENCE_LENGTH:
        raise ValueError(
            f"stress case {case!r} needs sequence_length >= {MIN_STRESS_SEQUENCE_LENGTH}, got {sequence_length}"
        )
    parts = [_generate_stress_sequence(rng, case, sequence_length, config) for _ in range(count)]
    return (
        np.stack([p[0] for p in parts]),
        np.stack([p[1] for p in parts]),
        np.stack([p[2] for p in parts]),
    )


# ---------------------------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------------------------

TENSOR_FIELDS = (
    "inputs", "targets", "loss_mask", "metric_mask", "event_mask", "similarities", "best_indices",
)


def _digest(*chunks):
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk)
    return h.hexdigest()


class NearestNeighborDataset(Dataset):
    """Materialized sequences; items are dicts (see module header for the fields)."""

    def __init__(self, tensors, case_types, sequence_ids, metadata=None):
        self.tensors = tensors
        self.case_types = list(case_types)
        self.sequence_ids = torch.as_tensor(sequence_ids, dtype=torch.long)
        self.metadata = dict(metadata or {})
        n = len(self.case_types)
        if any(v.shape[0] != n for v in tensors.values()) or self.sequence_ids.shape[0] != n:
            raise ValueError("all dataset fields must have the same number of sequences")

    @property
    def sequence_length(self):
        return self.tensors["inputs"].shape[1]

    def __len__(self):
        return len(self.case_types)

    def __getitem__(self, index):
        item = {name: self.tensors[name][index] for name in TENSOR_FIELDS}
        item["case_type"] = self.case_types[index]
        item["sequence_id"] = int(self.sequence_ids[index])
        return item

    def select(self, indices):
        indices = torch.as_tensor(np.asarray(indices), dtype=torch.long)
        return NearestNeighborDataset(
            {k: v[indices] for k, v in self.tensors.items()},
            [self.case_types[i] for i in indices.tolist()],
            self.sequence_ids[indices],
            self.metadata,
        )

    def case_counts(self):
        return {case: sum(c == case for c in self.case_types) for case in CASE_TYPES}

    def checksums(self):
        """Stable sha256 digests of the materialized tensors, ids, and case labels."""
        out = {}
        for name in TENSOR_FIELDS:
            t = self.tensors[name].contiguous().cpu()
            out[name] = _digest(name.encode(), str(t.dtype).encode(), str(tuple(t.shape)).encode(),
                                t.numpy().tobytes())
        out["sequence_ids"] = _digest(self.sequence_ids.numpy().astype("<i8").tobytes())
        out["case_types"] = _digest("\n".join(self.case_types).encode())
        out["combined"] = _digest(*(out[k].encode() for k in sorted(out)))
        return out


def assemble_dataset(reference, candidates, values, case_types, sequence_ids, metadata=None):
    """Build a dataset from generated geometry; labels come only from `compute_labels`."""
    reference = np.asarray(reference, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    n_seq, n_cand = values.shape
    length = n_cand + 1

    labels = compute_labels(reference, candidates, values)  # float32 inputs, upcast for the labels

    inputs = np.zeros((n_seq, length, INPUT_SIZE), dtype=np.float32)
    inputs[:, 0, :2] = reference
    inputs[:, 0, 2] = REFERENCE_SENTINEL
    inputs[:, 1:, :2] = candidates
    inputs[:, 1:, 2] = values

    def per_token(candidate_array, dtype):
        out = np.zeros((n_seq, length), dtype=dtype)
        out[:, 1:] = candidate_array
        return out

    loss_mask = np.zeros((n_seq, length), dtype=bool)
    loss_mask[:, 1:] = True
    metric_mask = np.zeros((n_seq, length), dtype=bool)
    metric_mask[:, 2:] = True
    best_indices = np.full((n_seq, length), -1, dtype=np.int64)
    best_indices[:, 1:] = labels["best_index"]

    tensors = {
        "inputs": torch.from_numpy(inputs),
        "targets": torch.from_numpy(per_token(labels["targets"], np.float32))[..., None],
        "loss_mask": torch.from_numpy(loss_mask)[..., None],
        "metric_mask": torch.from_numpy(metric_mask)[..., None],
        "event_mask": torch.from_numpy(per_token(labels["events"], bool))[..., None],
        "similarities": torch.from_numpy(per_token(labels["similarities"], np.float32))[..., None],
        "best_indices": torch.from_numpy(best_indices),
    }
    return NearestNeighborDataset(tensors, case_types, sequence_ids, metadata)


def balanced_case_counts(total):
    """Split `total` sequences over the four case types; the remainder goes to earlier cases first."""
    base, remainder = divmod(total, len(CASE_TYPES))
    return {case: base + (1 if i < remainder else 0) for i, case in enumerate(CASE_TYPES)}


def generate_dataset(case_counts, sequence_length, seed, stream="train_pool", config=None):
    """Deterministic dataset of `case_counts[case]` sequences per case, ordered by case type.

    Each (seed, stream, case) pair owns an independent numpy generator, so streams never overlap
    and no global RNG state is touched. Sequence ids start at the stream's offset.
    """
    config = config or NearestNeighborConfig()
    config.validate(sequence_length)
    if stream not in STREAM_IDS:
        raise ValueError(f"stream must be one of {tuple(STREAM_IDS)}, got {stream!r}")
    counts = {case: int(case_counts.get(case, 0)) for case in CASE_TYPES}
    total = sum(counts.values())
    if total >= MAX_STREAM_SEQUENCES:
        raise ValueError(f"at most {MAX_STREAM_SEQUENCES - 1} sequences per stream, got {total}")

    refs, cands, vals, case_types = [], [], [], []
    for case_index, case in enumerate(CASE_TYPES):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), STREAM_IDS[stream], case_index]))
        r, u, v = generate_case(rng, case, counts[case], sequence_length, config)
        refs.append(r), cands.append(u), vals.append(v)
        case_types += [case] * counts[case]

    ids = STREAM_ID_OFFSETS[stream] + np.arange(total)
    metadata = {
        "stream": stream,
        "seed": int(seed),
        "sequence_length": sequence_length,
        "n_candidates": sequence_length - 1,
        "case_counts": counts,
        "config": config.to_dict(),
    }
    return assemble_dataset(np.concatenate(refs), np.concatenate(cands), np.concatenate(vals),
                            case_types, ids, metadata)


def make_train_pool(size, sequence_length, seed, config=None):
    """IID training pool (validation is split off separately)."""
    return generate_dataset({"iid": size}, sequence_length, seed, "train_pool", config)


def make_balanced_suite(total, sequence_length, seed, stream="test", config=None):
    """Held-out (or extrapolation) suite balanced over the four case types."""
    return generate_dataset(balanced_case_counts(total), sequence_length, seed, stream, config)


def split_train_val(dataset, val_size, seed):
    """Deterministic split of the training pool; returns (train_ds, val_ds, train_idx, val_idx)."""
    n = len(dataset)
    if not 0 < val_size < n:
        raise ValueError(f"val_size must lie in (0, {n}), got {val_size}")
    perm = np.random.default_rng(np.random.SeedSequence([int(seed), 99])).permutation(n)
    val_idx, train_idx = np.sort(perm[:val_size]), np.sort(perm[val_size:])
    return dataset.select(train_idx), dataset.select(val_idx), train_idx, val_idx
