"""Nearest-neighbor task: geometry, labels, masks, stress-case invariants, reproducibility."""

import math

import numpy as np
import pytest
import torch

from q_slstm.datasets.nearest_neighbor import (
    CASE_TYPES,
    NearestNeighborConfig,
    balanced_case_counts,
    compute_labels,
    early_range,
    generate_dataset,
    late_range,
    make_balanced_suite,
    make_train_pool,
    split_train_val,
)

L = 32
N_CAND = L - 1
CFG = NearestNeighborConfig()
TOL = 1e-5


@pytest.fixture(scope="module")
def suite():
    return make_balanced_suite(200, L, seed=3)


def by_case(ds, case):
    return ds.select([i for i, c in enumerate(ds.case_types) if c == case])


def cand(ds, name):
    """Candidate-token slice (tokens 1..L-1) of a tensor field as float64 numpy."""
    t = ds.tensors[name]
    t = t[..., 0] if t.ndim == 3 and t.shape[-1] == 1 else t
    return t[:, 1:].numpy().astype(np.float64) if t.dtype != torch.bool else t[:, 1:].numpy()


def test_reference_and_candidate_vectors_have_unit_norm(suite):
    x = suite.tensors["inputs"].double()
    assert torch.allclose(x[:, :, :2].norm(dim=-1), torch.ones(x.shape[:2], dtype=torch.float64), atol=TOL)


def test_values_and_targets_stay_in_unit_interval(suite):
    values = suite.tensors["inputs"][:, 1:, 2]
    targets = suite.tensors["targets"][:, 1:, 0]
    assert values.min() >= 0 and values.max() <= 1
    assert targets.min() >= 0 and targets.max() <= 1


def test_inputs_have_exactly_three_features_without_privileged_channels(suite):
    x = suite.tensors["inputs"]
    assert x.shape == (len(suite), L, 3)
    assert torch.all(x[:, 0, 2] == -1.0)  # sentinel only on the reference token
    # third feature of candidates is the attached value, not a copied target/event/similarity channel
    assert torch.equal(x[:, 1:, 2], x[:, 1:, 2].clamp(0, 1))
    item = suite[0]
    assert {"inputs", "targets", "loss_mask", "metric_mask", "event_mask", "similarities", "best_indices",
            "case_type", "sequence_id"} <= set(item)
    assert item["inputs"].shape == (L, 3) and item["targets"].shape == (L, 1)
    assert item["best_indices"].shape == (L,) and item["best_indices"][0] == -1


def test_similarities_match_direct_dot_products(suite):
    x = suite.tensors["inputs"].double()
    direct = (x[:, :1, :2] * x[:, 1:, :2]).sum(-1)
    stored = suite.tensors["similarities"][:, 1:, 0].double()
    assert torch.allclose(direct, stored, atol=1e-6)


def test_events_are_strict_records_with_first_candidate_and_ties():
    r = np.array([[1.0, 0.0]])
    root = math.sqrt(0.75)
    # similarities: 0.5, 0.5 (tie), 0.25, 0.75, 0.75 (tie with the running best)
    u = np.array([[[0.5, root], [0.5, -root], [0.25, math.sqrt(1 - 0.0625)],
                   [0.75, math.sqrt(1 - 0.5625)], [0.75, -math.sqrt(1 - 0.5625)]]])
    v = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
    labels = compute_labels(r, u, v)
    assert labels["events"][0].tolist() == [True, False, False, True, False]
    assert labels["best_index"][0].tolist() == [1, 1, 1, 4, 4]
    assert labels["targets"][0].tolist() == [0.1, 0.1, 0.1, 0.4, 0.4]


def test_targets_equal_values_gathered_from_running_argmax(suite):
    x = suite.tensors["inputs"]
    values = x[:, 1:, 2]
    sims = suite.tensors["similarities"][:, 1:, 0].double()
    for i in range(len(suite)):
        for t in range(N_CAND):
            j = int(torch.argmax(sims[i, : t + 1]))  # torch.argmax returns the first maximum on ties
            assert suite.tensors["best_indices"][i, t + 1] == j + 1
            assert suite.tensors["targets"][i, t + 1, 0] == values[i, j]


def test_masks_exclude_exactly_the_expected_tokens(suite):
    loss, metric = suite.tensors["loss_mask"][..., 0], suite.tensors["metric_mask"][..., 0]
    assert not loss[:, 0].any() and loss[:, 1:].all()
    assert not metric[:, :2].any() and metric[:, 2:].all()
    assert suite.tensors["event_mask"][:, 1, 0].all()  # first candidate is an automatic event


@pytest.mark.parametrize("case", ["late", "early", "near_best"])
def test_stress_generators_satisfy_invariants(case):
    ds = generate_dataset({case: 150}, L, seed=11)
    sims = cand(ds, "similarities")
    events = cand(ds, "event_mask")
    targets = cand(ds, "targets")
    values = ds.tensors["inputs"][:, 1:, 2].numpy().astype(np.float64)
    final_best = ds.tensors["best_indices"][:, -1].numpy()  # 1-based candidate index of the global best
    lo_e, hi_e = early_range(N_CAND, CFG)
    lo_l, hi_l = late_range(N_CAND, CFG)
    margin, delta, sep = CFG.record_margin, CFG.near_best_delta, CFG.value_separation

    for i in range(len(ds)):
        b = int(final_best[i])
        s_best = sims[i, b - 1]
        if case == "late":
            assert lo_l <= b <= hi_l and b < N_CAND  # later distractor remains
        else:
            assert lo_e <= b <= hi_e
        others = np.delete(sims[i], b - 1)
        assert (others <= s_best - margin + TOL).all()  # strict margin below the global best
        if case in ("late", "early", "near_best"):
            assert events[i, b - 1] and not events[i, b:].any()
        if case == "near_best":
            later = sims[i, b:]
            in_band = (later >= s_best - delta - TOL) & (later <= s_best - margin + TOL)
            assert in_band.sum() >= math.ceil(len(later) / 2)
            assert not events[i, b:].any()
            # band distractors carry values separated from the retained target
            assert (np.abs(values[i, b:][in_band] - targets[i, b - 1]) >= sep - 1e-6).all()
        # forced revisions (every record after the first candidate) are value-separated
        for t in np.flatnonzero(events[i])[1:]:
            assert abs(targets[i, t] - targets[i, t - 1]) >= sep - 1e-6
    assert targets.min() >= 0 and targets.max() <= 1


def test_target_changes_only_at_events_in_iid_data():
    ds = generate_dataset({"iid": 100}, L, seed=5)
    targets, events = cand(ds, "targets"), cand(ds, "event_mask")
    changed = targets[:, 1:] != targets[:, :-1]
    assert not (changed & ~events[:, 1:]).any()


def test_invalid_configurations_fail_clearly():
    with pytest.raises(ValueError, match="near_best_delta"):
        NearestNeighborConfig(near_best_delta=0.01, record_margin=0.02).validate()
    with pytest.raises(ValueError, match="value_separation"):
        NearestNeighborConfig(value_separation=0.7).validate()
    with pytest.raises(ValueError, match="sequence_length"):
        generate_dataset({"late": 2}, 4, seed=0)
    with pytest.raises(ValueError, match="unknown case"):
        from q_slstm.datasets.nearest_neighbor import generate_case
        generate_case(np.random.default_rng(0), "bogus", 1, 10, CFG)


def test_generation_is_reproducible_and_does_not_touch_global_rng():
    torch_state, np_state = torch.get_rng_state(), np.random.get_state()[1].copy()
    a = make_balanced_suite(40, 12, seed=7).checksums()
    b = make_balanced_suite(40, 12, seed=7).checksums()
    c = make_balanced_suite(40, 12, seed=8).checksums()
    assert a == b and a["combined"] != c["combined"]
    assert torch.equal(torch_state, torch.get_rng_state())
    assert np.array_equal(np_state, np.random.get_state()[1])
    # different streams of one seed are independent
    assert make_balanced_suite(40, 12, 7, "test").checksums()["inputs"] != \
        make_balanced_suite(40, 12, 7, "extrapolation").checksums()["inputs"]


def test_balanced_counts_distribute_remainder_deterministically():
    assert balanced_case_counts(1000) == {c: 250 for c in CASE_TYPES}
    assert balanced_case_counts(10) == {"iid": 3, "late": 3, "early": 2, "near_best": 2}
    assert make_balanced_suite(10, 12, 0).case_counts() == balanced_case_counts(10)


def test_split_ids_do_not_overlap():
    pool = make_train_pool(100, 12, seed=1)
    train, val, train_idx, val_idx = split_train_val(pool, 10, seed=2)
    assert len(train) == 90 and len(val) == 10
    assert set(train.sequence_ids.tolist()).isdisjoint(val.sequence_ids.tolist())
    assert set(train.sequence_ids.tolist()) | set(val.sequence_ids.tolist()) == set(pool.sequence_ids.tolist())
    test = make_balanced_suite(8, 12, seed=1)
    assert set(test.sequence_ids.tolist()).isdisjoint(pool.sequence_ids.tolist())
    again = split_train_val(pool, 10, seed=2)
    assert np.array_equal(again[2], train_idx)


def test_paper_scale_sizes():
    assert make_train_pool(4000, 32, 0).case_counts()["iid"] == 4000
    assert make_balanced_suite(1000, 32, 0).case_counts() == {c: 250 for c in CASE_TYPES}
