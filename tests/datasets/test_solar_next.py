import numpy as np
import pytest
import torch

from q_slstm.datasets.solar_scalar import DEFAULT_SOLAR_ROOT, _series, reference_mse
from q_slstm.experiments.scalar_tasks import ALL_TASKS, make_datasets, resolve_config

pytestmark = pytest.mark.skipif(not (DEFAULT_SOLAR_ROOT / "prepared_paper.json").exists(),
                                reason="collected IESO solar data not present")


@pytest.fixture(scope="module")
def datasets():
    config = resolve_config({"model": "qlstm", "task": "solar_next", "seed": 1, "scale": "paper"})
    return config, make_datasets(config)


def test_listed_and_sized(datasets):
    config, ds = datasets
    assert "solar_next" in ALL_TASKS
    assert len(ds["train"]) == config["optimizer_train_size"]
    assert len(ds["val"]) == config["val_size"] and len(ds["test"]) == config["test_size"]
    assert all(d.sequence_length == config["sequence_length"] for d in ds.values())


def test_target_is_next_hour_input(datasets):
    _, ds = datasets
    for d in ds.values():
        x, y = d.tensors["inputs"][..., 0], d.tensors["targets"][..., 0]
        assert torch.equal(y[:, :-1], x[:, 1:])
        assert x.min() >= -1 and x.max() <= 1


def test_splits_are_chronological_and_clean(datasets):
    config, ds = datasets
    frame, bounds, _ = _series(DEFAULT_SOLAR_ROOT)
    ok = (frame["status"] == "ok").to_numpy()
    length = config["sequence_length"]
    for split in ("train", "val", "test"):
        lo, hi = bounds[split]["rows"]
        starts = np.asarray(ds[split].metadata["window_start_rows"])
        assert starts.min() >= lo and starts.max() + length < hi
        assert all(ok[s:s + length + 1].all() for s in starts)


def test_held_out_windows_do_not_depend_on_seed(datasets):
    _, ds = datasets
    other = make_datasets(resolve_config({"model": "qslstm", "task": "solar_next", "seed": 2, "scale": "paper"}))
    for split in ("val", "test"):
        assert ds[split].checksums()["combined"] == other[split].checksums()["combined"]
    assert ds["train"].checksums()["combined"] != other["train"].checksums()["combined"]


def test_reference_baselines(datasets):
    _, ds = datasets
    ref = reference_mse(ds["test"])
    x, y = ds["test"].tensors["inputs"], ds["test"].tensors["targets"]
    assert ref["persistence"] == pytest.approx(float(((x - y) ** 2).double().mean()))
    assert 0 < ref["persistence"] < ref["daily_persistence"]
