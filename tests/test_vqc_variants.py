"""The VQC option: both circuits build, differ, and never share result directories or analyses."""

import json

import pytest
import torch

from q_slstm.experiments import nearest_neighbor as nn_exp
from q_slstm.models.factory import build_quantum_model
from q_slstm.models.vqc import VQC_VARIANTS, config_vqc, get_vqc_class
from q_slstm.models.vqc_bounded import VQC as BoundedVQC
from q_slstm.models.vqc_original import VQC as OriginalVQC

from experiments.test_nn_pipeline import load_script


@pytest.mark.parametrize("model", ["qlstm", "qslstm", "qslstm_log"])
@pytest.mark.parametrize("vqc, cls", [("original", OriginalVQC), ("bounded", BoundedVQC)])
def test_factory_builds_the_chosen_vqc_for_every_gate(model, vqc, cls):
    net = build_quantum_model(model, 1, 2, 1, 2, seed=5, vqc=vqc)
    for gate in ("input_gate", "forget_gate", "cell_gate", "output_gate"):
        assert type(getattr(net.cell, gate)) is cls


def test_variants_share_initial_parameters_but_not_outputs():
    a = build_quantum_model("qslstm", 1, 2, 1, 2, seed=5, vqc="original")
    b = build_quantum_model("qslstm", 1, 2, 1, 2, seed=5, vqc="bounded")
    for x, y in zip(a.state_dict().values(), b.state_dict().values()):
        assert torch.equal(x, y)
    with torch.no_grad():
        a.cell.input_gate.weights.fill_(2.0)
        b.cell.input_gate.weights.fill_(2.0)
    x = torch.rand(3, 3)
    assert not torch.allclose(a.cell.input_gate(x), b.cell.input_gate(x))


def test_unknown_vqc_is_rejected():
    with pytest.raises(ValueError, match="vqc"):
        get_vqc_class("nope")
    with pytest.raises(ValueError, match="vqc"):
        nn_exp.resolve_config({"model": "qlstm", "scale": "pilot", "vqc": "nope"})


def test_configs_without_vqc_are_original():
    assert config_vqc({}) == "original"
    assert set(VQC_VARIANTS) == {"original", "bounded"}


def test_nearest_neighbor_directories_are_separated_by_vqc():
    base = {"model": "qslstm_log", "seed": 3, "scale": "pilot", "run_date": "2026-01-01", "save_dir": "r"}
    original = nn_exp.run_directory(nn_exp.resolve_config({**base, "vqc": "original"}))
    bounded = nn_exp.run_directory(nn_exp.resolve_config({**base, "vqc": "bounded"}))
    assert original.parts == ("r", "pilot", "2026-01-01", "seed_3", "qslstm_log")
    assert bounded.parts == ("r", "pilot", "vqc_bounded", "2026-01-01", "seed_3", "qslstm_log")


def test_analysis_refuses_to_mix_vqcs(tmp_path):
    ar = load_script("analyze_results")
    for model, vqc in (("qlstm", "original"), ("qslstm", "bounded")):
        run_dir = tmp_path / "seed_1" / model
        run_dir.mkdir(parents=True)
        config = {"kind": "nearest_neighbor_run", "model": model, "seeds": {"run_seed": 1}, "scale_label": "pilot"}
        if vqc != "original":
            config["vqc"] = vqc
        (run_dir / "config.json").write_text(json.dumps(config))
        (run_dir / "complete.json").write_text("{}")
    with pytest.raises(SystemExit, match="different VQCs"):
        ar.discover_runs(tmp_path)
