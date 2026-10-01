import json

import pytest
import torch

from q_slstm.experiments import scalar_tasks as st
from q_slstm.models.classical import CLASSICAL_SLSTM_RECURRENCE, ClassicalSLSTM


def unstabilized_slstm(model, x):
    """Direct xLSTM recurrence with raw i = exp(i~), f = exp(f~) (no stabilizer), float64."""
    batch, length, _ = x.shape
    h = c = n = x.new_zeros(batch, model.hidden_size)
    outputs = []
    for t in range(length):
        pre_i, pre_f, pre_z, pre_o = model.gates(torch.cat((x[:, t], h), dim=-1)).chunk(4, dim=-1)
        i, f = torch.exp(pre_i), torch.exp(pre_f)
        c = f * c + i * torch.tanh(pre_z)
        n = f * n + i
        h = torch.sigmoid(pre_o) * c / n
        outputs.append(h)
    return model.head(torch.stack(outputs, dim=1))


def test_slstm_matches_unstabilized_exponential_gates():
    torch.manual_seed(0)
    model = ClassicalSLSTM(1, 4, 1).double()
    x = torch.randn(3, 12, 1, dtype=torch.float64)
    y, (h, c, n, m) = model(x)
    torch.testing.assert_close(y, unstabilized_slstm(model, x))
    assert y.shape == (3, 12, 1) and all(s.shape == (3, 4) for s in (h, c, n, m))


def test_slstm_forget_gate_is_exponential_not_sigmoid():
    # With f~ large the exp gate exceeds 1 (sigmoid never does): old memory outweighs a new write.
    model = ClassicalSLSTM(1, 1, 1).double()
    with torch.no_grad():
        model.gates.weight.zero_()
        model.gates.bias.copy_(torch.tensor([0.0, 3.0, 5.0, 10.0]))  # i~, f~, z~, o~
    x = torch.zeros(1, 2, 1, dtype=torch.float64)
    _, (h, c, n, m) = model(x)
    # m accumulates log f = 3 per step (m_t = max(f~ + m_prev, i~)); c/n stays tanh(5).
    assert h.item() == pytest.approx(torch.sigmoid(torch.tensor(10.0)).item() * torch.tanh(torch.tensor(5.0)).item())
    assert m.item() == pytest.approx(6.0)


def test_slstm_stays_finite_where_raw_exponentials_overflow():
    model = ClassicalSLSTM(1, 2, 1)
    with torch.no_grad():
        model.gates.bias.fill_(200.0)  # exp(200) overflows float32
    y, state = model(torch.ones(2, 50, 1))
    assert torch.isfinite(y).all() and all(torch.isfinite(s).all() for s in state)


@pytest.mark.parametrize("model", ["lstm", "slstm"])
def test_classical_models_share_the_quantum_configuration(model):
    base = {"task": "ema", "seed": 5, "scale": "paper", "run_date": "2026-09-28"}
    classical = st.resolve_config({**base, "model": model})
    quantum = st.resolve_config({**base, "model": "qslstm"})
    differ = {k for k in quantum if quantum[k] != classical[k]}
    assert differ == {"model", "n_qubits", "qslstm_recurrence"}
    assert classical["n_qubits"] is None
    assert classical["qslstm_recurrence"] == (CLASSICAL_SLSTM_RECURRENCE if model == "slstm" else None)


@pytest.mark.parametrize("model", ["lstm", "slstm"])
def test_classical_models_run_the_scalar_pipeline(model, tmp_path):
    config = st.resolve_config({"model": model, "task": "ema", "seed": 1, "scale": "pilot",
                                "save_dir": str(tmp_path), "run_date": "2026-09-30"})
    run_dir = st.run_experiment(config)
    complete = json.loads((run_dir / "complete.json").read_text())
    assert complete["model"] == model and complete["test_mse"] >= 0
    # Same model seed -> same initial parameters.
    a, b = st.build_model(config), st.build_model(config)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
