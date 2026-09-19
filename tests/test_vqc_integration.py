"""Small integration checks against the real PennyLane VQC and the full quantum cell."""

import pytest
import torch

from q_slstm.models.q_slstm_cell import CustomQsLSTMCell
from q_slstm.models.vqc import VQC
from q_slstm.models.sequence_wrappers import CustomQsLSTM


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_vqc_broadcasts_batch_and_returns_batch_first(dtype):
    vqc = VQC(vqc_depth=2, n_qubits=4, n_class=3)
    x = torch.rand(5, 4, dtype=dtype)

    out = vqc(x)

    assert out.shape == (5, 3)
    assert out.dtype == dtype
    assert (out.abs() <= 1 + 1e-6).all()
    # broadcasting must agree with evaluating each batch row on its own
    single = torch.cat([vqc(x[i:i + 1]) for i in range(5)])
    assert torch.allclose(out, single, atol=1e-6)


def test_vqc_rejects_unbatched_input():
    vqc = VQC(vqc_depth=1, n_qubits=3, n_class=2)
    with pytest.raises(ValueError, match="batch"):
        vqc(torch.rand(3))


def test_real_cell_forward_backward_is_finite():
    torch.manual_seed(0)
    cell = CustomQsLSTMCell(1, 2, 1, vqc_depth=1)
    model = CustomQsLSTM(1, 2, cell)
    x = torch.randn(2, 3, 1, requires_grad=True)

    outputs, state = model(x)
    loss = outputs[:, -1, :].pow(2).mean()
    loss.backward()

    assert outputs.shape == (2, 3, 1) and outputs.dtype == torch.float32
    assert len(state) == 4 and all(s.shape == (2, 2) and s.dtype == torch.float32 for s in state)
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    for gate in (cell.input_gate, cell.forget_gate, cell.cell_gate, cell.output_gate):
        assert gate.weights.grad is not None and torch.isfinite(gate.weights.grad).all()
