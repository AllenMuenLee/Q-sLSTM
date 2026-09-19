import math

import pytest
import torch

from q_slstm.models.sequence_wrappers import CustomLSTM

from qslstm_test_utils import FixedGate
from qslstm_test_utils import make_qlstm_cell


def sigmoid(v):
    return 1 / (1 + math.exp(-v))


def test_single_step_matches_conventional_lstm_recurrence():
    q_i, q_f, q_z, q_o = [0.5, -0.4], [0.2, 0.6], [0.3, -0.7], [0.1, -0.2]
    gates = [FixedGate([q]) for q in (q_i, q_f, q_z, q_o)]
    cell = make_qlstm_cell(1, 2, output_size=1, gates=gates, dtype=torch.float64)

    x = torch.tensor([[0.4], [-0.1]], dtype=torch.float64)
    h_prev = torch.tensor([[0.1, -0.2], [0.3, 0.0]], dtype=torch.float64)
    c_prev = torch.tensor([[0.5, -0.4], [0.2, 0.9]], dtype=torch.float64)

    out = cell(x, (h_prev, c_prev))
    assert len(out) == 3  # (y, h, c): no normalizer or stabilizer state
    y_t, h_t, c_t = out

    for b in range(2):
        for k in range(2):
            c = sigmoid(q_f[k]) * c_prev[b, k].item() + sigmoid(q_i[k]) * math.tanh(q_z[k])
            h = sigmoid(q_o[k]) * math.tanh(c)
            assert c_t[b, k].item() == pytest.approx(c)
            assert h_t[b, k].item() == pytest.approx(h)

    assert torch.allclose(y_t, cell.output_post_processing(h_t))
    assert y_t.dtype == torch.float64


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sequence_shapes_dtype_and_chunking(dtype):
    cell = make_qlstm_cell(2, 4, output_size=3, dtype=dtype)
    model = CustomLSTM(2, 4, cell)
    x = torch.randn(3, 6, 2, dtype=dtype)

    outputs, state = model(x)
    assert outputs.shape == (3, 6, 3) and outputs.dtype == dtype
    assert len(state) == 2 and all(s.shape == (3, 4) and s.dtype == dtype for s in state)

    out_a, state_a = model(x[:, :2])
    out_b, _ = model(x[:, 2:], state_a)
    assert torch.allclose(torch.cat([out_a, out_b], dim=1), outputs, atol=1e-6)


def test_backward_is_finite():
    cell = make_qlstm_cell(2, 4, output_size=1)
    model = CustomLSTM(2, 4, cell)
    x = torch.randn(3, 5, 2, requires_grad=True)

    outputs, _ = model(x)
    loss = outputs[:, -1, :].pow(2).mean()
    loss.backward()

    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
