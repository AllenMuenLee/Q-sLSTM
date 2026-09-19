import pytest
import torch
import torch.nn as nn

from q_slstm.models.sequence_wrappers import CustomLSTM
from q_slstm.models.sequence_wrappers import CustomQsLSTM

from qslstm_test_utils import make_cell

BATCH, SEQ, IN, HID, OUT = 3, 5, 2, 4, 2


def make_wrapper(dtype=torch.float32):
    return CustomQsLSTM(IN, HID, make_cell(IN, HID, OUT, dtype=dtype))


def make_input(dtype=torch.float32, batch=BATCH, seq=SEQ, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, seq, IN, generator=generator).to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shapes_dtype_and_device(dtype):
    x = make_input(dtype)
    outputs, state = make_wrapper(dtype)(x)

    assert outputs.shape == (BATCH, SEQ, OUT)
    assert len(state) == 4
    for tensor in (outputs, *state):
        assert tensor.dtype == dtype
        assert tensor.device == x.device
    for tensor in state:
        assert tensor.shape == (BATCH, HID)


def test_chunked_recurrence_matches_full_sequence():
    model = make_wrapper()
    x = make_input(seq=6)

    full_out, full_state = model(x)

    out_a, state_a = model(x[:, :3])
    out_b, state_b = model(x[:, 3:], state_a)

    assert torch.allclose(torch.cat([out_a, out_b], dim=1), full_out, atol=1e-6)
    for got, want in zip(state_b, full_state):
        assert torch.allclose(got, want, atol=1e-6)

    # The chunk boundary must matter: resetting m (or n) changes the result.
    h, c, n, m = state_a
    out_reset_m, _ = model(x[:, 3:], (h, c, n, torch.zeros_like(m)))
    assert not torch.allclose(out_reset_m, out_b, atol=1e-6)


def test_backward_pass_is_finite():
    model = make_wrapper()
    x = make_input().requires_grad_(True)
    outputs, state = model(x)
    loss = outputs[:, -1, :].pow(2).mean()
    loss.backward()

    assert torch.isfinite(outputs).all() and torch.isfinite(loss)
    assert all(torch.isfinite(s).all() for s in state)
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_long_sequence_stays_finite():
    model = make_wrapper()
    x = make_input(seq=200) * 5
    outputs, state = model(x)
    assert torch.isfinite(outputs).all()
    assert all(torch.isfinite(s).all() for s in state)


@pytest.mark.parametrize(
    "x, match",
    [
        (torch.zeros(BATCH, IN), "batch, sequence, input_size"),
        (torch.zeros(BATCH, 0, IN), "at least 1"),
        (torch.zeros(BATCH, SEQ, IN + 1), "input_size"),
    ],
)
def test_rejects_malformed_input(x, match):
    with pytest.raises(ValueError, match=match):
        make_wrapper()(x)


def test_rejects_malformed_state():
    model = make_wrapper()
    x = make_input()
    good = tuple(torch.zeros(BATCH, HID) for _ in range(4))
    with pytest.raises(ValueError, match="4 tensors"):
        model(x, good[:3])
    with pytest.raises(ValueError, match="shape"):
        model(x, tuple(torch.zeros(BATCH + 1, HID) for _ in range(4)))
    with pytest.raises(ValueError, match="shape"):
        model(x, tuple(torch.zeros(BATCH, HID + 1) for _ in range(4)))


class TinyLSTMCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(IN, HID)
        self.output_post_processing = nn.Linear(HID, OUT)

    def forward(self, x, hidden):
        h_t, c_t = self.lstm_cell(x, hidden)
        return self.output_post_processing(h_t), h_t, c_t


def test_classical_wrapper_keeps_two_state_contract():
    model = CustomLSTM(IN, HID, TinyLSTMCell())
    x = make_input()
    outputs, state = model(x)

    assert outputs.shape == (BATCH, SEQ, OUT)
    assert len(state) == 2 and all(s.shape == (BATCH, HID) for s in state)

    out_a, state_a = model(x[:, :2])
    out_b, state_b = model(x[:, 2:], state_a)
    assert torch.allclose(torch.cat([out_a, out_b], dim=1), outputs, atol=1e-6)
