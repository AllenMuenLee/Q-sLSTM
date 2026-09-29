# Logarithmic-gate Q-sLSTM

Select `qslstm_log` to use the new variation; `qslstm` keeps its odds-ratio
gates `(1 + q) / (1 - q)`. In both, the input and the forget gate are amplified,
and the normalized readout `C_t / N_t` is kept:

```text
i = ln(2 / (1 - q_i))
f = ln(2 / (1 - q_f))
z = tanh(q_z)
o = sigmoid(q_o)
C_t = f * C_prev + i * z
N_t = f * N_prev + i
h_t = o * C_t / N_t
```

The natural logarithm is the gate value itself; it is not exponentiated.
At `q=-1` a gate is zero, and at `q=0` it is `ln(2)`. Expectations must
lie in `[-1, 1)`: nonfinite values, values outside `[-1, 1]`, and the singular
endpoint `q=1` raise `ValueError`. A zero normalizer also raises rather than
producing an undefined output. `gate_epsilon` is accepted for constructor
compatibility but does not clip gates or floor the normalizer.

The input and forget expectations are simulated in float64 (`float64_expectations` in the cell) and the gates are
evaluated in float64 before being cast to the float32 state. A float32 simulation uses a complex64
statevector whose `|<Z>|` can exceed 1 by ~3e-7, and casting a float64 expectation to float32 rounds
every value within ~3e-8 of +-1 onto +-1. Training pushes `q_i` toward +1 (a larger write), and at
t=0 (`h=0`) a hidden qubit's expectation is a pure `sin` of one weight, so the float32 path hit
`q=1` (singular), `|q|>1`, or `q=-1` (zero gate, and with the empty initial normalizer, `N=0`).

The four VQCs, candidate/output transforms, output projection, and parameter
initialization match `qslstm`. The sequence wrapper returns
`(outputs, (h, c, n, scale))`, where `C=c*2**scale` and `N=n*2**scale`.
Binary scaling keeps stored states bounded during repeated amplification (with an
amplified forget gate `N_t` can grow without bound).
Diagnostics report the new-write proportion `alpha=i/N_t`.

```python
from q_slstm.models.factory import build_quantum_model

model = build_quantum_model(
    "qslstm_log", input_size=1, hidden_size=2, output_size=1,
    qnn_depth=1, seed=42,
)
```

The cell is also available as
`q_slstm.models.q_slstm_log_cell.CustomQsLSTMLogCell`.
Use `--model qslstm_log` in the time-series, nearest-neighbor, solar-generation, scalar-task, or
solar_next training entrypoints.

## Recurrence tags

Experiment configs record the recurrence as `qslstm_recurrence`:

| tag | model | forget gate | gates evaluated on |
|---|---|---|---|
| `log_gate_binary_scale_f64_gates_v2` (current) | qslstm_log | `ln(2/(1-q_f))` | float64 expectations |
| `xlstm_stabilized_v1` (current) | qslstm | `(1+q_f)/(1-q_f)` | float32 expectations, eps clamp |
| `log_gate_binary_scale_v1` | qslstm_log | `ln(2/(1-q_f))` | float32 (could crash, see above) |
| `log_input_sigmoid_forget_binary_scale_f64_gates_v2` | qslstm_log | `sigmoid(q_f)` | float64 |
| `log_input_sigmoid_forget_binary_scale_v1` | qslstm_log | `sigmoid(q_f)` | float32 (could crash) |
| `xlstm_stabilized_sigmoid_forget_v1` | qslstm | `sigmoid(q_f)` | float32, eps clamp |

The sigmoid-forget tags are runs from 2026-09-27 to 2026-09-29. To replay such a checkpoint, build the
model and call `q_slstm.models.factory.match_recurrence(model, config["qslstm_recurrence"])`, which
switches the cell back to the sigmoid forget gate.
