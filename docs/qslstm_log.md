# Logarithmic-gate Q-sLSTM

Select `qslstm_log` to use the new variation; `qslstm` retains its odds-ratio
gates. Both the input and forget gates change:

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
At `q=-1` the gate is zero, and at `q=0` it is `ln(2)`. Expectations must
lie in `[-1, 1)`: nonfinite values, values outside `[-1, 1]`, and the singular
endpoint `q=1` raise `ValueError`. A zero normalizer also raises rather than
producing an undefined output. `gate_epsilon` is accepted for constructor
compatibility but does not clip gates or floor the normalizer.

The four VQCs, candidate/output transforms, output projection, and parameter
initialization match `qslstm`. The sequence wrapper returns
`(outputs, (h, c, n, scale))`, where `C=c*2**scale` and `N=n*2**scale`.
Binary scaling keeps stored states bounded during repeated amplification.
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
Use `--model qslstm_log` in the time-series, nearest-neighbor, or solar-generation
training entrypoints. Experiment configs identify its recurrence as
`log_gate_binary_scale_v1`. Existing paired comparison plots are still specific
to `qlstm` versus `qslstm`.
