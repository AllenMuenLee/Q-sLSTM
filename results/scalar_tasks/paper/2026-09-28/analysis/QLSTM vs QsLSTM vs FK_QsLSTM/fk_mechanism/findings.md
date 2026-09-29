# Why fk_qslstm wins on the scalar tasks

Produced by `scripts/experiments/scalar_tasks/diagnose_fk_qslstm.py`: every best checkpoint (20 seeds × 6 tasks ×
3 models) replayed on 256 test sequences. Tables: `per_run.csv`, `summary.csv`.

## 1. Architecture: the circuit is wrapped in classical affine layers

| | QLSTM / Q-sLSTM | fk_qslstm |
|---|---|---|
| gate pre-activation | raw Pauli-Z expectation q ∈ [−1, 1] of `VQC([x, h])` | `Linear(VQC(Linear([x, h])))`, unbounded |
| VQC input angles | raw x ∈ [−1, 1] and h (no scale, no bias) | learned affine map of [x, h] |
| forget gate | sigmoid(q) ∈ [0.269, 0.731] | exp(·) ∈ (0, ∞) |
| cell candidate | tanh(q) ∈ [−0.76, 0.76] | tanh(·) ∈ (−1, 1) |
| normalizer / stabilizer | Q-sLSTM: yes (h = o·c/n) | none (h = o·tanh(c)) |
| trainable params | 45 (40 in VQCs) | 261 (40 in VQCs, 216 in encoders) |

## 2. The forget-gate ceiling is binding for QLSTM / Q-sLSTM (`forget_gate_range.png`)

- QLSTM and Q-sLSTM never exceed f = 0.731 in any run. That limits retention to roughly 0.731^k: about 3 steps
  of time constant, with 4% of a value left after 10 steps.
- They press against that ceiling. On flip_flop, narma, running_max and sine_next, 12–41% of their raw VQC
  outputs have |q| > 0.9, against 0.5–6.5% for fk_qslstm. Q-sLSTM's median f is 0.72 on flip_flop, narma
  and running_max.
- fk_qslstm uses f ≥ 0.9 on 25–69% of steps (median f ≈ 0.94–0.98 on delay, running_max and narma), and often
  f > 1 (up to 18).
- The tasks QLSTM / Q-sLSTM fail are the ones that need long retention: flip_flop (hold a bit through long
  silences), running_max (hold the maximum) and delay (keep 4 separate past values). ema needs a decay of 0.8,
  just above the 0.731 ceiling.

## 3. How fk_qslstm latches: saturating cell state, not gating

On flip_flop, fk_qslstm's mean write proportion is only slightly lower on silent steps than on pulse steps
(0.24 vs 0.32). So it does not latch the way an ideal gated memory would (write on a pulse, freeze otherwise).
Because f can exceed 1 and there is no normalizer, |c| grows until tanh(c) saturates. The value is then held
at ±1 by the sign of c, a bistable attractor.

Spot check, 4 seeds × 64 sequences:

| task | model | median \|c\| | p95 \|c\| | share of \|tanh c\| > 0.95 |
|---|---|---|---|---|
| flip_flop | QLSTM | 0.05 | 0.49 | 0% |
| flip_flop | fk_qslstm | 1.6 | 5.3e3 | 46% |
| running_max | QLSTM | 0.44 | 1.6 | 0% |
| running_max | fk_qslstm | 0.74 | 8.2 | 35% |
| delay | QLSTM | 0.31 | 0.69 | 0% |
| delay | fk_qslstm | 0.69 | 1.8e8 | 42% |

This works over 32 steps but is fragile: c grows geometrically, so longer sequences risk overflow. This sweep
ran with `run_extrapolation: false`, so it was never tested.

## 4. fk_qslstm's trained solution does use the circuit's nonlinearity (`linearized_vqc_mse.png`)

Each VQC was replaced by its least-squares affine fit, with no retraining:

- fk_qslstm degrades by a median of 25× (delay) up to 8000× (flip_flop). The exception is narma (1.02×), where
  it is essentially linear.
- QLSTM changes by 0.9–2.7×, Q-sLSTM by 1.2–3.4×.
- The fk encoders spread the angles further: 95th-percentile |angle| is 1.1–1.8 rad, against ≤ 0.92 rad of raw
  input for QLSTM / Q-sLSTM. At those angles the circuit's cos/sin response is nonlinear enough to be useful.

This shows the trained fk solution relies on the circuit's nonlinearity. It does not show that a quantum
circuit is needed: a classical nonlinearity in the same slot might do as well.

## Conclusion

fk_qslstm's advantage comes mainly from the classical affine layers around each VQC:

1. They remove the [0.27, 0.73] forget-gate ceiling, which allows long retention and saturating latches.
2. They rescale the circuit's input angles and outputs, so the circuit works in a useful nonlinear range.
3. They add 5.8× more parameters.

The recurrence type (exponential sLSTM gates vs sigmoid) is secondary. Q-sLSTM also has exponential input
gates, but its forget gate is still sigmoid(q) with q ∈ [−1, 1].

## Ablations to confirm causally (need training)

- Q-sLSTM / QLSTM with a trainable per-gate scale and bias on q (+32 params). Tests the gate-range explanation
  without the full encoders.
- Q-sLSTM with the fk encoders. Separates recurrence type from encoders.
- fk_qslstm with each VQC replaced by a classical nonlinearity of the same width (e.g. Linear → tanh or cos).
  Tests whether the circuit matters.
- fk_qslstm evaluated on longer sequences (`run_extrapolation`). Tests the stability of the saturating latch.
