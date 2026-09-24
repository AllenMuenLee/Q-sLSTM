# Implementation Prompt: Convert the Current QLSTM to the Q-sLSTM Structure

> Historical specification. The current production gate arithmetic, state
> representation, and boundary handling are defined in
> [the polynomial update](q-slstm_polynomial_update.md), which supersedes the
> clipped log-gate/stabilizer instructions below.

Implement the Q-sLSTM architecture described in
`implementation_prompts/q-slstm_model_structure.md` in the current repository.
Modify the production code and add focused tests. Preserve the existing classical
LSTM baseline and the existing `qlstm` command-line model name.

This implementation intentionally extends the structure document's recurrent
state with the xLSTM stabilizer `m_t`. Wherever the structure document's
three-state `(h, c, n)` contract or raw input/forget ratio conflicts with this
prompt, this prompt takes precedence: use `(h, c, n, m)` and never materialize
the unstabilized ratio.

## Repository context

The relevant current files are:

- `src/q_slstm/models/q_slstm_cell.py`: PennyLane VQC and quantum cell.
- `src/q_slstm/models/vqc_components.py`: circuit layer helpers.
- `src/q_slstm/models/sequence_wrappers.py`: two-state sequence wrapper.
- `scripts/experiments/time_series/train_timeseries.py`: model construction,
  optional input projection, and the classical baseline cell.
- `src/q_slstm/trainers/train.py` and
  `src/q_slstm/trainers/utils.py`: select the last sequence output.

At present the quantum cell is still a conventional LSTM-like cell: it applies
sigmoid to both the input and forget VQC results, keeps only `(h, c)`, and computes
`h = o * tanh(c)`. Replace that quantum recurrence with the normalized Q-sLSTM
recurrence below. Do not change the classical baseline recurrence.

## Required behavior

### 1. Keep four independent VQCs

For every time step, concatenate the current input and previous hidden state on
the feature dimension:

```python
combined = torch.cat((x_t, h_prev), dim=-1)
```

The combined width must be `input_size + hidden_size`, which is also the number
of qubits. Feed the same combined tensor to four independently parameterized
VQCs: input, forget, cell/candidate, and output. Each VQC must return
`hidden_size` Pauli-Z expectation values.

Keep the circuit specified by the structure document:

1. Hadamard on every qubit.
2. One input `RY` rotation per qubit.
3. For each variational depth, the existing two-pass nearest-neighbor CNOT
   entangling layer followed by one trainable `RY` per qubit.
4. Pauli-Z expectation on the first `hidden_size` qubits.

Retain PennyLane parameter broadcasting and Torch autograd. Prefer an explicit
dimension operation such as `movedim(0, -1)` over `.T` when turning the stacked
expectations into `[batch_size, hidden_size]`, and verify the resulting shape.

### 2. Keep input and forget gates in the bounded log domain

Do not evaluate `(1 + q) / (1 - q)` directly, do not apply sigmoid, and do not
materialize that raw ratio with `exp(log_gate)`. A simulator can return a value
at, or a tiny numerical amount outside, `[-1, 1]`; denominator-only clamping
could still produce a negative or unbounded gate. The xLSTM stabilizer in the
next section consumes the log gates directly.

Add one shared helper used by both the input and forget VQC results. It must:

1. Accept the raw expectation tensor `q`.
2. Compute a dtype-aware effective epsilon:

   ```text
   effective_epsilon = max(configured_epsilon, torch.finfo(q.dtype).eps)
   ```

3. Clamp the expectation itself:

   ```text
   q_safe = clamp(q, -1 + effective_epsilon, 1 - effective_epsilon)
   ```

4. Compute the ratio in the log domain with stable primitives:

   ```text
   log_gate = log1p(q_safe) - log1p(-q_safe)
   ```

5. Return `log_gate` without exponentiating it. The expectation clamp gives the
   explicit finite log bound

   ```text
   epsilon_log_limit = log(2 - effective_epsilon) - log(effective_epsilon)
   -epsilon_log_limit <= log_gate <= epsilon_log_limit
   ```

The transform must remain differentiable for values inside the clamp interval.
Do not detach tensors, convert gate values through NumPy, force them to
`.float()`, or exponentiate the unnormalized log gates. Validate the configured
epsilon once in the cell constructor and raise `ValueError` unless
`0 < epsilon < 1`. Use a default of `1e-6`.

This logarithmic calculation is required for both `ell_i` and `ell_f`. Note that
the unstabilized ratio would map `q in (-1, 1)` to `(0, +infinity)`, not to
`(-1, 1)`, which is why it must not be materialized.

If a raw VQC result contains NaN or infinity, fail clearly rather than silently
turning it into an ordinary log gate. Tests must assert that normal forward and
backward passes remain finite.

### 3. Apply the original xLSTM exponential stabilizer

The stabilizer is a required recurrent state, not a temporary clamp. The Q-sLSTM
cell state is `(h_t, c_t, n_t, m_t)`. Its `forward` method must accept
`(h_prev, c_prev, n_prev, m_prev)` and compute elementwise:

```text
q_i = input_gate(combined)
q_f = forget_gate(combined)
q_z = cell_gate(combined)
q_o = output_gate(combined)

ell_i = bounded_log_ratio(q_i)  # returns the log ratio, not its exp
ell_f = bounded_log_ratio(q_f)

m_t = maximum(ell_f + m_prev, ell_i)
i_prime = exp(ell_i - m_t)
f_prime = exp(ell_f + m_prev - m_t)

z_t = tanh(q_z)
o_t = sigmoid(q_o)

c_t = f_prime * c_prev + i_prime * z_t
n_t = f_prime * n_prev + i_prime
h_t = o_t * (c_t / clamp_min(n_t, effective_epsilon))
y_t = output_post_processing(h_t)
```

Use `torch.maximum`, not a Python scalar `max`, because the stabilization is
elementwise for every batch and hidden coordinate. In exact arithmetic both
exponent arguments are non-positive, which guarantees:

```text
0 < i_prime <= 1
0 < f_prime <= 1
```

One of the two scaled gates is exactly one at each coordinate (both may be one
on a tie). Because floating-point subtraction can produce a tiny positive value,
clamp each exponent argument with `max=0` immediately before `torch.exp`. Do not
add an arbitrary lower clamp: underflow to zero is safe and avoids changing the
relative weighting. Assert the finite and upper-bound properties in tests.

Initialize `m_0` to zero, with the same shape, dtype, and device as the other
states. Carry `m_t` through every recurrent step and across chunked inference.
Do not detach it between steps.

Return:

```python
return y_t, h_t, c_t, n_t, m_t
```

Use the same effective epsilon policy for the normalizer denominator. Preserve
the input tensor's floating dtype and device throughout the cell. Remove the
current unconditional `.float()` conversions.

The stabilizer prevents either scaled input/forget gate from overflowing and
keeps both at or below one. Add finite-value checks in tests at the cell and
sequence levels. Do not add ad-hoc clamps to `c_t`, `n_t`, `m_t`, or `h_t`,
because independently clipping recurrent states would change the normalized
recurrence.

### 4. Add a dedicated four-state sequence wrapper

The existing `CustomLSTM` wrapper and `StandardLSTMCell` use the classical
two-state `(h, c)` contract. Keep that path working. Do not make the baseline
pretend it has a normalizer.

Add a clearly named quantum sequence wrapper (for example `CustomQLSTM`) in
`src/q_slstm/models/sequence_wrappers.py`, or refactor the wrappers around a
small shared unrolling helper while retaining explicit public two-state and
four-state contracts.

The quantum wrapper must:

- accept batch-first input `[batch, sequence, input_size]`;
- initialize `h`, `c`, `n`, and `m` with
  `x.new_zeros(batch_size, hidden_size)` when no state is supplied;
- accept an explicit `(h, c, n, m)` initial state for chunked inference;
- unroll one time step at a time and collect `y_t`;
- return `outputs, (h_t, c_t, n_t, m_t)`;
- produce `outputs` with shape `[batch, sequence, output_size]`;
- reject malformed input/state shapes and a zero-length sequence with clear
  errors rather than failing later in `torch.cat`.

Update only the `qlstm` branch in `make_model` to use the four-state wrapper.
Continue using the two-state wrapper for the `lstm` branch.

### 5. Preserve and verify optional input projection

Keep the existing learned input projection behavior. When the source feature
width exceeds `input_projection_size`, the projection must run before the
Q-sLSTM wrapper, and the quantum cell must be constructed with the projected
width. Thus:

```text
n_qubits = effective_input_size + hidden_size
```

Do not project `h_t`, and do not construct the VQCs using the original input
width when projection is active. The wrapper must transparently return the same
output/state structure whether projection is active or not.

### 6. Expose epsilon configuration

Add a CLI argument such as:

```text
--gate_epsilon (float, default 1e-6)
```

Pass it from `make_model` into `CustomQLSTMCell`. Because experiment metadata
already serializes `args`, it should then appear automatically in the saved
configuration. Update help text to explain that epsilon bounds quantum
input/forget transforms and normalizer division.

Do not add the setting to the classical cell, and do not change existing model
names or checkpoint key layout unnecessarily.

### 7. Keep final-step training behavior simple and correct

Both wrappers return `(outputs, state)`, so existing trainer unpacking can remain
model-agnostic. Where the final prediction is selected, use the direct
batch-first expression:

```python
prediction = outputs[:, -1, :]
```

Replace the current transpose-then-index form in both training and prediction
logging, without changing target alignment or the return contract.

### 8. Clean only what this implementation touches

Remove unused heavyweight imports from `QLSTM_v0.py` when they are no longer
needed by that module, but do not perform unrelated repository-wide cleanup.
Use consistent four-space indentation in modified Python blocks. Keep public
imports compatible unless a deliberate replacement is made at all call sites.

## Tests to add

Create focused tests under `tests/`. Quantum-cell logic tests should replace the
four VQC modules with small deterministic Torch modules where practical so most
tests are fast and do not depend on repeated simulator execution. Include at
least:

1. **Log-gate bounds:** raw values at `-1`, `0`, `1`, slightly outside the
   physical interval, and close to both boundaries produce finite log gates
   inside the documented symmetric epsilon-derived bounds. The log-gate value
   at zero is approximately zero.
2. **Log-gate gradients:** values strictly inside the clamp interval have finite
   autograd gradients. Test at least a value close to `1 - epsilon`.
3. **Epsilon validation:** zero, negative, and values greater than or equal to
   one are rejected.
4. **Stabilized-gate bounds:** for varied `ell_i`, `ell_f`, and `m_prev`, the
   computed `i_prime` and `f_prime` are finite and in `[0, 1]` in floating-point
   arithmetic, and at least one is approximately one per element. Include cases
   where the input branch wins, the forget branch wins, and the two tie.
5. **Deterministic recurrence:** fixed VQC outputs and an explicit
   `(h_prev, c_prev, n_prev, m_prev)` match a hand-calculated single-step result
   for `m_t`, `i_prime`, `f_prime`, `c_t`, `n_t`, `h_t`, and `y_t`.
6. **State and output shapes:** a batch-first sequence returns
   `[batch, sequence, output_size]` plus four states of
   `[batch, hidden_size]`, with matching dtype and device.
7. **Chunked recurrence:** processing a sequence in two chunks while passing the
   first chunk's `(h, c, n, m)` into the second produces the same outputs and
   final state as processing the full sequence at once. This test must fail if
   `m` is reset between chunks.
8. **Backward stability:** a short Q-sLSTM forward/backward pass produces finite
   outputs, loss, input gradients, and parameter gradients.
9. **Projection construction:** when projection is active, the VQC qubit count
   or weight width reflects `projected_input_size + hidden_size`.
10. **Classical regression:** the standard LSTM branch still returns two states
   and retains its former output shape.
11. **Final-step extraction:** training and prediction logging select
    `outputs[:, -1, :]` correctly for batch sizes greater than one and
    multi-output predictions.

If the installed PennyLane version cannot broadcast exactly as expected, add
one small integration test around the real `VQC` to establish the supported
shape. Fix the batching conversion without falling back to a Python loop over
batch elements unless broadcasting is genuinely unavailable in the pinned
environment.

## Acceptance criteria

The implementation is complete when:

- the Q-sLSTM has four independent trainable VQCs and the exact stabilized
  `(h, c, n, m)` recurrence;
- input and forget VQC outputs become bounded `log1p` log gates that feed
  directly into the xLSTM maximum stabilizer, never the old sigmoid, an
  unbounded direct ratio, or `exp(ell_i)`/`exp(ell_f)`;
- the only input/forget exponentials are the stabilized `i_prime` and `f_prime`,
  and both are at most one;
- all normalizer/state tensors have the expected batch-first shapes, dtype, and
  device;
- the optional projection determines the actual quantum input width;
- final-step sequence-to-one prediction remains `[batch, output_size]`;
- the classical LSTM path still works with `(h, c)`;
- all new tests pass and normal test/smoke-test output contains no NaN or
  infinity.

Run at minimum:

```text
python -m pytest -q
python -m compileall -q src scripts
```

Also run a minimal CPU forward/backward smoke test for both `qlstm` and `lstm`
with batch size greater than one. Report the files changed, commands run, and
any environment limitation that prevented an acceptance check.
