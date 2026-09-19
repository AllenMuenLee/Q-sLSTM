# QLSTM Model Structure

## Overview

The model is a quantum-enhanced, normalized LSTM for sequence prediction. Each gate is produced by an independent variational quantum circuit (VQC), while an additional normalizer state controls the scale of the cell state.

```text
Input sequence [batch, sequence, input features]
                         │
                         ▼
                Optional input projection
                         │
                         ▼
              QLSTM cell, unrolled over time
                         │
                         ▼
          Per-step predictions [batch, sequence, outputs]
                         │
                         ▼
                Final-step prediction
```

## Sequence wrapper

The sequence model receives a batch-first tensor:

```text
x: [batch_size, sequence_length, input_size]
```

It maintains three recurrent states:

```text
h_t: hidden state [batch_size, hidden_size]
c_t: cell state   [batch_size, hidden_size]
n_t: normalizer   [batch_size, hidden_size]
```

All three states start as zero tensors when no initial state is supplied. The model processes the sequence one time step at a time and collects the output from every step.

The recurrent state passed between time steps is therefore `(h_t, c_t, n_t)`.

## QLSTM cell

At time step `t`, the current input and previous hidden state are concatenated:

```text
v_t = concat(x_t, h_(t-1))
```

The combined vector has width:

```text
number of qubits = input_size + hidden_size
```

Four independent VQCs receive the same combined vector. Each circuit has its own trainable parameters and produces `hidden_size` expectation values denoted by `q_t`.

```text
v_t = [x_t, h_(t-1)]
  |
  +-- Input-gate VQC  --> q_t^i --> (1 + q_t^i)/(1 - q_t^i) --> i_t
  +-- Forget-gate VQC --> q_t^f --> (1 + q_t^f)/(1 - q_t^f) --> f_t
  +-- Cell-gate VQC   --> q_t^g --> tanh(q_t^g)               --> g_t
  +-- Output-gate VQC --> q_t^o --> sigmoid(q_t^o)            --> o_t
```

Because a Pauli-Z expectation lies in `[-1, 1]`, the input and forget transforms produce nonnegative gate values on the open interval `(-1, 1)`. In computation, clamp the denominator away from zero with a small positive `epsilon`:

```text
gate(q) = (1 + q) / max(1 - q, epsilon)
```

The cell and normalizer states use matching input and forget gates:

```text
c_t = f_t * c_(t-1) + i_t * g_t
n_t = f_t * n_(t-1) + i_t
```

The hidden state is the normalized cell state controlled by the output gate:

```text
h_t = o_t * (c_t / n_t)
```

For numerical safety, the division also uses a small positive lower bound for `n_t`:

```text
h_t = o_t * (c_t / max(n_t, epsilon))
```

A classical linear layer maps the hidden state to the requested output size:

```text
y_t = Linear(hidden_size, output_size)(h_t)
```

The cell returns the prediction and all three recurrent states:

```text
(y_t, h_t, c_t, n_t)
```

## Variational quantum circuit

Each LSTM gate uses the same circuit architecture but has separate trainable weights.

### Circuit input

The circuit accepts the combined input and hidden state as rotation angles. Batched inputs are evaluated together through PennyLane parameter broadcasting.

### Circuit layers

For `n` qubits and depth `d`, the circuit is:

1. Apply a Hadamard gate to every qubit.
2. Encode the combined input with one `RY` rotation per qubit.
3. Repeat `d` variational layers:
   - Apply nearest-neighbor CNOT entanglement in two staggered passes.
   - Apply one trainable `RY` rotation per qubit.
4. Measure the Pauli-Z expectation value of the first `hidden_size` qubits.

```text
|0> ─ H ─ RY(input) ─ Entangle ─ RY(weights[0]) ─ ... ─ <Z>
|0> ─ H ─ RY(input) ─ Entangle ─ RY(weights[0]) ─ ... ─ <Z>
                         repeated for circuit depth
```

The trainable parameter tensor of each gate has shape:

```text
[circuit_depth, number_of_qubits]
```

Its values are initialized near zero. The quantum circuit uses PennyLane's `default.qubit` simulator and Torch interface so gradients flow through the circuit during backpropagation.

## Optional input projection

High-dimensional inputs can make the quantum circuit expensive because every additional input feature requires another qubit. When the original feature width exceeds a configured limit, a learned linear projection reduces it before the recurrent model:

```text
[batch, sequence, original_input_size]
                    │
                    ▼
Linear(original_input_size, projected_input_size)
                    │
                    ▼
[batch, sequence, projected_input_size]
```

The QLSTM then uses:

```text
number of qubits = projected_input_size + hidden_size
```

## Output structure

After processing the entire sequence, the model returns:

```text
outputs: [batch_size, sequence_length, output_size]
hidden:  [batch_size, hidden_size]
cell:    [batch_size, hidden_size]
normalizer: [batch_size, hidden_size]
```

For sequence-to-one forecasting, training uses the prediction from the final time step:

```text
prediction = outputs[:, -1, :]
```

## Classical baseline

The project can retain a standard `LSTMCell` as a classical baseline. It uses the same input/output dimensions, output projection, and final-step prediction, but keeps the classical two-state `(h_t, c_t)` recurrence and does not use the quantum gate transform or normalizer state.
