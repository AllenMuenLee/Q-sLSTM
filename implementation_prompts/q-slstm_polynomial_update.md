# Q-sLSTM polynomial update: current recurrence

This specification supersedes the clipped log-gate update in
`q-slstm_implementation_prompt.md` and direct-ratio implementation in
`q-slstm_model_structure.md`. The version is `polynomial_binary_scale_v2`.

## Algebra

For unscaled memory `C`, normalizer `N`, and candidate `z`, define
`f=(1+q_f)/(1-q_f)` and `i=(1+q_i)/(1-q_i)`. Normalized memory is
`(f*C+i*z)/(f*N+i)`; the output gate multiplies this afterward.

The requested expression is correct for `-1 < q_i,q_f < 1`:

```text
A = (1-q_f^2) * (1-q_i)^2
B = (1-q_i^2) * (1-q_f)^2
normalized_memory = (A*C + B*z) / (A*N + B)
```

Cancel the common `(1-q_f)*(1-q_i)` factor before evaluation to avoid
unnecessary small products and cancellation in `1-q^2`:

```text
a = (1+q_f)*(1-q_i)
b = (1+q_i)*(1-q_f)
d = (1-q_f)*(1-q_i)
normalized_memory = (a*C + b*z) / (a*N + b)
C_next = (a*C + b*z) / d
N_next = (a*N + b) / d
```

Although `d` cancels from the current output, account for it in the next
recurrent state. Storing just the numerator and denominator changes future
write weights. For example, `q_i=.5`, `q_f=.2`, empty initial memory, and
successive candidates `1/3,-1/3` give `1/15` at step two. Naively carrying the
squared-polynomial numerator/denominator instead gives about `-0.204301`.

## Representation without clipping or logarithmic gates

Store `(h,c,n,scale)` where `C=c*2**scale`, `N=n*2**scale`. Initialize all
four tensors to zero; exponent zero means unit scale. Never materialize the
unscaled states during a rollout. The fourth tensor is an integer-valued
binary exponent, replacing the old logarithmic stabilizer `m`.

Split `a,b,d` into mantissas/exponents using `frexp`. Include the previous
state's exponent in the retained-memory branch. Use the larger active branch
exponent as the next scale, and shift mantissa ratios with `ldexp`. Nonzero
branch mantissas receive nonpositive shifts. Zero branches do not determine
the common scale.

Gate scaling alone does not bound the stored states: fractional gate mantissas
can still compound across timesteps. Normalize both incoming states together
before multiplication, and both outgoing states together after addition, using
the exponent of `max(abs(c), n)` from `frexp`. Add each removed common exponent
to the carried scale. For valid memory and bounded candidates, the stored
normalizer is then in `[0.5, 1)` and `abs(c) <= n`; intermediate gate weights
are below 2 and intermediate state magnitudes are below 4. This bounds the
stored mantissas without changing the unscaled recurrence or clipping gates.
The unscaled `C,N` may still be arbitrarily large. The scale exponent has a
finite supported range and raises explicitly if exceeded.

```text
c_next = forget_weight*c + input_weight*z
n_next = forget_weight*n + input_weight
h_next = sigmoid(q_o) * (c_next/n_next)
alpha = input_weight/n_next
```

The displayed weights include the common state rescaling when used for
diagnostics; scale `input_weight` together with `n_next` to preserve alpha.

No expectation clipping, denominator floor, or log/exp gate transform is used.
Alpha uses the actual denominator and is detached for analysis. The legacy
`gate_epsilon` argument is accepted/validated for configuration compatibility
but does not affect this update. Old log helpers remain for historical work.

The installed Torch's integer-exponent `ldexp` backward returns zero for
negative shifts. An explicit backward operation applies the same binary shift
to the upstream gradient. Multi-step output AND gradient comparisons against
independent unscaled arithmetic test this behavior.

## Domain and limits

Non-finite/out-of-range expectations fail explicitly. `q=+1` is singular in
the original gate and is rejected. One-step limits of the cancelled expression
do not generally define a finite recurrent normalizer; simultaneous boundary
limits can also depend on the approach path. No epsilon fabricates a result.

`q=-1` is a zero gate and is supported if the updated normalizer is positive.
No retained or new memory gives `0/0` and raises. Negative normalizers and
inconsistent empty states also fail. Near-boundary expectations remain
unchanged. Finite precision can still underflow negligible branch weights;
this does not guarantee arbitrary precision or bounded boundary gradients.

The initial normalizer remains zero. With a nonzero first write, normalized
memory equals `z`; this follows from the recurrence in every representation.

## Compatibility and validation

Parameters, VQC topology, and model sizes are unchanged. Old learned weights
can start a new rollout, but clipping-dependent old results are not reproduced.
Never pass old live `(h,c,n,m)` states to the new recurrence; reset the state.

Experiment configurations record the version. Solar study identities include
it; nearest-neighbor runs reject overwriting/resuming another version and
analysis rejects mixed versions. Keep existing result artifacts unchanged.
Version 2 adds the common incoming/outgoing state normalization; version 1
scaled only the gate branches and could still overflow or underflow its stored
states. Re-evaluate old weights in a fresh version-2 run rather than relabeling
version-1 artifacts or reconstructing version-1 traces with the corrected kernel.

Tests cover the expanded fraction, multi-step outputs/gradients, rescaled and
chunked states, tiny first writes, singularities, zero gates, real VQC backward
passes, and a high-precision reference whose raw normalizer exceeds float64
range before later forgetting restores the influence of new writes.
