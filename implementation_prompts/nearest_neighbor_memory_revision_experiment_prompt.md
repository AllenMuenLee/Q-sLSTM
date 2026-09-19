# Implementation Prompt: Nearest-Neighbor Memory-Revision Experiment

Implement one controlled synthetic experiment that tests whether the Q-sLSTM
can revise a stored value when a more relevant item arrives and preserve that
value through later distractors. The task is adapted from the xLSTM nearest-
neighbor search experiment described below.

This prompt assumes the Q-sLSTM architecture and stable input/forget gate
transform in `implementation_prompts/q-slstm_implementation_prompt.md` have
been implemented. If they have not, implement that prompt first. Reuse that
production Q-sLSTM. The comparison baseline is a conventional QLSTM with the
same four-VQC architecture, not a classical LSTM. If a production conventional
QLSTM is not present, add one as described below rather than substituting
`nn.LSTMCell`.

## Scientific question

Given a reference unit vector followed by candidate unit vectors with attached
scalar values, can the model:

1. replace its remembered value when a new candidate is more similar to the
   reference than every earlier candidate; and
2. retain the remembered value when later candidates are plausible but not
   better matches?

Compare conventional QLSTM (`qlstm`) against stabilized Q-sLSTM (`qslstm`)
under the same data, VQC architecture, hidden size, circuit depth, optimizer,
training budget, checkpoint-selection rule, and seeds. Do not include a
classical LSTM in this experiment. Report parameter counts and verify whether
the two quantum models are parameter matched.

The intended control is that both models use four independently parameterized
VQCs with identical input encoding, qubit count, circuit structure, circuit
depth, measurement layout, output layer, and parameter initialization. They
differ only in the gate transforms and recurrent update:

- **QLSTM:** conventional sigmoid input/forget gates and the conventional
  `(h, c)` LSTM-style recurrence.
- **Q-sLSTM:** bounded log gates, xLSTM stabilizer, and normalized
  `(h, c, n, m)` recurrence from the Q-sLSTM implementation prompt.

For each paired seed, initialize corresponding VQC and output-layer parameters
identically before training. Training then proceeds independently. Save a check
that the initial shared parameter tensors were equal, while never tying the two
models' live parameters together.

## Task definition and indexing

Make the sequence convention unambiguous. `sequence_length` means the total
number of recurrent input tokens, including the initial reference token. The
main study uses `sequence_length=32`, consisting of one reference token and 31
candidate tokens. Report both numbers in generated metadata so this cannot be
misread as 32 candidates.

For a sequence of length `L`:

- Sample a two-dimensional unit reference vector `r`.
- Token 0 is the reference token.
- Tokens `t=1,...,L-1` contain a two-dimensional unit candidate `u_t` and an
  attached scalar `v_t in [0, 1]`.
- Candidate similarity is `s_t = dot(r, u_t)`.
- Use a strict record rule. Candidate `t` is significant when
  `s_t > max(s_1, ..., s_(t-1))`; the first candidate is therefore significant.
- Let `j*_t` be the earliest index attaining the largest similarity through
  candidate `t`. The supervised target is `y_t = v_(j*_t)`.

Ties should be extremely rare, but resolve them in favor of the earliest
candidate and mark a later equal-similarity candidate as a non-event. Generate
targets from the computed similarities and running argmax, not from assumptions
made by a stress-case constructor.

### Input encoding

Use an input width of three:

```text
x[0] = [r_x,   r_y,   -1.0]       # reference token sentinel
x[t] = [u_t,x, u_t,y, v_t]        # t >= 1
```

The `-1` sentinel is outside the candidate-value interval and identifies the
reference token. It is not an event or similarity label. Do not append
similarity, running maximum, best index, event flag, target, case type, or any
other privileged feature to model input.

Represent the target at token 0 with a harmless placeholder such as zero. Use
two distinct masks so training supervision and reported evaluation are not
accidentally conflated:

- `loss_mask` is false at the reference token and true at every candidate,
  including the first candidate;
- `metric_mask` is false at both the reference token and the first candidate,
  and true from the second candidate onward.

The first candidate establishes the initial candidate memory. It is not a
revision between candidate data points, so its prediction must never contribute
to any reported metric, plot aggregate, event analysis, alpha analysis, or
model-comparison statistic. It may remain supervised during training through
`loss_mask`.

The dataset item should be a dictionary with at least:

```text
inputs:         [L, 3] float tensor
targets:        [L, 1] float tensor
loss_mask:      [L, 1] bool tensor
metric_mask:    [L, 1] bool tensor
event_mask:     [L, 1] bool tensor
similarities:   [L, 1] float tensor (placeholder at token 0)
best_indices:   [L] long tensor (-1 at token 0)
case_type:      string or stable integer label
sequence_id:    integer
```

Only `inputs`, and never the analysis fields, may be passed into a model.

## Dataset implementation

Add `src/q_slstm/datasets/nearest_neighbor.py`. Keep generation
synthetic, deterministic, and independent of the existing forecasting dataset
registry unless a clean reusable registration is helpful. A dedicated
experiment loader is preferred because this task returns per-timestep targets
and analysis metadata rather than the existing `(X, final_y)` contract.

Use a local `torch.Generator` (or NumPy generator, but not both without a clear
seed derivation) in every dataset constructor. Do not mutate global RNG state.
Materialize the generated tensors once at construction so repeated indexing and
different DataLoader worker counts cannot change samples.

### Unit-vector generation

For ordinary IID samples, draw angles uniformly from `[0, 2*pi)` and construct
unit vectors with `[cos(theta), sin(theta)]`. Draw candidate values independently
from `Uniform(0, 1)`.

For controlled stress cases, generate a candidate from a desired similarity
`s in [-1, 1]` without losing unit norm. If

```text
r_perp = [-r_y, r_x]
```

then use a random side sign and:

```text
u = s * r + sign * sqrt(max(0, 1 - s**2)) * r_perp
```

Normalize once after construction only to remove floating-point drift. Tests
must verify both reference and candidate norms.

### Case types

Implement the following four deterministic-by-seed generators. Make thresholds
configurable and save them with the run metadata. Use nonzero margins so strict
event/non-event invariants survive floating-point rounding.

1. **IID:** independently uniform reference and candidate angles; independently
   uniform candidate values.
2. **Late significant event:** force the global-best similarity to occur in the
   final quarter, but not necessarily at the final token. All earlier candidates
   must be at least `record_margin` below it. Leave at least one later distractor
   when possible so retention after the late revision can be measured.
3. **Early best followed by distractors:** force the global-best candidate into
   the first quarter. Every later candidate must remain at least
   `record_margin` below it.
4. **Near-best distractors:** place the global best in the first quarter, then
   make at least half of the subsequent candidates fall in a narrow similarity
   band below it, for example
   `[s_best - near_best_delta, s_best - record_margin]`. They must never tie or
   exceed the best candidate.

Use sensible defaults such as `record_margin=0.02` and
`near_best_delta=0.05`, validate that the configured intervals are feasible,
and fail clearly for incompatible settings.

Stress cases should make revision and retention failures observable. At a
forced revision event, ensure the new attached value differs from the previous
target by at least a configurable `value_separation` (for example `0.4`). For
near-best post-event distractors, preferentially sample values separated from
the retained target by the same amount. All values must still remain in
`[0, 1]`. Implement bounded conditional sampling directly or with a bounded
rejection loop that raises on failure; never use an unbounded retry loop.

After constructing inputs, always recompute similarities, event flags, running
best indices, and targets with one common target-generation function. The stress
constructors should be verified against these recomputed labels:

- late case has its global best in the configured late range;
- early and near-best cases have their global best in the configured early
  range;
- forced near-best distractors are non-events and inside the configured band;
- `target[t]` changes only at significant events (apart from coincidentally
  equal attached values in IID data).

## Data scale and splits

The paper-scale defaults are:

```text
training pool:        4,000 IID sequences
held-out total:       1,000 sequences
main sequence length: 32 total tokens
seeds:                5 (support 5-10 without code changes)
```

Split the 4,000-sequence training pool deterministically into 3,600 optimizer
samples and 400 validation samples. Use validation only for checkpoint
selection or early stopping. Never use held-out results for model selection.

Make the 1,000 held-out sequences a balanced, deterministic suite of 250 IID,
250 late-event, 250 early-best, and 250 near-best sequences. If a requested test
size is not divisible by four, distribute the remainder deterministically and
record the exact counts. Shuffle evaluation order only if `case_type` and
`sequence_id` remain available for regrouping.

Provide an optional length-64 extrapolation suite. It must be evaluation-only,
generated independently from training and validation, use the same four-case
balance, and default to a smaller documented count such as 250 total sequences.
Models evaluated on it must be the checkpoints trained only at length 32.

Expose a clearly named scale preset:

- `paper`: the defaults above and five seeds;
- `pilot`: a much smaller explicitly reported configuration for pipeline and
  VQC-cost checks.

Do not silently reduce sample counts or seeds. If VQC simulation cost requires
a smaller study, require the user to select the pilot preset or pass explicit
overrides, and label all output artifacts with the actual scale.

## Experiment entry point

Add a dedicated entry point such as:

`scripts/experiments/nearest_neighbor/train_nearest_neighbor.py`

Do not route this experiment through the existing final-step forecasting loss
in `src/q_slstm/trainers/train.py`. Refactor model construction into an
importable helper if needed so this script can construct the production QLSTM
and Q-sLSTM without importing a CLI script or duplicating model classes.

If the conventional QLSTM must be added, give it the same four VQCs and output
projection as Q-sLSTM and implement only this baseline recurrence:

```text
i_t = sigmoid(q_i)
f_t = sigmoid(q_f)
z_t = tanh(q_z)
o_t = sigmoid(q_o)
c_t = f_t * c_prev + i_t * z_t
h_t = o_t * tanh(c_t)
y_t = output_projection(h_t)
```

Its production recurrent state remains `(h, c)`. Do not add Q-sLSTM
normalization or stabilization to the QLSTM forward dynamics.

Support at least these arguments:

```text
--model {qlstm,qslstm}
--scale {paper,pilot}
--sequence-length 32
--train-size 4000
--test-size 1000
--extrapolation-length 64
--extrapolation-size 250
--run-extrapolation
--seed
--data-seed
--hidden-size
--qnn-depth
--gate-epsilon
--batch-size
--epochs
--lr
--weight-decay
--patience
--record-margin
--near-best-delta
--value-separation
--device
--save-dir
```

Set `input_size=3` and `output_size=1` internally and reject conflicting
overrides. Disable input projection for the main comparison because a width of
three is already small; save that choice in metadata. For both QLSTM and
Q-sLSTM, the expected qubit count is therefore `3 + hidden_size`.

Add a sweep launcher that runs QLSTM and Q-sLSTM for the same ordered list of seeds.
For paired fairness, a given seed must produce identical dataset tensors,
train/validation indices, DataLoader ordering, and case assignments for both
models. Derive separate, documented data, split, loader, and model seeds from
the run seed. Across repetitions, change the run seed; within a repetition,
pair the two models on the same data realization.

Record all arguments, resolved preset values, seed derivations, dataset sizes,
case counts, trainable parameter count, environment, and source revision in the
run directory. Save a dataset manifest containing generation settings and
stable checksums of the materialized split tensors so paired runs can be
verified without guessing from filenames.

## Training protocol

Both quantum models return `outputs` with shape `[batch, L, 1]`. Compute
unweighted per-candidate mean squared error:

```text
squared_error = (outputs - targets)**2
loss = squared_error[loss_mask].mean()
```

Do not include the reference token in training loss. The first candidate may be
included in training loss, but must be excluded later from every reported
metric through `metric_mask`. Do not weight events more heavily and do not use
event labels, similarity values, best indices, case labels, or alpha values in
the loss. They are for target construction checks and post-hoc analysis only.

Use the same Adam configuration, batch size, maximum epochs, gradient handling,
validation schedule, and early-stopping rule for both models. Save both the
last checkpoint and the checkpoint with the lowest validation MSE; evaluate the
predeclared best-validation checkpoint exactly once on held-out data. If
gradient clipping is needed, configure and apply the same rule to both models
and record gradient norms and clipping frequency.

Assert shapes before loss calculation. During training, detect non-finite
inputs, outputs, loss, and gradients and fail with the run/epoch/batch identity
rather than continuing with corrupted metrics.

The default study should use one fixed architecture configuration rather than a
large hyperparameter search. If any pilot tuning is performed, document the
search space and freeze the selected settings before running the final seeds.

## Write-proportion diagnostic

Do not report, aggregate, plot, or interpret the raw input-gate value `i_t` by
itself. A fixed input-gate magnitude is not comparable when the amount of
retained memory differs. Measure the proportion of post-update memory assigned
to the new write:

```text
n_after_t = f_t * n_before_t + i_t
alpha_t = i_t / clamp_min(n_after_t, epsilon)
```

Equivalently, if the implementation calls the incoming normalizer `n_t`, this
is the requested expression:

```text
alpha_t = i_t / (f_t * n_t + i_t)
```

Use explicit `n_before_t` and `n_after_t` names in code to avoid an indexing
ambiguity. `alpha_t` is the elementwise fraction of normalized memory mass due
to the current write, not the ratio of two batch-averaged quantities. It must be
computed per batch item and hidden coordinate before any reduction.

For the two models:

- **Q-sLSTM:** use its actual stabilized gates and normalizer,
  `i_t = i_prime`, `f_t = f_prime`,
  `n_after_t = f_prime * n_prev + i_prime`. Thus alpha is also
  `i_prime / clamp_min(n_t, epsilon)`.
- **QLSTM:** its forward recurrence has no normalizer. Maintain an
  analysis-only accumulator `n_diag`, initialized to zero and updated as
  `n_diag_after = f_t * n_diag_before + i_t` using its sigmoid quantum gates.
  Compute alpha from that accumulator. `n_diag` and alpha must not feed back
  into the QLSTM hidden state, cell state, output, loss, or gradients.

Expose gate telemetry through an optional diagnostics path, such as
`return_diagnostics=True` or a dedicated evaluation rollout, while preserving
the ordinary model forward contract. Compute raw gates only transiently. Do not
write raw `i_t` or `f_t` to result artifacts. Assert that finite alpha values are
in `[0, 1]` up to numerical tolerance.

For timestep-level analysis, first compute elementwise alpha and then report
its mean across hidden coordinates as `alpha_mean`; optionally save a separate
long-form per-hidden-unit alpha artifact. Never compute
`mean(i) / mean(f*n + i)` as a substitute. Apply `metric_mask`, so neither the
reference token nor first candidate contributes to alpha summaries.

## Evaluation and analysis

Evaluate with no gradient tracking and retain predictions and alpha diagnostics.
Rows may be retained for auditing at every timestep, but set `is_metric_step`
from `metric_mask` and use only true rows in every metric. Write a tidy
prediction artifact (CSV or Parquet) with at least:

```text
run_seed, model, split, case_type, sequence_id, sequence_length,
timestep, candidate_index, similarity, running_best_similarity,
is_event, is_metric_step, best_index, target, prediction,
absolute_error, squared_error, alpha_mean
```

Do not include raw `i_t` or `f_t` columns. Do not feed any analysis column back
into the model.

Compute metrics first per sequence, then aggregate per seed so long sequences or
sequences with more events do not silently dominate. Save raw per-sequence and
per-seed tables, not only final averages. Include:

1. overall MSE and MAE from the second candidate onward;
2. final-step MSE and MAE;
3. significant-event MSE and MAE, measured on the post-update output at the
   event timestep, excluding the first candidate's automatic event;
4. non-event/distractor MSE and MAE;
5. metrics separately for IID, late-event, early-best, and near-best cases;
6. retention drift on non-events, `abs(pred_t - pred_(t-1))`, especially after
   the global-best event;
7. revision gain at events,
   `abs(pred_(t-1) - target_t) - abs(pred_t - target_t)`, where larger positive
   values mean the output moved toward the new target;
8. event-aligned absolute error at the event and at fixed lags after it, with
   unavailable lags masked rather than filled with zero;
9. alpha at significant revision events and at non-event distractors, including
   their per-sequence means and event-aligned alpha curves;
10. alpha contrast, defined per sequence as mean alpha on eligible significant
    events minus mean alpha on eligible non-events, with missing groups reported
    as missing rather than zero.

Exclude the first candidate from all metrics, including ordinary prediction
error, final aggregates, event metrics, retention/revision metrics, and alpha
statistics. This exclusion is broader than only the metrics requiring
`pred_(t-1)`. The final-step metric remains valid when the sequence contains at
least two candidates. Report the number of eligible sequences, events,
non-events, and timesteps contributing to every aggregate after applying
`metric_mask`.

Aggregate the final comparison across paired seeds as mean and standard
deviation. Also report the per-seed paired Q-sLSTM-minus-QLSTM difference for
the predeclared primary metric, held-out MSE from the second candidate onward.
If confidence intervals are included, state the method and operate on
seed-level paired differences rather than treating timesteps as independent
samples.

Create only plots that support the memory-revision question:

- held-out MSE by case type and model with seed-level points;
- event-aligned error curves;
- event-aligned alpha curves and event-versus-distractor alpha summaries;
- error or retention drift by timestep for the early-best and near-best cases;
- optional fixed, preselected sequence traces showing similarity, event
  locations, target, and both model predictions.

Generate trace sequence IDs by a deterministic rule before viewing results; do
not cherry-pick visually favorable examples. Label the length-64 results as
extrapolation and keep them separate from the primary length-32 result.

## Files and artifacts

A clean implementation will likely add or modify:

- `src/q_slstm/datasets/nearest_neighbor.py`
- a matched conventional QLSTM cell and sequence wrapper under
  `src/q_slstm/models/` if they are not already present
- an importable model-construction helper under
  `src/q_slstm/models/` if one does not already exist
- `scripts/experiments/nearest_neighbor/train_nearest_neighbor.py`
- `scripts/experiments/nearest_neighbor/run_sweep.py`
- `scripts/experiments/nearest_neighbor/analyze_results.py`
- tests under `tests/datasets/` and `tests/experiments/`

Each run directory should contain:

- resolved configuration and seed map;
- dataset manifest/checksums and exact case counts;
- training and validation histories;
- last and best-validation checkpoints;
- per-timestep predictions;
- per-sequence and per-seed prediction and alpha metrics;
- parameter count and timing information;
- failure diagnostics if the run aborts.

The sweep-level analysis directory should contain the combined seed table,
paired comparison table, plots, and a concise machine-generated Markdown
summary suitable for checking the paper's reported numbers.

## Tests

Add fast tests that cover at least:

1. all reference and candidate vectors have unit norm within tolerance;
2. candidate values and targets stay in `[0, 1]`;
3. input tensors contain exactly three features and contain no copied event,
   similarity, best-index, or target channels;
4. similarities equal a direct dot-product recomputation;
5. event masks implement strict record events, including first-candidate and
   synthetic tie behavior;
6. targets equal values gathered from the recomputed running argmax;
7. `loss_mask` excludes exactly the reference token, while `metric_mask`
   excludes exactly the reference and first-candidate tokens;
8. every stress generator satisfies its location, margin, near-best, and value-
   separation invariants over many deterministic samples;
9. the same seed reproduces identical tensors and different seeds change them;
10. split IDs do not overlap and paired model runs use identical dataset
    checksums;
11. training loss ignores an arbitrarily bad reference-token prediction but
    still includes the first candidate, while every metric is unchanged by
    arbitrarily altering reference and first-candidate predictions;
12. metric functions return hand-calculated results on a tiny synthetic batch,
    including event gain and missing post-event lags, after excluding the first
    candidate;
13. alpha equals a hand-calculated elementwise
    `i_t / (f_t * n_before_t + i_t)`, lies in `[0, 1]`, and is reduced only after
    computing the elementwise ratio;
14. Q-sLSTM diagnostics use its actual `i_prime`, `f_prime`, and normalizer,
    while changing the QLSTM analysis-only normalizer cannot change QLSTM
    predictions or gradients;
15. no raw input/forget gate columns or raw-gate aggregate metrics are written;
16. the QLSTM and Q-sLSTM have the same VQC topology, qubit count, depth,
    measurement layout, output projection shape, and paired initial shared
    parameter tensors;
17. both quantum model wrappers accept `[batch, L, 3]`, return
    `[batch, L, 1]`, and complete a small finite backward pass;
18. a tiny CPU pilot run trains, selects a validation checkpoint, evaluates all
    four case types, and writes the required artifacts;
19. a length-32-trained checkpoint can evaluate length 64 without rebuilding or
    retraining the model.

Keep unit tests small by using tiny datasets and shallow models. Mark any
genuinely expensive VQC integration test so the ordinary test suite remains
usable, but do not replace all quantum-path coverage with mocks.

## Acceptance criteria

The experiment is complete when:

- the task and all stress cases are reproducible and satisfy their mathematical
  invariants;
- the model receives only the reference/candidate/value tokens;
- loss supervises every candidate timestep and excludes the reference token;
- every reported metric excludes both the reference and first candidate;
- QLSTM and Q-sLSTM runs are paired on identical data, corresponding initial
  quantum parameters, and training protocols, with no classical LSTM included;
- held-out data is not used for checkpoint selection;
- event, distractor, revision, retention, alpha, and case-specific metrics are
  saved without reporting raw input/forget gate values;
- the paper-scale preset is 4,000 training-pool sequences, 1,000 balanced held-
  out sequences, length 32, and at least five seeds;
- any reduced-cost run is explicitly labeled rather than silently substituted;
- optional length-64 evaluation uses only length-32-trained checkpoints;
- all tests pass, including a tiny end-to-end pilot.

Run at minimum:

```text
python -m pytest -q
python -m compileall -q src scripts
```

Then run one tiny CPU pilot for each model and the combined analysis command.
Report exact commands, elapsed time, artifacts produced, and any VQC runtime
constraint that would affect the paper-scale sweep.
