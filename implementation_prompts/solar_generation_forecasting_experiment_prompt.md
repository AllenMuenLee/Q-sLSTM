# Implementation Prompt: Ontario Solar Generation Forecasting Experiment

Implement a reproducible hourly data collector and a controlled comparison of
conventional QLSTM (`qlstm`) and stabilized Q-sLSTM (`qslstm`) on Ontario solar
generation. Follow the organization and experiment workflow of
`implementation_prompts/nearest_neighbor_memory_revision_experiment_prompt.md`
and the **current implementation** under
`scripts/experiments/nearest_neighbor/` and
`src/q_slstm/experiments/nearest_neighbor.py`.

This document specifies work to implement; it does not report completed data
collection, training, or experimental results. Defaults below are proposed
study settings, not findings. Keep the nearest-neighbor experiment working.

## Scientific question

Does the stabilized, normalized Q-sLSTM recurrence improve next-hour solar
generation prediction relative to the conventional QLSTM recurrence when both
receive the same historical generation, weather, and calendar features?

Use the production quantum models and four independently parameterized VQCs
per recurrent cell. Pair data, initialization, optimizer, training budget,
checkpoint selection, and seeds. Compare only the two quantum architectures;
also report persistence forecasts as non-trained reference forecasts.

The primary outcome is held-out next-hour MSE in MWh squared. Do not assume
Q-sLSTM will outperform QLSTM. Report unfavorable and inconclusive results.

## Required data and feature contract

Collect all of the following. The 11 feature groups expand to **13 numeric
input columns** because each calendar group contributes two columns.

| # | Feature | Variable name | Unit | Source |
| --- | --- | --- | --- | --- |
| 1 | Historical solar generation | `solar_generation` | MWh per hourly interval | [IESO hourly generation](https://reports-public.ieso.ca/public/GenOutputbyFuelHourly/) |
| 2 | Global horizontal irradiance | `shortwave_radiation` | W/m² | [Open-Meteo archive](https://open-meteo.com/en/docs/historical-weather-api) |
| 3 | Direct horizontal solar radiation | `direct_radiation` | W/m² | Open-Meteo archive |
| 4 | Diffuse solar radiation | `diffuse_radiation` | W/m² | Open-Meteo archive |
| 5 | Cloud cover | `cloud_cover` | % | Open-Meteo archive |
| 6 | Air temperature | `temperature_2m` | °C | Open-Meteo archive |
| 7 | Relative humidity | `relative_humidity_2m` | % | Open-Meteo archive |
| 8 | Wind speed | `wind_speed_10m` | km/h | Open-Meteo archive |
| 9 | Precipitation | `precipitation` | mm per hourly interval | Open-Meteo archive |
| 10 | Hour of day | `hour_sin`, `hour_cos` | [-1, 1] | Derived |
| 11 | Day of year | `day_sin`, `day_cos` | [-1, 1] | Derived |

Use Open-Meteo for all eight weather variables in the main study. ECCC is an
optional separate adapter and sensitivity study, not a required dependency or
an automatic fallback. Do not silently mix station observations with reanalysis.

The IESO report's documented unit is MWh and its scope is registered Ontario
generators of at least 20 MW. Telemetry problems can affect aggregate values.
Preserve the source's quality indicators. Do not relabel these observations as
all Ontario solar production or household rooftop production. See the
[IESO report description](https://reports.ieso.ca/docrefs/helpfile/GenOutputbyFuelHourly_h1.pdf).

Open-Meteo archive data are reanalysis. Radiation fields are preceding-hour
means, precipitation is a preceding-hour sum, and the selected temperature,
humidity, cloud, and wind fields are instantaneous. `direct_radiation` is
horizontal radiation, not direct normal irradiance. See the
[official variable definitions](https://open-meteo.com/en/docs/historical-weather-api#hourly-parameter-definition).

## Data collector implementation

Implement a Python command-line collector using HTTP requests and structured
XML/JSON parsing. No browser automation is needed. Separate download, parsing,
normalization, validation, and dataset construction so tests can run offline.
Use standard-library HTTP/XML support or explicitly declare any new dependency.

### IESO generation adapter

1. Read the directory index and discover annual files matching
   `PUB_GenOutputbyFuelHourly_YYYY.xml` and their versioned counterparts. Annual
   files are listed in the [public directory](https://reports-public.ieso.ca/public/GenOutputbyFuelHourly/).
2. Select one authoritative snapshot per requested year. Prefer the unversioned
   annual file at collection time; if absent, choose the highest numeric
   revision for that year. Never concatenate revisions or add the rolling
   `PUB_GenOutputbyFuelHourly.xml` alias as another dataset.
3. Cache the exact response bytes before parsing. Save source URL, filename,
   revision/publication fields when present, retrieval time, response validators
   such as ETag/Last-Modified, and SHA-256. Subsequent offline builds use these
   pinned bytes, not whatever a mutable URL returns later.
4. Inspect a real response and commit a small attributed XML fixture before
   finalizing field paths. Parse namespaces explicitly and extract date,
   hour-ending number, and the output for fuel type `SOLAR`. Do not use fuel
   order, scrape rendered color as the sole quality signal, or confuse output
   with capacity, forecasts, storage, or another fuel's value.
5. Preserve source date/hour and any telemetry/status fields alongside parsed
   values. Document the source-to-column mapping. Missing solar fields are
   missing observations, never inferred zero production. Reject unexpected
   schema changes with a useful file/record error.
6. Keep the documented MWh values unchanged. If a different source schema
   explicitly supplies hourly average MW, require a documented adapter and
   convert with `energy_MWh = mean_power_MW * interval_hours`; do not infer
   units merely from numerically similar one-hour values.

### Open-Meteo weather adapter

Use `https://archive-api.open-meteo.com/v1/archive` with explicit latitude,
longitude, start/end dates, `timezone=UTC`, `models=era5`,
`temperature_unit=celsius`, `wind_speed_unit=kmh`,
`precipitation_unit=mm`, and the eight weather names above in `hourly`.
ERA5 is selected to keep the weather model fixed. Validate the returned units,
array lengths, timestamps, model metadata, and coordinates. Preserve the API's
resolved grid coordinates/elevation as well as the requested coordinates.
See the [archive API documentation](https://open-meteo.com/en/docs/historical-weather-api#api-documentation).

Make monthly requests per location with sufficient UTC padding for the IESO
intervals, then deduplicate shared boundaries and trim to the desired interval
range. Cache raw JSON and complete request parameters with checksums. Do not
use daily aggregates or the `_instant` radiation variants.

The generation target is geographically aggregated. Use a frozen, explicit
multi-location weather proxy rather than silently treating Toronto weather as
representative of all generation. Ship `configs/solar_generation/locations.csv`
with these proposed representative points and equal weights:

| location_id | Latitude | Longitude | Weight |
| --- | ---: | ---: | ---: |
| windsor | 42.3149 | -83.0364 | 0.2 |
| london | 42.9849 | -81.2453 | 0.2 |
| toronto | 43.6532 | -79.3832 | 0.2 |
| kingston | 44.2312 | -76.4860 | 0.2 |
| ottawa | 45.4215 | -75.6972 | 0.2 |

These are study-design proxy coordinates, not asserted solar plant locations or
capacity weights. Preserve each location's normalized series, then compute the
weighted mean of each weather variable at each timestamp to produce eight
weather columns. Require all configured locations to be valid; do not silently
renormalize weights when one is missing. Report spatial representativeness as
a limitation. A plant-capacity-weighted configuration is an optional separately
identified study and must cite the plant/capacity source and freeze weights
without fitting them to validation/test targets.

### Timestamp and interval convention

Use a unique, timezone-aware UTC **interval-end** timestamp as the join key.
Each generation row represents `(timestamp_utc - 1 hour, timestamp_utc]`.
Retain `interval_start_utc` for auditing.

IESO market-time conventions use EST; validate the report-specific convention
against its schema/display and fixtures. The implementation contract is fixed
UTC-05:00 for IESO report dates/hours, without daylight saving. Do not localize
raw market hours as `America/Toronto`. The general convention is documented in
the [IESO market manual, section 2.4](https://www.ieso.ca/-/media/Files/IESO/Document-Library/Market-Rules-and-Manuals-Library/market-manuals/day-ahead-commitment/MM9-dacp-manual.pdf).

For source date `D` and hour ending `H` in `1..24`:

```text
end_est   = midnight(D, fixed UTC-05:00) + H hours
start_est = end_est - 1 hour
timestamp_utc = end_est converted to UTC
```

Thus `2024-01-01, H=1` maps to `2024-01-01T06:00:00Z`, and `H=24`
maps to `2024-01-02T05:00:00Z`. The same offset applies in July. Test both
DST transition dates and leap day. If the verified source uses another hour
convention, stop and amend the explicit mapping rather than guessing.

Join the archive timestamp directly to this interval end: its preceding-hour
radiation/precipitation describes that same completed interval, while its
instantaneous variables describe the endpoint. Request the following UTC date
where needed to include the last EST day. Never join on an ambiguous local
date/hour string or introduce an extra one-hour weather shift.

Derive calendar inputs from the interval **start** in fixed EST, giving hourly
labels 0..23. For day-of-year `d` and days in that local year `D_year`:

```text
hour_sin = sin(2*pi*hour/24)
hour_cos = cos(2*pi*hour/24)
day_sin  = sin(2*pi*(d-1)/D_year)
day_cos  = cos(2*pi*(d-1)/D_year)
```

Use 366 in leap years and 365 otherwise; record this convention in metadata.

### Reliability, coverage, and provenance

- Set connect/read timeouts, bounded retries (five attempts), exponential
  backoff with jitter, and `Retry-After` handling for 429/temporary server
  failures. Fail immediately on permanent request/schema errors. Default to
  one download worker; make limits configurable and respect provider limits.
- Write cache files atomically. An interrupted download must not become a
  valid cache hit. Provide `--offline` and explicit `--refresh`; refresh creates
  a new snapshot instead of mutating the dataset used by existing runs.
- Build the full expected hourly grid before filtering. Report duplicates,
  missing hours, non-finite values, source quality flags, and per-source and
  per-location coverage. Conflicting duplicate values fail; identical
  boundary duplicates can collapse with an audit count.
- Treat valid nighttime zeros as observations. Require nonnegative generation,
  radiation, wind speed, and precipitation, and humidity/cloud in [0, 100].
  Quarantine invalid rows with reasons rather than silently clipping them.
  Temperature must be finite; document any plausibility checks separately.
- Default to no imputation. Retain missing/flagged rows in the quality report
  and exclude any model window that touches one. Never interpolate targets,
  backfill from future values, or compress time by deleting missing hours
  before constructing windows.
- Require at least 95% valid hourly rows in each chronological split and at
  least one usable window in each split; otherwise fail dataset preparation.
  Make the threshold configurable and record overrides. Report valid-window
  coverage too, since long windows amplify the effect of missing rows.
- Store attribution, source documentation links, collection dates, schema
  version, units, feature order, locations/weights, raw-file checksums,
  normalized-table checksums, quality policy, and preparation configuration
  in `dataset_manifest.json`. Generate an immutable `dataset_id` from these
  content/configuration identities, excluding volatile local paths/times.

## Forecast task and data leakage controls

Default task: given the preceding `L=32` consecutive hourly rows ending at
origin `t`, predict generation in the next hourly interval ending at `t+1`.
Stride is one hour. Reset recurrent state for every window.

```text
inputs  = features[t-L+1 : t+1]    # [L, 13], includes observed generation at t
target  = solar_generation[t+1]   # [1], strictly after all input timestamps
outputs, state = model(inputs[None, ...])
prediction = outputs[:, -1, :]    # [batch, 1]
```

Pass only the 13 declared columns to the model, in table order. Do not include
target-hour generation, target-hour observed weather, quality flags, split
labels, future ramps, or analysis metadata. Fit no transformation on held-out
rows. Known future calendar features are also excluded from this default
contract so every input row describes a completed historical interval.

This is a **retrospective forecasting benchmark** with historical reanalysis
and revised generation. Their availability/publication delays are not modeled;
do not claim the exact archive inputs were available in real time at the
forecast origin. An operational study requires issue-time weather forecasts
and generation vintages/latency, with a separate availability-aware manifest.

Dataset items contain `inputs [L,13]`, `target [1]`, `window_id`,
`origin_timestamp_utc`, `target_timestamp_utc`, and `split`. Analysis metadata
must remain separate from tensors passed to the model. A batch must not
broadcast a `[B]` target against a `[B,1]` prediction.

### Date range and chronological splits

Proposed paper data range: source dates **2024-01-01 through 2025-12-31**,
inclusive. Verify coverage before starting training; never silently substitute
other years. Fetch the extra UTC weather dates required by interval alignment.

Split the complete expected hourly grid chronologically into 70% optimizer
training, 10% validation, and 20% test. With `N` expected rows, boundaries are
`floor(0.70*N)` and `floor(0.80*N)`; save exact timestamp boundaries. Compute
them before quality filtering, identically for every seed and model. The
requested two-year grid contains 17,544 hours before quality exclusions.

Assign windows by target timestamp. Validation/test windows may use earlier
historical inputs across a split boundary, as these are past observations at
that forecast origin; labels never cross into another split. Later test origins
may use earlier observed test generation as lag input in a rolling one-hour
evaluation with fixed weights. Do not retrain on those observations. No
random window split and no seed-dependent resampling of real observations.

Fit per-column mean and standard deviation for the nine physical columns on
unique valid training rows only. Leave the four cyclic columns unchanged.
Use the training `solar_generation` mean/std for target scaling too. Handle a
zero variance with scale 1 and record it. Apply frozen transforms everywhere,
allow held-out scaled values outside the training range, and save the scaler
and fit-row checksum. Invert target scaling before reporting metrics.

## Model construction and paired fairness

Reuse `build_quantum_model` in `src/q_slstm/models/factory.py`, the production
cells, and `sequence_wrappers.py`. The existing factory constructs models
without input projection. Add an importable solar wrapper that applies the
same trainable `Linear(13, 3)` followed by `tanh` at every input timestep,
then calls the factory with `input_size=3`, `output_size=1`.

This preserves all 13 raw feature channels while reducing circuit input
width. With default `hidden_size=6`, each VQC has `3+6=9` qubits. Save both
`raw_input_size=13` and `quantum_input_size=3`, the projection activation,
and the qubit count; do not misreport the model as receiving only three raw
features. A no-projection study is optional and must report its larger qubit
count and cost separately.

The intended controlled difference is:

- QLSTM: sigmoid input/forget gates and conventional `(h,c)` recurrence.
- Q-sLSTM: the [unclipped polynomial update](q-slstm_polynomial_update.md)
  with normalized `(h,c,n,binary_scale)` state. Record its recurrence version
  in the study identity. Do not redefine its equations in the experiment.

Both models use identical VQC topology, encoding, measurement layout, depth,
hidden size, input projection, output head, and parameter initialization.
For each seed, initialize corresponding tensors exactly equally using isolated
RNG contexts; assert tensor equality and distinct storage, then train
independently. Include projection parameters in counts and initialization
checks. No live parameter sharing between models.

Reuse/extract the existing stable seed utilities without importing CLI scripts.
Record independent model/projection/loader seeds and deterministic epoch
permutations. The dataset and chronological split are fixed across all run
seeds. Preserve any `data_seed` field only as metadata marked inapplicable;
it must not randomize temporal splits. Paired runs share optimizer settings,
loader order, batch size, simulator configuration, and numerical precision.

## Scale presets and training protocol

| Setting | `paper` | `pilot` |
| --- | --- | --- |
| Source dates | 2024-01-01 to 2025-12-31 | 2024-06-01 to 2024-06-14 |
| Split | chronological 70/10/20 | chronological 70/10/20 |
| Sequence length | 32 | 8 |
| Forecast horizon | 1 hour | 1 hour |
| Input projection width | 3 | 3 |
| Hidden size | 6 | 2 |
| VQC depth | 3 | 1 |
| Batch size | 64 | 8 |
| Epochs | 100 | 2 |
| Adam learning rate | 0.01 | 0.01 |
| Weight decay | 0 | 0 |
| Gradient clipping | disabled | disabled |
| Paired run seeds | 5 | 2 |

Use every eligible window; no silent subsampling. Preset values are explicit
starting settings aligned with the existing experiment's architecture/training
defaults. Record every override and use a configuration hash in addition to
`paper-overridden`/`pilot-overridden` so distinct studies cannot overwrite one
another. Pilot results are pipeline/cost checks, not evidence about annual
forecast performance. If pilot tuning changes the study, freeze and document
the settings before the final seed sweep.

Train with mean squared error on the standardized **final output only**:

```text
pred = outputs[:, -1, :]
assert pred.shape == target.shape == (batch_size, 1)
loss = ((pred - target) ** 2).mean()
```

No synthetic reference token, first-candidate exclusion, or nearest-neighbor
event masks apply here. Every valid forecast target contributes. Aggregate
epoch MSE by sample count, including partial final batches.

Match the current nearest-neighbor runner: train the full epoch budget with
no early stopping; retain `last.pt` and the checkpoint with lowest validation
MSE, choosing the earliest epoch on exact ties. If clipping is enabled by an
override, apply the same rule to both models and log norms/clipping frequency.
Fail with run/epoch/batch identity on non-finite inputs, outputs, losses, or
gradients. Save configuration, scaler, dataset identity, and model state in
checkpoints, plus optimizer/RNG state in resumable training checkpoints.

Select the best checkpoint using validation alone, then evaluate test data once
with no gradient tracking. Predictions and diagnostics should be captured in
that same evaluation; subsequent analysis reads artifacts. Log wall time and
device/backend information. Do not claim a paper sweep completed if only a
pilot ran, and do not automatically downscale expensive quantum simulation.

## Evaluation and analysis

Save a tidy `predictions_test.csv` with at least:

```text
run_seed, model, dataset_id, window_id, origin_timestamp_utc,
target_timestamp_utc, horizon_hours, target_mwh, prediction_mwh,
absolute_error, squared_error, persistence_mwh, daily_persistence_mwh,
target_month, target_hour_est, is_daylight_proxy, alpha_final_mean
```

Save unmodified predictions; do not clamp negative predictions to zero for
the primary comparison. Report their frequency. Optional clipped metrics must
be clearly secondary and calculated identically for both models.

Report the following per seed, with eligible counts and physical units:

1. **Primary:** overall test MSE (MWh²).
2. MAE (MWh) and RMSE (MWh); optional R² is null when target variance is zero.
   Avoid MAPE because zero nighttime generation makes it undefined/unstable.
3. Metrics by target month, fixed-EST hour, and daylight proxy. Define the
   proxy in advance as aggregated target-hour GHI > 20 W/m². This future
   weather is used for retrospective grouping only, never as a model input
   or checkpoint-selection criterion. Report missing grouping metadata.
4. Hourly persistence `y_hat[t+1]=y[t]` and daily persistence
   `y_hat[t+1]=y[t+1-24]`. Missing baseline history remains missing, never zero.
   Compare baselines and both models on the same eligible target subset and
   report its size; retain full-set model metrics separately.
5. Optional large-ramp metrics: define a threshold from the 90th percentile
   of absolute consecutive-hour generation differences in training only.
   Freeze it and classify test ramps post hoc. Record threshold and counts.

There is one prediction per target hour, so aggregate across target rows within
each seed. Also save daily metrics for temporal inspection; do not silently
replace the primary hourly-weighted score with an unweighted average of days.
Across paired seeds, report mean and sample standard deviation and each
Q-sLSTM-minus-QLSTM MSE difference (negative favors Q-sLSTM). Require equal
target sets, configurations, data hashes, and initial parameter hashes before
accepting a pair. Flag failed/unpaired runs and disclose the requested and
completed seed counts. A one-seed pilot has undefined sample standard deviation.

If intervals are reported, identify the method and use paired seed-level
differences, not overlapping windows as independent samples. Seed variability
measures training randomness on this fixed time period; it does not establish
generalization uncertainty across years. Optional temporal block-bootstrap
analysis must be distinguished from seed variability.

### Write-proportion diagnostic

Reuse the existing optional `return_diagnostics=True` path. Q-sLSTM computes
elementwise `alpha = input_weight / n_after` using its actual binary-scaled
update, without an epsilon floor, before averaging hidden units. QLSTM uses
the analysis-only accumulator already implemented in its diagnostics path.
Diagnostics must not affect forward dynamics, loss, or gradients.

Save final-input-step alpha per window and optionally a tidy per-input-step
trace with `window_id` and input timestamp. Exclude the first input step from
aggregate alpha curves because its zero-initialized normalizer makes it
uninformative. This exclusion affects diagnostics only, not forecast metrics.
Never save or compare raw input/forget gate values. Weather/ramp associations
are exploratory; do not import synthetic event/revision semantics into this
forecasting task or claim alpha demonstrates causal memory improvement.

Create plots for seed-level model MSE, paired differences, train/validation
learning curves, monthly/daylight error, and actual-versus-predicted series.
For traces, select the first complete seven-day test block before inspecting
predictions and log its dates. Show both models, truth, and persistence on
identical axes. Add alpha plots only when they support a stated analysis.

## Files, entry points, and artifacts

Suggested implementation layout:

```text
configs/solar_generation/locations.csv
src/q_slstm/data_sources/ieso.py
src/q_slstm/data_sources/open_meteo.py
src/q_slstm/datasets/solar_generation.py
src/q_slstm/models/projected_quantum.py
src/q_slstm/experiments/solar_generation.py
src/q_slstm/experiments/solar_generation_metrics.py
scripts/experiments/solar_generation/collect_data.py
scripts/experiments/solar_generation/prepare_dataset.py
scripts/experiments/solar_generation/train_solar_generation.py
scripts/experiments/solar_generation/run_sweep.py
scripts/experiments/solar_generation/analyze_results.py
tests/data_sources/
tests/datasets/test_solar_generation.py
tests/experiments/test_solar_generation.py
tests/fixtures/solar_generation/
```

Keep scripts thin and library logic importable. Reuse generic training/artifact
utilities when suitable; do not pass this task through synthetic masks or
duplicate quantum model classes. Large downloaded/generated data and results
should be excluded from version control; commit small parser fixtures only.

Store data under `data/solar_generation/` with raw snapshots, normalized
per-source tables, quality reports, and processed datasets keyed by
`dataset_id`. Prefer CSV for normalized tables to avoid an implicit Parquet
dependency; Parquet is acceptable with an explicit dependency declaration.

Match the current nearest-neighbor sweep UX: **one model per invocation**, with
`--model`, `--scale`, `--seeds` or `--n-seeds/--master-seed`, `--workers`
(`--jobs` alias), `--resume`, and `--dry-run`. Use the same ordered seed list
for both model invocations. Each run executes in an isolated subprocess and
has a console log. Analysis combines both model directories afterward.

Collector arguments include `--scale`, `--start-date`, `--end-date`,
`--locations-file`, `--data-dir`, `--offline`, and `--refresh`. Preparation
accepts `--scale`, `--data-dir`, `--sequence-length`, `--min-coverage`,
and `--offline`, and writes `prepared_<scale>.json` pointing to the immutable
processed dataset. Reject incompatible scale/date/window settings unless
explicitly overridden. Training consumes `--dataset-manifest` and supports
`--model`, `--scale`, `--seed`, `--hidden-size`, `--projection-size`,
`--qnn-depth`, `--gate-epsilon`, `--batch-size`, `--epochs`, `--lr`,
`--weight-decay`, `--grad-clip`, `--device`, and `--save-dir`.
Fix horizon to one and raw input width to 13 for the main experiment; reject
conflicting overrides rather than accepting unused arguments.

Use this result layout, preserving the existing per-seed/per-model convention:

```text
results/solar_generation/<scale_label>/<study_id>/
  study_manifest.json
  seeds_qlstm.json
  seeds_qslstm.json
  seed_<seed>/<model>/
    config.json
    environment.json
    source_revision.json
    dataset_manifest.json
    scaler.json
    parameters.json
    init_parameters.json
    history.csv
    checkpoints/best.pt
    checkpoints/last.pt
    train_summary.json
    predictions_test.csv
    per_day_metrics_test.csv
    per_seed_metrics_test.csv
    alpha_metrics_test.csv
    timing.json
    console_log.txt
    complete.json                 # only after all required artifacts validate
    failure.json                  # on failure, no successful completion marker
  analysis/
    combined_seed_metrics.csv
    paired_comparison.csv
    pairing_checks.json
    plots/
    summary.md
```

Derive `study_id` from dataset identity and common resolved experiment settings,
excluding model name/run seed so paired runs share a study root. `--resume`
may skip a run only if completion, configuration, dataset, and artifact hashes
match; a file named `complete.json` alone is insufficient. Refuse accidental
overwrites of incompatible runs. Analysis must not pool different study IDs.

## Example workflow after implementation

These are required future CLI examples, not commands that exist or have run
as a result of writing this document. Run from the repository root:

```text
python scripts/experiments/solar_generation/collect_data.py --scale pilot --locations-file configs/solar_generation/locations.csv --data-dir data/solar_generation
python scripts/experiments/solar_generation/prepare_dataset.py --scale pilot --data-dir data/solar_generation --offline
python scripts/experiments/solar_generation/run_sweep.py --model qlstm --scale pilot --dataset-manifest data/solar_generation/prepared_pilot.json --seeds 0 1 --workers 1 --save-dir results/solar_generation
python scripts/experiments/solar_generation/run_sweep.py --model qslstm --scale pilot --dataset-manifest data/solar_generation/prepared_pilot.json --seeds 0 1 --workers 1 --save-dir results/solar_generation
python scripts/experiments/solar_generation/analyze_results.py --runs-dir results/solar_generation/pilot/<study_id>
```

Print the resolved study directory and exact analysis command at sweep
completion; replace `<study_id>` with that value. The paper workflow uses
`--scale paper`, `prepared_paper.json`, and five paired seeds such as
`--seeds 0 1 2 3 4` for both models. Acquisition and preparation happen once
per frozen study, not once per training seed.

## Tests

Add focused offline tests using small, attributed XML/JSON fixtures and
hand-calculated series. Cover:

1. Namespace-aware SOLAR extraction, fuel-order independence, source units,
   missing fields, telemetry flags, malformed payloads, and schema changes.
2. Numeric revision ordering, duplicate conflict handling, atomic cache
   recovery, bounded retries, offline cache misses, and manifest checksums.
3. Hour-ending 1/24, year boundaries, summer/winter fixed EST, both DST dates,
   leap day, and an explicitly matched generation/radiation interval.
4. Weather variable/unit validation, monthly request boundary deduplication,
   equal-weight aggregation, and missing-location behavior.
5. Exactly 13 ordered features and correct cyclic values, with separate
   metadata. Every input timestamp is earlier than its forecast target.
6. Chronological split/window membership, permitted historical boundary
   context, no overlapping target IDs across splits, and gaps invalidating
   windows rather than shortening time.
7. Scaler statistics remain unchanged after arbitrarily changing held-out
   values; constant features and inverse target transforms work correctly.
8. Identical data/scalers/orders and equal but independently stored initial
   parameters for paired models, including the input projection.
9. Real projected quantum forward/backward pass: `[B,L,13]` produces
   `[B,L,1]`, finite gradients reach the projection and VQCs, and final-step
   loss ignores earlier outputs without target-shape broadcasting.
10. Hand-calculated MSE/MAE/RMSE, partial batches, baseline eligibility,
    empty daylight/ramp groups, constant-target R², and paired differences.
11. Alpha diagnostics preserve predictions/gradients, reduce elementwise
    ratios correctly, and export no raw gate values.
12. Checkpoint selection uses validation only; resume verifies dataset/config
    identity; failures never count as successful paired results.
13. Tiny end-to-end offline pilots for both models, followed by analysis,
    produce internally consistent artifacts. Mark real VQC integration
    coverage `slow` where appropriate; retain existing experiment coverage.

## Acceptance criteria

The implementation is complete when the collector can download and rebuild a
frozen dataset offline, all required variables and units are present, temporal
alignment and quality exclusions are auditable, and both quantum models train
and evaluate on identical chronological data under the declared protocol.

All manifests, histories, checkpoints, physical-unit predictions, paired seed
metrics, diagnostics, plots, and a machine-generated Markdown summary must be
present and reproducible. Record actual data coverage, sample counts, seeds,
runtime, source revision/dirty state, and any departure from the defaults.
Passing a pilot completes pipeline validation; the paper experiment remains
incomplete until its full declared sweep and paired analysis finish.

Run at minimum after implementation:

```text
python -m pytest -q
python -m compileall -q src scripts
```

Then run data acquisition/preparation, one tiny CPU pilot per model, and the
combined analysis command. Report exact commands and artifacts, distinguish
fixture tests from live-source validation, and document any source outage or
simulation-cost constraint without fabricating data or results.
