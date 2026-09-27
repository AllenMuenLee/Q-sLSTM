# Input gate vs signed similarity gap

All 20 paired seeds, all test sequences, best-validation checkpoints from ../sources.json.

Gap = current similarity minus the running best BEFORE the current step. Eligibility and bins match revision_gain_vs_gap.csv: both this step and its predecessor must be metric steps.

The plotted quantity is the transformed input gate before numerical stabilization/scaling, not the raw circuit expectation q and not the effective write fraction alpha. QLSTM: sigmoid(q); QsLSTM: exp(log(1+q)-log(1-q)) with the saved epsilon clamp; QsLSTM_LOG: ln(2/(1-q)). Different gate scales do not directly imply different realized memory writes.

Average over hidden units per step, then eligible steps per bin within each seed. Lines average these seed means; bands show sample standard deviation across seeds.

Checkpoint replay used the actual model forward pass with an observation-only input-gate hook. All regenerated targets and saved predictions were checked. Maximum prediction difference: 1.78813934e-05 (absolute tolerance 2e-5). Bin counts match the existing revision plot.
