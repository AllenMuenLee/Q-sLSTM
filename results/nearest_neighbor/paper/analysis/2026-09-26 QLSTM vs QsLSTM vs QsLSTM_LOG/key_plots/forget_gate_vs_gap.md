# Forget gate vs signed similarity gap

All 20 paired seeds, all test sequences, best-validation checkpoints from ../sources.json.

Gap = current similarity minus the running best BEFORE the current step. Eligibility and bins match revision_gain_vs_gap.csv: both this step and its predecessor must be metric steps.

The plotted quantity is the transformed forget gate before numerical stabilization/scaling, not the raw circuit expectation q and not the effective (stabilized/scaled) weight the recurrence applies. QLSTM: sigmoid(q); amplified QsLSTM: exp(log(1+q)-log(1-q)) with the saved epsilon clamp; amplified QsLSTM_LOG: ln(2/(1-q)); runs with a sigmoid-forget recurrence: sigmoid(q). Amplified per model: {'qlstm': False, 'qslstm': True, 'qslstm_log': True}. Different gate scales do not directly imply different realized memory writes.

Average over hidden units per step, then eligible steps per bin within each seed. Lines average these seed means; bands show sample standard deviation across seeds.

Checkpoint replay used the actual model forward pass with an observation-only forget-gate hook. All regenerated targets and saved predictions were checked. Maximum prediction difference: 1.78813934e-05 (absolute tolerance 2e-5). Bin counts match the existing revision plot.
