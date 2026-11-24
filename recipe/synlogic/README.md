# SynLogic size and noise baselines

This recipe adapts the released GradAlign selection loss to the project's
four-GPU VERL runtime. It also runs the matched GRPO baselines. Use the parent
project's [protocol](https://github.com/Allliance/noisy-rlvr-parent/blob/6c7de737d471f8c89022b3ede627620607805bc3/experiments/protocols/qwen3-synlogic-mixtures-20261103.md)
for frozen memberships, noise masks, and comparison limits.

Inputs are the already prepared, frozen run stores and their audited input
contracts; the one-off preparation script is not part of this recipe. Nothing
here generates labels or changes a verifier.

Launch from an allocated, isolated four-GPU runtime with the matching backend
revision and a prepared run's `portable-provenance.json`:

```bash
python recipe/synlogic/run.py --root "$RUN_ROOT" \
  --framework-code "$FRAMEWORK_CODE" --condition mixture-gradalign --gpus 4
```

`--resolve-only` resolves the exact Hydra configuration without starting a
model. Other conditions are `noise30`, `noise10`, `small1-noise10`,
`mixture-sampling99`, and `mixture-gradient-equal`. Keep dependencies in the
pinned training environment rather than mixing the upstream repositories'
environments.

The builder's values (including `trainer.total_training_steps=970`) are the
historical 2026-11-03 replay settings. A new run must pass its approved,
explicit step budget, thinking mode and GPU settings as extra overrides and
check the `--resolve-only` output. Run provenance controls compatibility:

- The tokenizer's chat-template hash must equal the run's
  `prompt_template_sha256`; each pinned template keeps its own rendering check.
- `fresh_from_base=true` and `fresh_from_pretrained=true` both request a fresh
  start. An Instruct run may record `fresh_from_base=false` with
  `fresh_from_pretrained=true`; the reverse combination fails. Without either,
  OPD forks from `initial_checkpoint`.
- Validation IDs come from the input contract's `validation_source_ids`, or
  IDs 0-269 for contracts that predate that field.
- `trainer.label_only_reporting=true` is an opt-in, separately versioned report
  of only the `val-core` masks. It defaults to false.

## OPD and TIP-OPD

Native OPD uses `actor_rollout_ref.actor.opd_loss_mode=importance_sampling`:
the unclipped sampled reverse-KL teacher term beside the outcome PPO loss.
TIP-OPD (`opd.tip=true`) is a distinct mode and requires
`opd_loss_mode=tip_ppo`, a fixed teacher coefficient, its recorded
`opd.outcome_coefficient` and `opd.tip_retention`, preserved global minibatches
and `use_kl_loss=false`. The frozen teacher (`OPDReferenceWorker`) scores the
student head's full-vocabulary entropy and reverse KL on identical prefixes.
Selected tokens use dual-clip PPO, normalized by each optimizer minibatch's
global selected-token count; a minibatch without selected tokens contributes
zero teacher loss. Packed frozen-teacher sharing (`teacher_sharing.py`) is
optional and needs `SYNLOGIC_TEACHER_PACKED=1` plus the helper's pinned hash.

GradAlign deliberately uses the released `select/grpo_loss.py` variant:
per-response token means, summed upstream and rescaled here to a per-problem
mean. The logged reference norm uses this mean scale.
The upstream `select/parallel/grpo_loss.py` instead weights longer responses
more; that variant is not used here. FSDP distributes the chosen objective
without changing its reduction. Selection uses full FSDP gradient shards,
16 responses per reference/candidate prompt, a candidate-to-selection ratio
of 4, and selection every 10 training updates. The 156-prompt smaller training
pool supplies guidance; the larger training pool supplies candidates. Zero
gradients score zero and ties use the saved random ordering. Selection never
reads validation rewards. Ordinary training uses 8 responses per prompt.
Selection has substantial extra generation/gradient cost; report its GPU time
alongside the matched number of optimizer updates.

The gradient baseline weights each response by `15610/(2*subset_size)` under
natural pool sampling. Weights apply before the original loss denominator,
including KL/entropy, so microbatching cannot cancel the intended scaling.

Complete saves include all four model/optimizer shards, scheduler/RNG state,
and sampler or selected-curriculum state. Validate with
`python recipe/synlogic/validate_checkpoint.py CHECKPOINT` before recovery.
The launcher honors root/run `STOP` files and preserves logging identities.
Resource holders, monitoring, and asynchronous checkpoint publication belong
to the execution environment and are separate from this scientific recipe.

Portable checks:

```bash
python recipe/synlogic/test_sampling.py
PYTHONPATH="recipe:$FRAMEWORK_CODE:$FRAMEWORK_CODE/SynLogic" python -m pytest \
  recipe/synlogic/test_parallelism.py recipe/synlogic/test_tip.py \
  recipe/synlogic/test_teacher_sharing.py recipe/synlogic/test_provenance_compat.py
SYNLOGIC_FRAMEWORK_CODE="$FRAMEWORK_CODE" \
  torchrun --standalone --nproc-per-node=4 recipe/synlogic/test_alignment.py
```

Existing runs must keep their frozen runtime and recorded parser, thinking mode,
validation scope, and objective. This canonical recipe preserves the backend
main's native answer parsing. Older branch runtimes used different answer parsing
and verifier deadlines; moving an existing run requires an explicit protocol
compatibility decision. Historical TIP used an ambiguous `importance_sampling`
mode name; do not silently resume it as `tip_ppo` or rewrite its provenance.
This consolidation does not qualify GPU placement or authorize a launch. The
legacy teacher worker uses one GPU; new Yale OPD placements still require the
user's TP2/co-allocation policy and measured fit qualification.
