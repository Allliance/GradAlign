# SynLogic size and noise baselines

This recipe adapts the released GradAlign selection loss to the project's
four-GPU VERL runtime. It also runs the matched GRPO baselines. Use the parent
project's [protocol](https://github.com/Allliance/noisy-rlvr-parent/blob/codex/qwen3-synlogic-mixtures-20261103/experiments/protocols/qwen3-synlogic-mixtures-20261103.md)
for frozen memberships, noise masks, and comparison limits.

Prepare inputs once from the existing audited SynLogic run store:

```bash
python recipe/synlogic/prepare.py --source "$SOURCE_RUN" --root "$RUN_ROOT"
```

The source must contain the original clean parquet, its membership manifest,
and version 4 verified synthetic labels. Preparation refuses an existing output
manifest. It does not generate labels or change the verifier.

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

GradAlign uses the released `select/grpo_loss.py`, full FSDP gradient shards,
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
SYNLOGIC_FRAMEWORK_CODE="$FRAMEWORK_CODE" \
  torchrun --standalone --nproc-per-node=4 recipe/synlogic/test_alignment.py
```
