# LLM_SEM v0.10.36

## Latent-Progress Partial Commit Sleep

v0.10.35 still produced:

```text
Accepted rows: 0
Source/candidate similarity: 1.000000
```

This does not necessarily prove that optimization did nothing.

With deterministic top-1 generation, the correct token probability can improve while the generated string remains unchanged until an argmax boundary is crossed.

v0.10.36 therefore measures internal learning progress directly.

### New diagnostics

For every one-row fine-tuning attempt, the trainer now reports:

```text
Target QA NLL:
before -> after

Trainable parameter delta:
L2
relative L2
```

These values are also written to the per-try result JSON.

### LATENT progress

A row may now be committed even when runtime text similarity has not changed yet.

A LATENT commit requires:

```text
target NLL absolute drop >= 0.05
target NLL relative drop >= 5%
trainable parameter relative delta > 0
protected failures == 0
```

The existing visible runtime progress rule remains valid.

Thus:

```text
runtime canonical: 0.25 -> 0.25
target NLL       : 2.10 -> 1.65
protected        : safe

=> ACCEPT kind=LATENT
```

A later `/sleep` continues from that checkpoint. Repeated latent improvements can eventually cross the top-1 generation boundary and become visible runtime improvement.

### Why this matters

Previous versions judged progress only after text generation. That discarded useful sub-argmax learning.

v0.10.36 distinguishes:

```text
TARGET   final runtime threshold reached
PROGRESS visible runtime similarity improved
LATENT   target likelihood improved but argmax text has not changed yet
```

### Default candidate

```text
model/model-sem-sleep-v0136.pt
```

### Regression

```powershell
python run_latent_progress_commit_regression_v01036.py
```
