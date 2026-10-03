# LLM_SEM v0.10.18

## Improvement-Aware Runtime Retention

v0.10.17 demonstrated that concept-balanced sleep can internalize a second concept while preserving the first, but the v0.10.15 runtime-retention rule rejected the improvement because it required the candidate answer to remain similar to the old source answer.

Observed case:

```text
query                  : 量子暗号とは
source canonical sim   : 0.164384
candidate canonical sim: 0.891566
canonical gain         : +0.727183
source/candidate sim   : 0.205128
```

The candidate was much closer to the canonical target, but the old answer-similarity rule treated the change as a regression.

v0.10.18 separates runtime probes into two modes.

### RETENTION mode

Used when the source already knows the target:

```text
source canonical similarity >= 0.70
```

Requirements:

```text
source/candidate similarity >= 0.70
canonical drop              <= 0.05
quality gates               PASS
```

### IMPROVEMENT mode

Used when the source does not yet know the target and the candidate clearly improves toward the canonical answer:

```text
source canonical similarity < 0.70
canonical gain              >= 0.10
candidate canonical sim     >= 0.70
quality gates               PASS
```

In IMPROVEMENT mode, low source/candidate similarity is expected and is not treated as a failure.

This allows acquisition of new internal knowledge while continuing to protect previously internalized knowledge.

Default next candidate:

```text
model/model-sem-sleep-v0118.pt
```

Regression:

```powershell
python run_improvement_aware_runtime_retention_regression_v01018.py
```
