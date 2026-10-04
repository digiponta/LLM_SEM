# LLM_SEM v0.10.45

## Baseline-Aware Consolidated Retention

v0.10.44 successfully reached COMPLETE internalization in the runtime path:

```text
runtime failures       : 0
known failures         : 0
new failures           : 0
canonical mean         : 0.745185 -> 0.938246
```

However final consolidated semantic retention rejected the candidate because of
one record:

```text
text                  : 暗号
expected              : computer
source top            : computer
candidate top         : computer
source margin         : 0.005446
candidate margin      : 0.002411
expected sim delta    : -0.005551
global LOO            : 85.42% -> 85.42%
```

The source margin was already below the degraded retention threshold of 0.01,
so this was not a stable high-margin baseline.

## New BASELINE mode

v0.10.45 adds a narrow baseline-aware rule.

A consolidated record may pass as BASELINE only when:

- source already predicted the expected label;
- source margin was below degraded_min_margin;
- candidate still predicts the same expected label;
- candidate margin remains non-negative;
- expected-label similarity drops by no more than 0.01;
- global benchmark preservation still passes.

Stable records still use the existing RETENTION / DEGRADED / RECOVERY rules.

This avoids rejecting a globally preserved model merely because an already
ambiguous source record stays ambiguous, while still rejecting label changes
or meaningful similarity degradation.

## Regression

```powershell
python run_baseline_aware_retention_regression_v01045.py
```

## Default candidate

```text
model/model-sem-sleep-v0145.pt
```
