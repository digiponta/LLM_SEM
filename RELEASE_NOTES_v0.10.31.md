# LLM_SEM v0.10.31

## Canonical Replay Protection

Protected replay no longer copies malformed source-model generations as training targets.

Previous behavior:

```text
actual /internal prompt
+ source-model generated answer
-> protected replay
```

This could freeze generation errors such as:

```text
...実現するなも電力です。
```

v0.10.31 changes the protected target to:

```text
actual /internal prompt
+ trusted canonical answer
-> protected canonical replay
```

The source-model generation is still printed for diagnostics but is not used as the teacher target.

### Training trace

Protected rows now show:

```text
query
source generated
canonical target
target kind = TRUSTED_CANONICAL
actual INTERNAL_PROBE prompt
```

### One-by-one sleep

The v0.10.30 one-by-one ACCEPT / ROLLBACK loop remains unchanged.

### Default candidate

```text
model/model-sem-sleep-v0131.pt
```

### Regression

```powershell
python run_canonical_replay_regression_v01031.py
```
