# LLM_SEM v0.10.7

## Sleep Retention Recovery

v0.10.6 could reject a candidate even when the candidate repaired a consolidated semantic item that the current source model misclassified.

Example observed during /sleep:

```text
expected=science
source_top=computer
candidate_top=science
candidate_margin=0.010451
expected_sim_delta=+0.007204
```

The old validator required `candidate_margin >= 0.02` for every item, so this improvement was marked FAIL.

v0.10.7 adds two validation paths:

```text
RETENTION:
  candidate_top == expected
  candidate_margin >= 0.02

RECOVERY:
  source_top != expected
  candidate_top == expected
  candidate_margin >= 0.005
  expected similarity gain > 0
```

The global benchmark preservation check remains mandatory in both cases.

Regression:

```powershell
python run_retention_recovery_regression_v0107.py
```
