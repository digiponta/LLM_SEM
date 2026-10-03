# LLM_SEM v0.10.10

## Degraded Retention

Adds a third retention path for consolidated knowledge that remains correctly classified but with a reduced margin.

Validation modes:

- RETENTION: candidate keeps the expected label with margin >= 0.02.
- DEGRADED: source and candidate both keep the expected label with margin >= 0.01.
- RECOVERY: source was wrong, candidate becomes correct with margin >= 0.005 and positive expected-similarity gain.

The global benchmark preservation check remains mandatory.

Observed case now treated as DEGRADED PASS:

```text
expected=computer
source_top=computer
candidate_top=computer
candidate_margin=0.012957
expected_sim_delta=-0.042809
```

Regression:

```powershell
python run_degraded_retention_regression_v01010.py
```
