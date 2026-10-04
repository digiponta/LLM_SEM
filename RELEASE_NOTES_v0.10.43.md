# LLM_SEM v0.10.43

## Internal Probe Consistency

v0.10.42 exposed another runtime mismatch.

Before this version:

```text
/internal
  temperature=0.8
  top_k=40
  repetition_penalty=1.15
  max_new_tokens=80 (default)

/internal-batch / retention validator
  temperature=0.2
  top_k=1
  repetition_penalty=1.10
  max_new_tokens=96
```

Thus the same query could produce a noisy answer interactively while the validator reported a stable deterministic answer.

v0.10.43 gives `/internal` its own deterministic generation path matching the retention validator exactly.

Normal chat generation remains unchanged and can remain stochastic.

## Regression

```powershell
python run_internal_probe_consistency_regression_v01043.py
```

## Default candidate

```text
model/model-sem-sleep-v0143.pt
```
