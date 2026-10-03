# LLM_SEM v0.10.26

## Pareto Incremental Sleep

v0.10.25 made runtime retention part of checkpoint selection, but 40-epoch chunks were too conservative. The final round remained almost identical to the source model and failed to learn the new concept:

```text
known/protected failures : 1
new/improvement failures : 4
canonical mean           : 0.607721
```

At the same time, v0.10.24 had shown that a longer protected-incremental run could get close to success:

```text
runtime probes passed : 8 / 9
canonical mean         : 0.846235
```

v0.10.26 therefore switches from sequential short checkpoints to an independent hyperparameter sweep from the same source model.

### Search space

The default sweep explores six candidates around the v0.10.24 near-pass region:

```text
epochs  lr-scale  distill
160     0.50      6
200     0.50      8
240     0.50      8
240     0.50      10
240     0.50      12
240     0.40      10
```

Each candidate starts from the original active model, not from another candidate.

### Evaluation

Every candidate is evaluated against the full mandatory runtime probe set:

```text
SemanticRouter
  -> selected label
  -> semantic-guided prompt
  -> deterministic generation
  -> runtime retention policy
```

A candidate is safe only when:

```text
known/protected failures == 0
```

Among safe candidates, the sweep prefers:
1. fewer new/improvement failures;
2. higher canonical mean similarity.

### Promotion rule

Only a candidate with:

```text
known failures = 0
new failures   = 0
total failures = 0
```

is copied into the final sleep candidate and allowed to continue through the remaining gates.

If no full-runtime PASS candidate is found, the source model stays active.

### Default candidate

```text
model/model-sem-sleep-v0126.pt
```

### Regression

```powershell
python run_pareto_incremental_sleep_regression_v01026.py
```
