# LLM_SEM v0.10.24

## Protected Incremental Sleep

v0.10.23 successfully learned the new concept `量子通信`, but existing concepts were catastrophically forgotten.

Observed runtime:

```text
文学      0.937500 -> 0.645161
量子暗号  0.891566 -> 0.347826
量子通信  0.250000 -> 1.000000
```

This shows that removing retained concepts completely from incremental training was too aggressive.

v0.10.24 keeps the same incremental-learning principle but represents already-passing concepts as protected distillation anchors.

### Protected knowledge

Precheck PASS concepts are kept in the incremental dataset with:

```text
must_train = false
protected  = true
```

They are not trained again against the canonical answer. Instead, the source active model acts as a teacher and the candidate is penalized when its answer-token logits drift from the source model.

### Incremental objective

```text
total loss
  = new-concept QA loss
  + semantic preservation loss
  + protected source-logit distillation loss
```

Default protected distillation weight:

```text
8.0
```

### Reduced plasticity

Incremental learning also uses safer defaults:

```text
trainable Transformer blocks : 1
learning-rate scale          : 0.5
LM-head learning-rate scale  : 0.5
```

The final full-dataset retention/runtime/multi-knowledge gates are unchanged.

### Target behavior

```text
文学      -> protected
量子暗号  -> protected
量子通信  -> train target
```

The desired result is to learn `量子通信` without materially changing the two existing concepts.

### Default candidate

```text
model/model-sem-sleep-v0124.pt
```

### Regression

```powershell
python run_protected_incremental_sleep_regression_v01024.py
```
