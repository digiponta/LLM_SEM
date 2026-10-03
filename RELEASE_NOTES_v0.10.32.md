# LLM_SEM v0.10.32

## One-by-One Safe Retry Sleep

v0.10.31 showed that every one-by-one row was rejected, leaving the final candidate byte-equivalent in behavior to the source model.

v0.10.32 keeps the one-row-at-a-time policy but retries each row from the same committed source using several small hyperparameter settings.

### Per-row retry loop

For each new QA row:

```text
committed source
  -> try 1
  -> validate target + all protected probes
  -> try 2 from same source
  -> validate
  -> ...
  -> select best protected-safe candidate
  -> ACCEPT or ROLLBACK
```

No rejected try becomes the source of another try.

### Retry settings

```text
epochs  lr-scale  replay  new-weight
120     0.60      2       4
160     0.80      2       5
200     1.00      2       6
240     1.00      1       6
240     1.20      1       8
```

### Expanded protection

Previously, the precheck protected only the representative query for a passing concept.

v0.10.32 protects every mandatory runtime probe belonging to every PASS concept.

For the current dataset this means all existing 文学 probes plus 量子暗号 are protected before learning 量子通信.

### Acceptance

A try is eligible only when:
- target canonical similarity >= 0.70;
- target gain >= 0.10, or exact canonical match;
- no protected probe drops more than 0.05;
- every protected source/candidate similarity remains >= 0.70.

Among safe tries, the best target similarity is selected.

### Default candidate

```text
model/model-sem-sleep-v0132.pt
```

### Regression

```powershell
python run_one_by_one_safe_retry_regression_v01032.py
```
