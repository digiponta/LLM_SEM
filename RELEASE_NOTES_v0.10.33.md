# LLM_SEM v0.10.33

## Runtime-Aligned New-Knowledge Training

v0.10.32 still rejected every one-by-one attempt. The final candidate therefore remained behaviorally identical to the source model.

The remaining mismatch was on the NEW side.

Previously:

```text
training:
  dataset label
  gate=ACCEPT
  dataset-derived prompt

runtime validation:
  source SemanticRouter label
  gate=INTERNAL_PROBE
  actual runtime prompt
```

So the model was optimized under a prompt distribution different from the one used by `/internal`.

v0.10.33 aligns NEW training with the actual source runtime path.

For every `must_train` row:
1. fit the source-model SemanticRouter;
2. route the exact query;
3. extract the same intent/concepts as `/internal`;
4. build an `INTERNAL_PROBE` prompt;
5. train the canonical target answer on that exact prompt.

Protected rows continue to use canonical runtime replay from v0.10.31.

### Training trace

NEW rows now print:
- query
- canonical answer
- actual runtime-selected label
- exact runtime prompt used for training

### One-by-one safe retry

The v0.10.32 retry loop remains:
- one row at a time;
- up to five independent tries;
- protect all known probes;
- ACCEPT only protected-safe target improvement;
- otherwise rollback.

### Default candidate

```text
model/model-sem-sleep-v0133.pt
```

### Regression

```powershell
python run_runtime_aligned_new_training_regression_v01033.py
```
