# LLM_SEM v0.10.27

## Runtime-Replay Protected Incremental Sleep

v0.10.26 confirmed that hyperparameter search alone could not solve the remaining interference.

The repeated failure pattern was especially informative:

```text
文学      retained
量子暗号  repeatedly regressed
量子通信  often remained unlearned
```

The key mismatch was in the protection objective.

Previous protected distillation used the QA dataset's stored label and canonical-answer teacher-forcing prompt. Actual `/internal` generation instead uses:

```text
SemanticRouter
  -> runtime selected label
  -> INTERNAL_PROBE semantic prompt
  -> deterministic source-model generation
```

Therefore the protected loss was preserving a different prompt distribution than the runtime retention gate evaluated.

### v0.10.27 change

Protected knowledge is now replayed from the source model's actual runtime behavior.

For each protected row:

1. fit the source-model SemanticRouter;
2. route the exact query;
3. construct the same `INTERNAL_PROBE` prompt used by `/internal`;
4. generate the source-model answer deterministically;
5. cache that prompt + generated answer;
6. distill candidate logits against the source model on that replay sequence.

Thus:

```text
protected knowledge:
  source runtime prompt
  + source runtime answer
  -> replay distillation

new knowledge:
  taught canonical answer
  -> QA learning
```

This directly protects the behavior that the runtime gate later measures.

### Pareto sweep update

The sweep is re-centered for runtime replay:

```text
epochs  lr-scale  replay weight
240     0.50      2
240     0.50      4
320     0.50      4
320     0.50      6
480     0.40      4
480     0.50      6
```

Only a full 9/9 runtime PASS candidate can proceed to promotion.

### Default candidate

```text
model/model-sem-sleep-v0127.pt
```

### Regression

```powershell
python run_runtime_replay_incremental_regression_v01027.py
```
