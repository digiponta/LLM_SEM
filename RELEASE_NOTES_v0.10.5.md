# LLM_SEM v0.10.5

## Semantic Sleep Consolidation

Adds `/sleep` to `chat.py`.

The command moves ACTIVE Semantic Memory into the internal LLM through the existing validated lifecycle:

```text
ACTIVE
  -> TRAINING
  -> semantic-preserving internal training
  -> VALIDATING
  -> batch validation
  -> CONSOLIDATED / FAILED
  -> retention validation
  -> active-model promotion
```

Promotion is blocked if any target record fails validation or retention.

The internal training stage reuses:

- `semantic_memory_semantic_preserve_train_v078.py`
- `semantic_memory_batch_validate_v080.py`
- `consolidated_retention_v094.py`
- `promote_active_model_v094.py`

The new orchestration entry point is:

```text
semantic_sleep_v0105.py
```

Runtime:

```text
/sleep
```

Default candidate checkpoint:

```text
model/model-sem-sleep-v0105.pt
```

After successful promotion, restart `chat.py` so the promoted model in `model/active-model.json` becomes the runtime model.

Regression:

```powershell
python run_semantic_sleep_regression_v0105.py
```
