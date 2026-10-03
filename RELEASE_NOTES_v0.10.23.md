# LLM_SEM v0.10.23

## Incremental New-Knowledge Sleep

v0.10.22 established retention-first/no-op sleep: if every mandatory concept is already internalized, balanced QA retraining is skipped.

v0.10.23 extends this to the mixed case:

```text
existing concepts PASS
new concept FAIL
```

Instead of retraining all mandatory knowledge, only the failed/new concepts are included in the mandatory QA training set.

### Flow

```text
/sleep
  -> build full canonical sleep dataset
  -> retention-first runtime precheck
       all PASS
         -> skip QA training
       some FAIL
         -> build incremental dataset
              protected PASS concepts removed from mandatory training
              failed/new concepts retained as must_train
              base/optional rows retained for stabilization
         -> train only incremental dataset
  -> selective surface repair on full knowledge set
  -> full dataset answer retention
  -> actual /internal runtime retention
  -> full multi-knowledge validation
  -> promote only if every gate passes
```

### Why

This protects already-internalized knowledge from unnecessary reconsolidation while allowing a new third, fourth, or later concept to be learned.

Example:

```text
文学      PASS -> protected
量子暗号  PASS -> protected
量子通信  FAIL -> training target
```

The new builder is:

```text
build_incremental_sleep_dataset_v01023.py
```

It consumes the v0.10.22 precheck JSON and creates an incremental QA dataset containing:
- all optional/base rows,
- mandatory rows only for failed/new concepts.

Final retention gates still use the original full dataset, so protected concepts must remain intact.

### Default next candidate

```text
model/model-sem-sleep-v0123.pt
```

### Regression

```powershell
python run_incremental_new_knowledge_sleep_regression_v01023.py
```
