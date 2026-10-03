# LLM_SEM v0.10.29

## Sleep Training Text Trace

This release makes the actual `/sleep` training content visible in the console.

The log now distinguishes three categories:

```text
[NEW / MUST_TRAIN]
[PROTECTED / RUNTIME_REPLAY]
[OPTIONAL / BASE_STABILIZATION]
```

For new knowledge the log prints:
- concept
- label
- sleep weight
- query
- target answer

For protected knowledge the log prints:
- runtime-selected label
- query
- source-model replay answer
- exact INTERNAL_PROBE prompt used for replay distillation

For optional/base stabilization rows the log prints:
- label
- source
- sleep weight
- query
- answer

The incremental dataset builder also prints every selected row with `[NEW]`, `[PROTECTED]`, or `[OPTIONAL]` so it is possible to verify the dataset before training begins.

### Default candidate

```text
model/model-sem-sleep-v0129.pt
```

### Regression

```powershell
python run_sleep_training_trace_regression_v01029.py
```
