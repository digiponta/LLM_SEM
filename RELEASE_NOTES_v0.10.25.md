# LLM_SEM v0.10.25

## Runtime-Constrained Incremental Checkpoint Search

v0.10.24 reduced catastrophic forgetting substantially, but the final incremental candidate still failed one of nine actual `/internal` runtime probes.

Observed summary:

```text
Source/candidate mean similarity : 0.746480
Canonical mean similarity        : 0.618770 -> 0.846235
Runtime failures                 : 1 / 9
```

This means the new concept was learned strongly and most retained knowledge survived, but selecting only the final training state was still too coarse.

### v0.10.25 policy

Incremental learning is now split into short chunks.

Default:

```text
incremental chunk epochs = 40
max chunks               = 5
```

After every chunk the candidate is evaluated through the full actual runtime path:

```text
SemanticRouter
  -> selected label
  -> semantic-guided prompt
  -> deterministic generation
  -> full mandatory runtime probes
```

### Safe checkpoint rule

A checkpoint is considered safe only when:

```text
known/protected failures == 0
```

Among safe checkpoints the selector prefers:

1. fewer new/improvement failures;
2. higher canonical mean similarity when tied.

A checkpoint that damages protected knowledge is never used as the source for the next incremental chunk.

### Full-pass selection

As soon as a checkpoint reaches:

```text
known failures = 0
new failures   = 0
```

the incremental search stops and that checkpoint is selected.

If no checkpoint passes within the search budget, the source model remains active.

### Legacy repair disabled

Previous versions automatically invoked `retention_repair_loop_v01013.py` after a sleep failure. That could modify a candidate already rejected by runtime constraints and create `.repair1.pt`.

v0.10.25 disables that legacy automatic repair path. Runtime-constrained checkpoint search is now the sole selector for incremental learning.

### Default candidate

```text
model/model-sem-sleep-v0125.pt
```

### Regression

```powershell
python run_runtime_constrained_incremental_regression_v01025.py
```
