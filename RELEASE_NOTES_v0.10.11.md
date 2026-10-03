# LLM_SEM v0.10.11

## Iterative Sleep Consolidation

`/sleep` now repeats internal QA learning until internal-generation quality reaches a configurable completion threshold.

Default completion criteria:

```text
mandatory generation mean similarity >= 0.80
mandatory generation minimum similarity >= 0.60
semantic cosine >= 0.98
```

Default loop controls:

```text
max rounds        = 5
epochs per round  = 240
min improvement   = 0.01
max stall rounds  = 2
```

Each round starts from the checkpoint produced by the previous round:

```text
active/source
  -> round1
  -> round2
  -> ...
  -> completion
```

Round checkpoints and metrics are retained:

```text
model/model-sem-sleep-v0111.round1.pt
model/model-sem-sleep-v0111.round1.json
...
```

The final model is copied to:

```text
model/model-sem-sleep-v0111.pt
```

Promotion occurs only after:

1. Internal generation completion criteria pass.
2. Semantic cosine remains >= 0.98 during the iterative loop.
3. Semantic validation passes.
4. Consolidated retention passes.

If completion is not reached before the maximum rounds, or generation improvement stalls, `/sleep` stops without promoting the candidate.

Regression:

```powershell
python run_iterative_sleep_regression_v01011.py
```
