# LLM_SEM v0.10.21

## Runtime-Path Selective Surface Repair

v0.10.20 introduced concept-by-concept optional surface repair, but its acceptance test still used `generation_similarity()`, which builds prompts from the fixed dataset label.

That can disagree with the real runtime path:

```text
actual SemanticRouter
  -> selected_label
  -> semantic-guided prompt
  -> generation
```

The observed v0.10.20 candidate reproduced the same runtime regression as v0.10.19:

```text
量子暗号 canonical similarity
source v0118 : 0.891566
candidate     : 0.787234
drop          : -0.104332
```

v0.10.21 changes selective-repair scoring to use the exact runtime path used by `/internal`.

### Acceptance rule

For each concept:

```text
runtime target gain >= 0.01
AND
maximum runtime drop on any other concept <= 0.02
```

If this condition fails, the repair is rejected and the previous model is kept.

### Runtime diagnostics

The repair selector now prints, per target concept:

```text
runtime query
selected label
route margin
canonical similarity
generated answer
```

This makes route-dependent regressions visible before promotion.

### Default next candidate

```text
model/model-sem-sleep-v0121.pt
```

### Regression

```powershell
python run_runtime_path_selective_surface_repair_regression_v01021.py
```
