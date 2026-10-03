# LLM_SEM v0.10.20

## Selective Surface Repair

v0.10.19 showed that unconditional surface repair could make an already-good multi-knowledge model worse.

Observed regression:

```text
量子暗号 canonical similarity
v0.10.18 source : 0.891566
v0.10.19 repair : 0.787234
drop            : -0.104332
```

The repaired candidate also reintroduced literature-specific material into the quantum-cryptography answer.

v0.10.20 changes surface repair from mandatory replacement to optional concept-by-concept refinement.

### Per-concept repair candidate

Each repairable concept is trained independently.

A repair candidate is evaluated against:
- the target concept before/after,
- every other internally learned concept before/after.

### Acceptance rule

A repair is accepted only when:

```text
target canonical gain >= 0.01
AND
maximum drop on any other concept <= 0.02
```

Otherwise that concept repair is rejected and the current model is kept.

### Automatic rollback

The repair stage can now produce:

```text
ACCEPT
REJECT
KEEP_SOURCE
```

If no candidate improves safely, the original balanced-sleep model is copied forward unchanged.

### Training scope

Selective repair still trains only:

```text
final_norm
lm_head
```

with a lower default learning rate:

```text
1e-6
```

and stronger semantic preservation weight:

```text
8.0
```

### Sleep pipeline

```text
Balanced Sleep
  -> Selective Surface Repair
       -> per-concept candidate
       -> before/after comparison
       -> ACCEPT or rollback
  -> Semantic Retention
  -> Dataset Answer Retention
  -> Improvement-Aware Runtime Retention
  -> Multi-Knowledge Internalization
  -> Promotion
```

Default next candidate:

```text
model/model-sem-sleep-v0120.pt
```

Regression:

```powershell
python run_selective_surface_repair_regression_v01020.py
```
