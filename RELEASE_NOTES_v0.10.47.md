# LLM_SEM v0.10.47

## Concept Bootstrap for Fully Unseen Knowledge

v0.10.46 successfully preserved and consolidated previously learned concepts, but a new reproduction test with `量子センサー` showed no safe row-level progress:

```text
accepted rows : 0
rejected rows : 4
state         : UNLEARNED
```

The model repeatedly answered with the nearest known concept (`量子通信`).

## New pipeline

```text
incremental failed concept
        ↓
Concept Bootstrap
  train all QA paraphrases for the same concept together
        ↓
runtime safety check
        ↓
safe progress ?
  yes → bootstrap checkpoint
  no  → original source
        ↓
existing one-by-one refinement
        ↓
retention / internalization gates
```

Concept Bootstrap is only attempted for unseen concepts with at least two mandatory QA rows. Existing protected rows remain replay anchors.

A bootstrap candidate is accepted only when:

- known/protected runtime failures remain zero;
- mean canonical similarity for the target concept improves by at least 0.05.

If no safe progress is found, the original source model is retained and the existing one-by-one process is used unchanged.

## Parameters

```text
epochs                    : 180
new knowledge weight      : 6.0
protected distill weight  : max(4.0, incremental_distill_weight * 0.5)
train blocks              : incremental_train_blocks
```

## Regression

```powershell
python run_concept_bootstrap_regression_v01047.py
```

## Default candidate

```text
model/model-sem-sleep-v0147.pt
```
