# LLM_SEM v0.10.14

## Answer-Preserving Retention Repair

v0.10.13 restored semantic retention, but internal answer generation could regress after the repair stage.

Observed symptom:

```text
/internal 文学とは
semantic routing retained/repaired
but generated wording degraded
```

v0.10.14 introduces two protections.

### 1. Answer-preservation loss during semantic repair

`retention_repair_v01013.py` now loads mandatory sleep QA rows and adds their answer LM loss to the repair objective.

The repair objective is now conceptually:

```text
semantic target repair
+ benchmark preservation
+ consolidated semantic preservation
+ mandatory answer preservation
```

The LM head remains frozen. The final Transformer block and final_norm are the only trainable semantic stages.

### 2. Dual retention gate before promotion

A repaired checkpoint must now pass both:

```text
Semantic Retention
AND
Answer Retention
```

`answer_retention_v01014.py` compares mandatory internal generation before and after repair.

Default preservation limits:

```text
mean similarity drop <= 0.03
per-item similarity drop <= 0.05
abnormal-character ratio <= 0.02
repetition ratio <= 0.20
natural sentence termination required
```

If Semantic Retention passes but Answer Retention fails, the repaired checkpoint is not promoted.

Default next sleep candidate:

```text
model/model-sem-sleep-v0114.pt
```

Regression:

```powershell
python run_answer_preserving_retention_regression_v01014.py
```
