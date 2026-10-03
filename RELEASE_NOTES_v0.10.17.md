# LLM_SEM v0.10.17

## Multi-Knowledge Balanced Sleep

v0.10.16 successfully detected multi-knowledge failure:

```text
文学      PASS
量子暗号  FAIL
```

The second concept collapsed to the already dominant literature answer.

v0.10.17 changes sleep training from row-weighted mandatory QA to concept-balanced mandatory QA.

### Concept-balanced QA loss

Mandatory rows are grouped by their first semantic concept.

Instead of allowing a concept with many paraphrases to dominate the loss, each concept contributes equally:

```text
concept loss = mean(losses for all rows of one concept)
mandatory loss = mean(concept losses)
```

Optional/base QA remains as a smaller stabilizing term.

### Per-concept generation metrics

Fine-tuning now reports:

```text
concept_generation_mean
concept_generation_min
concept_generation_details
```

These are written to the iterative sleep result JSON.

### New sleep completion gate

A sleep round is complete only when both row-level and concept-level generation thresholds pass.

Default thresholds:

```text
row mean similarity       >= 0.80
row minimum similarity    >= 0.60
concept mean similarity   >= 0.75
concept minimum similarity>= 0.70
```

This specifically prevents a case where one concept scores very high while another concept collapses.

### Runtime

Default next candidate:

```text
model/model-sem-sleep-v0117.pt
```

### Regression

```powershell
python run_multi_knowledge_balanced_sleep_regression_v01017.py
```
