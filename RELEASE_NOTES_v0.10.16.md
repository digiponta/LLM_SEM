# LLM_SEM v0.10.16

## Multi-Knowledge Internalization

v0.10.15 demonstrated that one internally learned concept can survive semantic repair and be recalled through the real `/internal` runtime path.

v0.10.16 extends the experiment from one concept to multiple learned concepts.

### New validator

`multi_knowledge_internalization_v01016.py` reads the sleep QA dataset and collects all `must_train=True` rows originating from learned Answer Memory and Relation Memory.

Rows are grouped by concept and one representative runtime probe is selected per concept, preferring:

```text
<concept>とは
<concept>
<concept>について教えて
<concept>を説明して
```

Each concept is then evaluated with the candidate model alone:

```text
SemanticRouter
  -> selected label
  -> semantic-guided prompt
  -> generation
  -> canonical answer comparison
```

External Semantic Memory, Answer Memory and Relation Memory are not used.

### Absolute candidate gate

Default requirements are:

```text
minimum unique concepts : 2
per-concept similarity  : >= 0.70
mean canonical similarity: >= 0.75
natural sentence termination required
abnormal-character ratio <= 0.02
repetition ratio <= 0.20
```

If fewer than two learned concepts exist, the validator reports `SKIP` rather than failing promotion. Once two or more concepts are present, the multi-knowledge gate becomes active.

### Promotion path

All direct and repair promotion paths now require:

```text
Semantic Retention PASS
AND
Dataset Answer Retention PASS
AND
Runtime /internal Answer Retention PASS
AND
Multi-Knowledge Internalization PASS/SKIP
```

The direct `semantic_sleep_v0105.py` promotion path now runs the same answer/runtime/multi-knowledge checks as the targeted repair path.

### Interactive command

`chat.py` adds:

```text
/internal-batch
```

This runs the multi-knowledge validator against the currently loaded model.

### Default next checkpoint

```text
model/model-sem-sleep-v0116.pt
```

### Regression

```powershell
python run_multi_knowledge_internalization_regression_v01016.py
```
