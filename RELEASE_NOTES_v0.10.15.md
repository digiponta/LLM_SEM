# LLM_SEM v0.10.15

## Runtime Answer Retention

v0.10.14 protected mandatory QA using fixed dataset labels, but the real `/internal` path can still change its selected semantic label after repair.

Observed example:

```text
v0.10.13 /internal 文学とは
label=animal
answer=文学は、言語による芸術表現を研究する分野であり、数学を含む。

v0.10.14 /internal 文学とは
label=computer
answer=文学を研究する発表現も中であり、分類上、数学を含む。
```

The answer-retention test could pass because it used the dataset label rather than the model's actual runtime route.

v0.10.15 adds `runtime_answer_retention_v01015.py`.

For each mandatory sleep query, source and candidate independently execute:

```text
SemanticRouter
  -> selected label
  -> actual semantic-guided prompt
  -> deterministic internal generation
```

Promotion after retention repair now requires all three gates:

```text
Semantic Retention PASS
AND Dataset Answer Retention PASS
AND Runtime /internal Answer Retention PASS
```

Default runtime thresholds:

```text
source/candidate answer similarity >= 0.70
canonical similarity drop <= 0.05
natural sentence termination required
abnormal-character ratio <= 0.02
repetition ratio <= 0.20
```

The route transition itself is reported so changes such as `animal -> computer` are visible and their actual effect on generation is measured.

Default next sleep checkpoint:

```text
model/model-sem-sleep-v0115.pt
```

Regression:

```powershell
python run_runtime_answer_retention_regression_v01015.py
```
