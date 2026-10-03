# LLM_SEM v0.10.3

## Fact Merge Learning

Restores the LLM_TRY fact-composition behavior for complementary facts about the same subject.

Example:

```text
XはY
XはZ
```

is stored as two structured `is` facts and rendered canonically as:

```text
Xは、Yであり、Zである。
```

Three facts become:

```text
Xは、Yであり、Zであり、Wである。
```

### Runtime

`/teach` accepts short copular facts such as `XはY` as trusted answers.

Each taught fact is persisted to Relation Memory. Compatible facts with the same subject are merged immediately and the canonical merged answer replaces the previous learned Answer Memory entry.

### Regression

```powershell
python run_fact_merge_regression_v0103.py
```
