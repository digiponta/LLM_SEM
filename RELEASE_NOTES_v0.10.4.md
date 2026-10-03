# LLM_SEM v0.10.4

## Natural Multi-Relation Composition

This release extends v0.10.3 fact merge from same-relation composition to mixed-relation composition.

Examples:

```text
XはY
XはZ
-> Xは、Yであり、Zである。
```

```text
文学は言語による芸術表現を研究する分野
文学は、分類上、数学を含む
-> 文学は、言語による芸術表現を研究する分野であり、分類上、数学を含む。
```

```text
XはY
XはAを持つ
XはBを含む
-> Xは、Yであり、Aを持ち、Bを含む。
```

Conditional facts remain separate so their condition scope is not lost.

Regression:

```powershell
python run_multi_relation_composition_regression_v0104.py
```
