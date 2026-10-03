# LLM_SEM v0.10.6

## Full Sleep Consolidation

`/sleep` now consolidates all three external knowledge layers:

```text
Semantic Memory
Learned Answer Memory
Relation Memory
```

Pipeline:

```text
Semantic Memory ACTIVE
    -> semantic-preserving internal training
    -> VALIDATING

Learned Answer Memory
Relation Memory
    -> canonical QA dataset
    -> answer fine-tuning with LM head
    -> strict QA/semantic preservation check

Final candidate
    -> semantic validation
    -> consolidated retention validation
    -> active-model promotion
```

Relation facts are canonicalized before training. Example:

```text
文学は言語による芸術表現を研究する分野
文学は、分類上、数学を含む
```

becomes:

```text
文学は、言語による芸術表現を研究する分野であり、分類上、数学を含む。
```

Default final checkpoint:

```text
model/model-sem-sleep-v0106.pt
```

Intermediate semantic checkpoint:

```text
model/model-sem-sleep-sem-v0106.pt
```

Promotion is blocked when strict QA fine-tuning fails, semantic validation fails, or retention fails.

Regression:

```powershell
python run_full_sleep_regression_v0106.py
```
