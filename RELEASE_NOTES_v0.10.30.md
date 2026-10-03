# LLM_SEM v0.10.30

## One-by-One Sleep Consolidation

`/sleep` now learns new QA rows one at a time.

### Core loop

```text
one NEW row
  -> fine-tune
  -> validate current target
  -> validate all already committed/protected rows
  -> ACCEPT or ROLLBACK
  -> next row
```

Only accepted rows are accumulated into the next source checkpoint.

Rejected rows do not modify the active accumulation state.

### Protection policy

At each step:
- previously known rows are protected with actual runtime replay;
- previously accepted new rows are promoted to protected replay anchors;
- only the current row is marked `must_train=true`.

Thus the training dataset for each step contains exactly one new learning target.

### Console trace

Each step prints:

```text
ONE-BY-ONE STEP 1/N
SOURCE :
CONCEPT:
QUERY  :
ANSWER :
PROTECTED ANCHORS:
TARGET : canonical before -> after
PROTECTED FAILURES:
DECISION: ACCEPT / REJECT / ROLLBACK
```

### Final promotion

After all rows are processed, the accumulated candidate is checked against the complete runtime probe set.

Only a complete runtime PASS can continue to promotion.

### Default candidate

```text
model/model-sem-sleep-v0130.pt
```

### Regression

```powershell
python run_one_by_one_sleep_regression_v01030.py
```
