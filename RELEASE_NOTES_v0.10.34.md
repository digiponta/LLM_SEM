# LLM_SEM v0.10.34

## Partial Commit Sleep

v0.10.34 changes `/sleep` from an all-or-nothing concept commit to row-level partial commits.

### Internalization states

```text
UNLEARNED
PARTIAL
COMPLETE
```

### Row-level commit

Each new QA row is still trained independently.

If a row:
- reaches the target canonical similarity;
- improves enough;
- does not damage any protected row;

then that row is immediately accepted into the accumulated checkpoint.

A later rejected row rolls back only that row's attempt. Earlier accepted rows remain committed.

### Partial promotion

After all rows have been attempted:

```text
all probes PASS
    -> COMPLETE

one or more rows accepted
AND known/protected failures == 0
    -> PARTIAL

otherwise
    -> UNLEARNED
```

A PARTIAL candidate is promoted to `active-model.json` so the next ChatGPT session loads the accumulated safe progress.

The complete runtime gate is no longer required merely to preserve partial progress. It remains the criterion for COMPLETE.

### Resume behavior

On the next `/sleep`, the current active model is probed first.

Any individual row already at canonical similarity >= 0.70 is recovered as a protected anchor even if its overall concept is still incomplete.

Thus:

```text
/sleep #1
  row A ACCEPT
  row B REJECT
  row C ACCEPT
  -> PARTIAL promoted

/sleep #2
  row A PROTECTED
  row C PROTECTED
  retry row B and remaining rows
```

### Safety condition for PARTIAL promotion

PARTIAL promotion requires:

```text
accepted_rows > 0
known/protected failures == 0
```

### Default candidate

```text
model/model-sem-sleep-v0134.pt
```

### Regression

```powershell
python run_partial_commit_sleep_regression_v01034.py
```
