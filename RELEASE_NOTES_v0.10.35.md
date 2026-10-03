# LLM_SEM v0.10.35

## Progressive Partial Commit Sleep

v0.10.34 introduced row-level PARTIAL commits, but each row still had to reach the final canonical threshold before it could be accepted.

That meant useful intermediate learning such as:

```text
0.25 -> 0.40
0.40 -> 0.55
0.55 -> 0.68
```

was discarded even when all protected knowledge remained safe.

v0.10.35 changes row acceptance to allow safe incremental progress.

### Progressive row acceptance

A row can now be committed in either of two ways.

#### Target acceptance

```text
canonical >= 0.70
AND
gain >= 0.10
```

or exact canonical completion.

#### Progressive acceptance

```text
gain >= 0.02
AND
candidate canonical > source canonical
AND
protected failures == 0
```

Thus a safe improvement below 0.70 is no longer thrown away.

### Example

```text
/sleep #1
0.25 -> 0.36  PROGRESS COMMIT

/sleep #2
0.36 -> 0.51  PROGRESS COMMIT

/sleep #3
0.51 -> 0.73  TARGET COMMIT
```

Each accepted checkpoint becomes the source of the next learning step.

### Cross-session recovery

PARTIAL state is stored alongside the promoted checkpoint.

On the next `/sleep`, rows recorded in the previous PARTIAL state are restored as protected anchors even when their canonical similarity is still below 0.70.

This preserves intermediate progress across sessions.

### Repeated PARTIAL candidates

If the active source model has the same filename as the configured sleep candidate, the sleep pipeline automatically creates:

```text
<name>.next.pt
```

instead of failing with a source/candidate path collision.

### States

```text
UNLEARNED
PARTIAL
COMPLETE
```

PARTIAL still requires zero known/protected failures.

### Default candidate

```text
model/model-sem-sleep-v0135.pt
```

### Regression

```powershell
python run_progressive_partial_commit_regression_v01035.py
```
