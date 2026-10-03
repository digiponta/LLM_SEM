# LLM_SEM v0.10.37

## Protected-Repair Partial Commit Sleep

v0.10.36 confirmed that new-knowledge learning is occurring.

A representative rejected candidate showed:

```text
量子通信を説明して
canonical 0.259259 -> 0.333333
Target NLL 4.060950 -> 1.233730
relative NLL improvement 69.620%
```

The candidate was rejected because one protected query regressed:

```text
量子暗号とは
canonical 0.891566 -> 0.813187
drop = 0.078379
allowed drop = 0.05
```

So the bottleneck is now protected-knowledge interference, not lack of learning.

### Repair-before-reject

When a candidate has visible target progress but one or more protected rows fail:

```text
new candidate
  -> target improved
  -> protected regression detected
  -> repair only failed protected rows
  -> keep new target as protected replay
  -> full runtime revalidation
```

The repair pass uses:
- 60 epochs;
- 0.25x learning rate;
- canonical target training for failed protected rows;
- replay weight 4.0;
- the newly learned row as a protected anchor.

### Acceptance after repair

A repaired candidate is accepted only when:
- every protected row is safe again;
- the new target still has either canonical >= 0.70 or at least +0.02 visible canonical gain.

If repair destroys the new progress, the candidate is still rejected.

### Why repair only visible progress

LATENT-only candidates without visible runtime gain remain eligible for ordinary v0.10.36 acceptance when protected-safe.

Protected repair is triggered only when the target already has visible runtime improvement, because that improvement can be revalidated directly after the repair pass.

### Default candidate

```text
model/model-sem-sleep-v0137.pt
```

### Regression

```powershell
python run_protected_repair_commit_regression_v01037.py
```
