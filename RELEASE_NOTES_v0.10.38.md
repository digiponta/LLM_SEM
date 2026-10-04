# LLM_SEM v0.10.38

## Iterative Protected Repair

v0.10.37 proved that one repair pass is not enough.

For the representative repair candidate:
- protected target NLL improved from 0.123733 to 0.096682;
- semantic cosine stayed at 0.999994;
- but `量子暗号とは` generation remained at canonical similarity 0.813187;
- protected drop therefore stayed at 0.078379, above the allowed 0.05.

## New policy

A protected repair is now iterative.

```text
new-knowledge candidate
    ↓
protected regression detected
    ↓
repair round 1
    ↓
revalidate
    ├─ protected safe + target retained → ACCEPT
    ├─ target lost                   → STOP / REJECT
    └─ still unsafe                  → repair round 2
                                      ...
```

Default maximum repair rounds: 4.

Each round:
- trains only the currently failing protected rows as `must_train`;
- keeps the newly learned target as a protected anchor;
- uses 60 epochs;
- uses 0.25x learning rate;
- increases protected replay weight by round:
  - round 1: 4.0
  - round 2: 6.0
  - round 3: 8.0
  - round 4: 10.0

A repair chain continues from the previous repair candidate, not from the original candidate.

## Safety

Repair succeeds only when:
- all protected rows satisfy the original retention thresholds;
- the new target still retains either canonical >= 0.70 or at least +0.02 visible gain.

If target progress disappears, repair stops and the row is rejected.

## Default candidate

```text
model/model-sem-sleep-v0138.pt
```

## Regression

```powershell
python run_iterative_protected_repair_regression_v01038.py
```
