# LLM_SEM v0.10.22

## Retention-First / No-Op Sleep

v0.10.21 confirmed that the active model can already satisfy all multi-knowledge runtime probes:

```text
文学      0.937500 PASS
量子暗号  0.891566 PASS
mean      0.914533
```

but repeated `/sleep` still re-entered balanced QA training because learned answer/relation memory files were non-empty. This unnecessary reconsolidation repeatedly degraded the already-good quantum-cryptography answer.

v0.10.22 changes the sleep order.

### Retention-first precheck

After rebuilding the canonical sleep dataset, the current model is evaluated through the actual `/internal` runtime path.

If:
- enough concepts are present,
- every mandatory concept passes,
- mean canonical similarity passes,

then balanced QA retraining is skipped.

### New flow

```text
/sleep
  -> build canonical dataset
  -> retention-first runtime precheck
       PASS -> skip Balanced QA
       FAIL -> run Balanced QA
  -> runtime-path selective surface repair
  -> if unchanged and no semantic work is pending:
       KEEP_SOURCE / NO-OP
     else:
       run normal retention gates
       promote only if all pass
```

### True NO-OP behavior

If the current active model already passes and selective surface repair produces no safe improvement, v0.10.22 does not create/promote a duplicate model.

Expected result:

```text
RETENTION-FIRST NO-OP SLEEP
SLEEP> KEEP_SOURCE: model/model-sem-sleep-v0118.pt
SLEEP> promotion skipped; active-model manifest unchanged.
```

### Default candidate names

```text
model/model-sem-sleep-v0122.pt
model/model-sem-sleep-sem-v0122.pt
```

### Regression

```powershell
python run_retention_first_sleep_regression_v01022.py
```
