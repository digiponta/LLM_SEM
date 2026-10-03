# LLM_SEM v0.10.13

## Targeted Retention Repair

v0.10.12 demonstrated successful internal answer learning, but a consolidated semantic item could still flip to the wrong label after sleep QA training.

Observed example:

```text
量子コンピュータとは何ですか
expected      : computer
source_top    : computer
candidate_top : science
```

v0.10.13 keeps this as a real retention failure and does not relax the acceptance threshold.

Instead it adds a targeted repair stage:

```text
/sleep
  -> iterative Answer/Relation learning
  -> output stabilization
  -> retention validation
  -> FAIL
  -> targeted semantic repair
  -> retention re-validation
  -> PASS only
  -> active model promotion
```

### New components

- `retention_repair_v01013.py`
  - detects misclassified CONSOLIDATED records
  - freezes the LM head
  - trains only the final Transformer semantic block and final_norm
  - preserves benchmark geometry
  - protects all consolidated rows while emphasizing damaged rows

- `retention_repair_loop_v01013.py`
  - runs retention validation
  - retries targeted repair up to three times
  - promotes only after retention passes

- `chat.py`
  - automatically invokes the repair loop when /sleep fails after a final candidate has already been created

The default sleep output advances to:

```text
model/model-sem-sleep-v0113.pt
```

Regression:

```powershell
python run_retention_repair_regression_v01013.py
```
