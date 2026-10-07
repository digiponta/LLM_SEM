# LLM_SEM v0.19.8 Release Notes

## Generalized Stable NDC Runtime Integration

v0.19.8 promotes the strongest cross-holdout selected 3-digit NDC router into
the default interactive runtime.

### Runtime

- `/ndc <text>`: v0.18.11 stable main-class NDC router
- `/ndc3 <text>`: v0.19.6 generalized selected 3-digit NDC router

Both use the dedicated frozen checkpoint:

```text
model/model-sem-internalized-v01575.pt
```

and remain isolated from live chat-model changes caused by `/sleep`,
`/repair`, or reload operations.

## v0.19 experiment progression

- v0.19.0: independent holdout exposed regression-fit gap
  - raw known accuracy 45.00%
  - unknown reject 66.67%
  - balanced 55.83%
- v0.19.1: prototype coverage expansion
  - raw known accuracy 83.33%
  - balanced 75.00%
- v0.19.2: expanded UNKNOWN gate
  - known accuracy 90.00%
  - unknown reject 87.50%
  - balanced 88.75%
- v0.19.4: conditional rescue + pairwise adjudication
  - HOLDOUT-V5 balanced 98.33%
- v0.19.5: local NDC 830 stabilization
- v0.19.6: local NDC 930 stabilization
- v0.19.7: cross-holdout robustness sweep
- v0.19.8: stable runtime integration

## Cross-holdout robustness

v0.19.7 ranked v0.19.6 as the strongest router across HOLDOUT-V3 through
HOLDOUT-V7:

- mean known accepted accuracy: 96.00%
- mean UNKNOWN rejection: 93.75%
- mean balanced score: 94.88%
- minimum known accepted accuracy: 90.00%
- minimum UNKNOWN rejection: 87.50%
- minimum balanced score: 88.75%
- result: ROBUST_CANDIDATE

## Integrated verification

Run:

```powershell
python .\verify_ndc_stable_v0198.py
```

The verifier combines:

1. v0.18.16 stable regression
2. v0.19.7 cross-holdout robustness sweep

Expected final state:

```text
stable main-class runtime          : READY
generalized selected-3digit router: READY
chat /ndc3 promotion              : READY
RESULT                             : PASS
```

## Methodological note

The cross-holdout sweep is retrospective. Later v0.19 router variants were
designed after observing errors on earlier holdouts. Therefore these results
support a robust runtime candidate but should not be presented as a fully
independent estimate of generalization over the complete NDC taxonomy.
