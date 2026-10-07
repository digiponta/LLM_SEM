# LLM_SEM v0.18.16 Stable NDC Runtime

## Summary

v0.18.16 consolidates the NDC semantic-routing experiments into a stable runtime path.

The final architecture combines:

- frozen semantic encoder
- augmented NDC prototype classification
- contrastive known-vs-unknown gating
- beam-based selected 3-digit routing
- strict UNKNOWN rescue
- small deterministic keyword prior for fine-code boundary cases

## Stable results

### Main-class NDC runtime

- regression: 16/16 PASS
- known NDC 0-9 examples: accepted correctly
- ambiguous / undefined probes: UNKNOWN

### Selected 3-digit NDC runtime

- regression: 33/33 PASS
- main-class rescue supported
- UNKNOWN-gate rescue supported only under strict fine-code evidence
- UNKNOWN remains separate from NDC 000

## Evolution

- v0.18.1: NDC metadata baseline
- v0.18.2: single centroid experiment
- v0.18.3: multi-prototype routing
- v0.18.4: full projection experiment
- v0.18.5-v0.18.6: regularized / leakage-free projection
- v0.18.7: confusion-aware prototypes
- v0.18.8: dual-router consensus
- v0.18.9: separated classification / unknown detection
- v0.18.10: contrastive unknown gate
- v0.18.11: stable main-class runtime
- v0.18.12: hard hierarchical selected 3-digit routing
- v0.18.13: beam hierarchical rescue
- v0.18.14: fine-code + UNKNOWN rescue
- v0.18.15: stable hybrid selected 3-digit router
- v0.18.16: stable runtime consolidation

## Default runtime commands

```text
/ndc <text>   # stable main-class NDC routing
/ndc3 <text>  # stable selected 3-digit NDC routing
```

Experimental compatibility commands remain available:

```text
/ndc3beam
/ndc3rescue
/ndc3stable
```

## Stable verification

```powershell
python .\verify_ndc_stable_v01816.py
```

Expected final result:

```text
RESULT: PASS
  main-class regression : PASS (16/16 expected)
  selected 3-digit      : PASS (33/33 expected)
  stable runtime        : READY
```
