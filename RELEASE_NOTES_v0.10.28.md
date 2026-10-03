# LLM_SEM v0.10.28

## Canonical-Aware Runtime + New-Knowledge Boost

v0.10.27 exposed two distinct remaining issues.

### 1. False runtime failure on exact canonical text

For `量子暗号とは`, the candidate produced the canonical answer exactly:

```text
canonical similarity = 1.000000
candidate:
量子力学の物理法則を利用して、理論上絶対に盗聴されない安全な通信を実現する技術
```

but the runtime gate rejected it only because the text did not end in `。`.

v0.10.28 treats an exact canonical match as semantically complete even when terminal punctuation is absent.

### 2. New knowledge was underweighted

Runtime replay protection preserved old behavior, but the new `量子通信` QA target was too weak relative to replay and semantic-preservation losses.

A new `--new-knowledge-weight` parameter explicitly scales the new-concept QA term:

```text
total loss
  = new_weight * new-concept QA
  + preserve_weight * semantic preservation
  + replay_weight * runtime replay protection
```

The Pareto sweep now searches:

```text
epochs  lr-scale  replay  new-weight
240     0.50      1       2
240     0.50      2       2
320     0.50      1       3
320     0.50      2       3
480     0.40      1       4
480     0.50      2       4
```

Promotion still requires a full runtime pass.

### Default candidate

```text
model/model-sem-sleep-v0128.pt
```

### Regression

```powershell
python run_canonical_aware_incremental_regression_v01028.py
```
