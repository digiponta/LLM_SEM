# LLM_SEM v0.6.6 Release Notes

## Overview

LLM_SEM v0.6.6 completes the v0.6 Semantic Composition series.

The series starts from the v0.6.1 Unified Semantic Runtime and extends it from
semantic routing and adaptive memory into proposition-level structural
composition. The final runtime can extract a proposition, preserve
Subject / Predicate / Object structure, select a composition policy from
semantic novelty, generate an adaptive proposition vector, and store the result
with provenance in SemanticDataV2.

## Major milestones

### v0.6.1 — Unified Semantic Runtime

Integrated:

- Base Semantic Router
- Adaptive Semantic Memory
- Multi-Prototype memory
- Local k-NN evidence
- Conditional adaptive override
- Semantic Data v2
- Purpose / intent extraction
- Relation and proposition extraction

Regression:

```powershell
python run_v061_regression.py
```

Result:

```text
RESULT: PASS
```

### v0.6.2 — Semantic Composition

Compared four proposition-composition methods:

```text
simple_mean
role_aware
relation_weighted
learned_comp
```

Original A/B/C holdout aggregate:

```text
simple_mean       4/11   mean margin -0.009362
role_aware       10/11   mean margin +0.012679
relation_weighted 9/11   mean margin +0.006192
learned_comp      9/11   mean margin +0.010585
```

Key finding:

The frozen structural role-aware composition generalized better overall than a
simple mean and better than the train-only learned composer on the holdout set.

### v0.6.3 — Adaptive Semantic Composition

Introduced a constrained adaptive composition gate.

Composition modes:

```text
balanced
  S=0.333 P=0.333 O=0.333

relation_aware
  S=0.25 P=0.50 O=0.25

concept_aware
  S=0.40 P=0.20 O=0.40
```

Exploratory A/B/C aggregate:

```text
balanced       10/11   +0.012679
adaptive_gate  10/11   +0.013768
```

The adaptive gate preserved positive cases while increasing mean structural
margin.

### v0.6.4 — Confirmatory Holdout

The v0.6.3 gate was frozen and evaluated on new D/E/F/G holdout cases.

No gate training, weight tuning, threshold tuning, or seed selection was
performed.

Confirmatory aggregate:

```text
balanced       10/13   +0.005945
adaptive_gate  10/13   +0.006517

Positive-case delta : +0
Mean-margin delta   : +0.000572
Confirmed           : True
```

This confirmed the adaptive policy on the newly introduced holdout set.

### v0.6.5 — Runtime Promotion

Added:

```text
adaptive_composition_runtime_v065.py
```

The confirmed v0.6.4 policy was promoted into the interactive runtime.

Frozen policy:

```text
known relation
  -> balanced

unseen relation + subject/object both seen in TRAIN
  -> relation_aware

otherwise
  -> balanced
```

SemanticDataV2 proposition specifications can now carry precomputed adaptive
proposition vectors.

Each proposition records composition provenance:

```text
composition_version
composition_mode
composition_weights
composition_reason
relation_seen
subject_seen
object_seen
composition_confidence
role_checkpoint
```

Regression:

```powershell
python run_v065_regression.py
```

Result:

```text
RESULT: PASS
```

### v0.6.6 — Proposition Runtime Validation

Added:

```powershell
python run_composition_runtime_validation_v066.py
```

The validation exercised the actual runtime path:

```text
proposition extraction
    -> adaptive composition
    -> adaptive proposition vector
    -> SemanticDataV2
```

Validated examples:

```text
GPUは高速
  -> has_property
  -> balanced
  -> S:0.333 P:0.333 O:0.333

GPUはcomputer
  -> has_predicate
  -> relation_aware
  -> S:0.250 P:0.500 O:0.250

GPUは計算装置
  -> has_predicate
  -> balanced
  -> S:0.333 P:0.333 O:0.333
```

Result:

```text
RESULT: PASS
Actual proposition extraction -> adaptive composition -> SemanticDataV2 path is working.
```

## Current runtime architecture

```text
User utterance
    |
    +--> Base Semantic Router
    |
    +--> Adaptive Semantic Memory
    |      +--> Multi-Prototype memory
    |      +--> Local k-NN evidence
    |      +--> Conditional override
    |
    +--> Purpose / Intent extraction
    |
    +--> Relation / Proposition extraction
    |
    +--> Structural Role Projection
    |      +--> Subject
    |      +--> Predicate
    |      +--> Object
    |
    +--> Adaptive Composition Gate
    |      +--> balanced
    |      +--> relation_aware
    |
    +--> Adaptive Proposition Vector
    |
    +--> SemanticDataV2
           +--> global vector
           +--> concept vectors
           +--> purpose vector
           +--> relation vectors
           +--> proposition vectors
           +--> confidence / uncertainty
           +--> composition provenance
           +--> runtime provenance
```

## Interpretation

The v0.6 series provides implementation evidence that semantic data should not
be reduced to a single embedding vector.

The current representation instead combines:

```text
semantic components
+ structural roles
+ explicit relations
+ proposition structure
+ adaptive composition policy
+ composed proposition vectors
+ provenance
```

## Next stage

v0.7 will extend the current single-proposition runtime toward:

```text
multiple propositions
    -> cross-proposition relations
    -> semantic graph
    -> context / purpose-aware graph composition
```

The v0.6.6 state is therefore the baseline for Semantic Graph research in the
v0.7 branch.
