# LLM_SEM v0.15.7.5 Release Notes

## Semantic Memory Internalization Completion

v0.15.7.5 closes the Semantic Memory internalization experiment by promoting a
candidate that satisfies both decoder-boundary crossing and source-trajectory
preservation.

## Objective

The experiment asked whether knowledge stored outside the language model in
Semantic Memory could be transferred into model parameters strongly enough to
affect runtime decoder preference without unacceptable regression of protected
source behavior.

The required stages were:

```text
Semantic Memory
  -> latent internalization
  -> decoder preference movement
  -> decoder boundary crossing
  -> trajectory preservation
  -> qualified candidate
  -> promoted checkpoint
```

## Experimental history

### v0.15.5 — Latent internalization

The update clearly changed model likelihood and weights, but the generated
behavior remained below the decoder decision boundary. This stage established
that latent internalization alone was insufficient.

### v0.15.6 / v0.15.6.1 — Decoder-boundary diagnostics

Prompt-aware continuation scoring removed the earlier subject-duplication
artifact and showed that canonical knowledge still lost to the confuser before
decoder-focused training.

### v0.15.7 — Contrastive decoder crossing

Only `final_norm` and `lm_head` were trainable. Embedding and Transformer
blocks remained frozen. Initial learning rates moved the margin in the correct
direction but did not cross the boundary.

### v0.15.7.1 — Learning-rate sweep

The sweep confirmed that larger decoder learning rates produced larger,
monotonic margin gains.

### v0.15.7.2 — Aggressive sweep

Two candidates crossed the target decoder boundary. The strongest-margin model
was not automatically selected because preservation had not yet been proven.

### v0.15.7.3 — Initial preservation check

Greedy string similarity proved too brittle for a small autoregressive model:
small early-token changes amplified into large whole-string differences even
when local distributions remained similar.

### v0.15.7.4 — Trajectory preservation

The preservation criterion was changed to distribution- and trajectory-based
metrics under source prefixes. The `5e-4 / 2e-4` candidate passed.

Final qualified metrics:

```text
Prompt-aware result      : PROMPT_AWARE_PASS
Mean prompt JS           : 0.042576
Prompt top1 retention    : 83.3%
Mean trajectory JS       : 0.042938
Mean trajectory top1     : 76.6%
Mean source NLL delta    : +0.117602
Boundary check           : PASS
Trajectory preservation : PASS
QUALIFIED                : YES
```

Prompt-aware decoder result:

```text
Mean margin BEFORE : -1.807448
Mean margin AFTER  : +0.436255
Mean margin gain   : +2.243703
Boundary crossed   : 4/4
RESULT             : PROMPT_AWARE_PASS
```

The higher-rate `1e-3 / 5e-4` candidate crossed the boundary but failed
trajectory preservation and was rejected.

### v0.15.7.5 — Final promotion

The qualified candidate was promoted through a guarded promotion script.

Promoted model:

```text
model/model-sem-internalized-v01575.pt
```

Manifest:

```text
results/promotion_manifest_v01575.json
```

Promotion result:

```text
Boundary             : PROMPT_AWARE_PASS
Trajectory JS        : 0.042938
Trajectory top1      : 76.6%
Source NLL delta     : +0.117602
RESULT               : PROMOTED
```

## Key scripts

```text
run_decoder_boundary_diagnostic_v01561.py
run_contrastive_decoder_crossing_v0157.py
run_decoder_crossing_lr_sweep_v01571.py
run_decoder_crossing_aggressive_sweep_v01572.py
run_candidate_preservation_v01573.py
run_trajectory_preservation_v01574.py
run_final_promotion_gate_v01575.py
```

## Conclusion

The experiment demonstrates, for this model and validation policy, that
Semantic Memory content can be moved into model weights strongly enough to
cross a decoder decision boundary while preserving protected source behavior
within the defined thresholds.

This is a scoped experimental result rather than a general guarantee for
arbitrary knowledge, model sizes, prompts, or future internalization methods.

## Next branch

Development continues on:

```text
v0.16.0
```

The next phase can treat `model-sem-internalized-v01575.pt` as the reference
internalized checkpoint and investigate repeated/multi-memory internalization,
automatic promotion policy, and long-term cumulative preservation.
