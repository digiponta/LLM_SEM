# LLM_SEM v0.10.42

## Core Audit Fixes

A repository-wide audit inventoried 153 Python files and reviewed the active runtime/training path in detail.

### Fixed 1: repetition penalty incorrectly included prompt tokens

Previously `LanguageModel.generate()` applied the repetition penalty to every token in `generated`, including the original prompt.

With the character tokenizer, normal answer tokens such as characters from `量子通信` were penalized merely because they appeared in the question.

v0.10.42 now penalizes generated continuation tokens only by default.

Legacy behavior remains available with:

```python
penalize_prompt=True
```

### Fixed 2: Answer Memory candidate truth status was ignored

`truth_allows_answer_memory()` accepted a `candidate` argument but did not check its own `truth_status`.

A learned answer marked FALSE / CONTESTED / OUTDATED could therefore be used when no separate semantic truth record was present.

v0.10.42 blocks either source when marked:

- FALSE
- CONTESTED
- OUTDATED

### Fixed 3: selected checkpoint metadata could describe the wrong state

When quality selection restored an earlier best model state, the checkpoint could still save the final epoch/loss and final optimizer state.

v0.10.42:

- recomputes the objective on the actually selected weights;
- saves the actual selected epoch;
- omits optimizer state when it no longer matches restored best weights.

### Confirmed intentional behavior

`FAILED` Semantic Memory remains authoritative in adaptive routing until successful internal consolidation. This is intentional fallback behavior, not a bug.

### Remaining non-blocking risks

- Several JSON/JSONL rewrites are not fully transactional against abrupt process interruption.
- Legacy experiment scripts duplicate older logic and are not all part of the active runtime path.
- The base Transformer has no explicit positional embedding; this is an architectural limitation, not a runtime bug.
- Learned-answer upsert is concept-wide and may remove alternate paraphrases for the same concept by design.

## Regression tests

```powershell
python run_repetition_penalty_scope_regression_v01042.py
python run_answer_truth_gate_regression_v01042.py
python run_checkpoint_metadata_regression_v01042.py
```

## Default candidate

```text
model/model-sem-sleep-v0142.pt
```
