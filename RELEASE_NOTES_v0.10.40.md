# LLM_SEM v0.10.40

## Bug-fix audit

A code review of the current LLM_SEM training/runtime path found two implementation mismatches.

### 1. PARTIAL rows were frozen too early

Before v0.10.40, any query listed in the previous PARTIAL state was removed from `new_rows`, even when its canonical similarity was still below 0.70.

That meant a row could make partial progress once and then never be trained again.

v0.10.40 changes the behavior:

- if canonical >= 0.70: protect and stop retraining the row;
- if the row is PARTIAL but canonical < 0.70:
  - protect it while other rows are trained;
  - keep it as a future target so its own learning continues.

The current target is excluded from its own anchor list to avoid duplicate target/protected supervision.

### 2. Teacher forcing did not match runtime context length

`LanguageModel.generate()` uses:

```python
context = generated[-self.context_length:]
```

but the old answer fine-tuning loss fed the entire prompt + answer prefix to the Transformer.

For a context length of 64, a typical semantic prompt plus a 30-character answer can exceed the runtime window. The optimizer could therefore reduce NLL using prefix tokens that runtime generation would no longer see.

v0.10.40 replaces the answer loss with rolling teacher forcing. Every answer token is predicted using at most the same `model.context_length` previous tokens that `generate()` would see.

The same rule is used for protected canonical replay.

## Core model review

No immediate correctness fault was found in:
- causal attention masking;
- residual connections;
- LayerNorm placement;
- checkpoint load/save;
- greedy/top-k generation indexing.

The model intentionally has no explicit positional embedding. This remains an architectural limitation, not a newly introduced code bug.

## Regressions

```powershell
python run_partial_resume_bugfix_regression_v01040.py
python run_context_aligned_loss_regression_v01040.py
```

## Default candidate

```text
model/model-sem-sleep-v0140.pt
```
