# LLM_SEM v0.10.12

## Sleep Output Stabilization

v0.10.11 exposed a new failure mode: iterative sleep learned the beginning of the target answer, but generation continued into repetition and garbled text. It also repeatedly selected epoch 1 because holdout NLL worsened, discarding later sleep learning.

v0.10.12 changes sleep behavior in two ways.

### 1. Preserve iterative learning progress

During /sleep, QA fine-tuning now uses the final state of each round instead of restoring the epoch selected only by holdout NLL.

This lets:

```text
round1 -> round2 -> round3 -> ...
```

actually accumulate answer memorization.

### 2. Stabilize generated answers

Generated answers are cut at the first complete Japanese sentence terminator `。` for sleep evaluation and runtime internal generation.

Sleep now also records:

- natural termination rate
- maximum abnormal-character ratio
- maximum repetition ratio

Default completion gates additionally require:

```text
termination rate       >= 1.00
abnormal-char ratio    <= 0.02
repetition ratio       <= 0.20
```

The similarity and semantic-preservation conditions from v0.10.11 remain in force.

Default next checkpoint:

```text
model/model-sem-sleep-v0112.pt
```

Regression:

```powershell
python run_sleep_output_stabilization_regression_v01012.py
```
