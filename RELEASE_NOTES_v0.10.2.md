# LLM_SEM v0.10.2

## LLM_TRY Command Compatibility

Fixes two interactive issues found in v0.10.1:

1. `/teach <sentence>` was interpreted as a semantic label.
2. `/train` was not a command and fell through as a normal user query.

### Changes

- `/teach <label|answer>`
  - short label-like values keep the original semantic-label behavior
  - sentence-like values are redirected to trusted-answer teaching
- `/forget`
  - removes the semantic-memory entry for the previous utterance
- `/train`
  - reloads learned Semantic/Answer Memory for LLM_TRY workflow compatibility
  - does not perform an implicit gradient update
- learned Answer Memory is loaded before unified/base Answer Memory so human-taught answers win score ties

### Recovery from an accidental label teaching

```text
文学とは
/forget
文学とは
/teach 文学は言語による芸術表現を研究する分野です。
/train
文学とは
```

### Regression

```powershell
python run_command_compat_regression_v0102.py
```
