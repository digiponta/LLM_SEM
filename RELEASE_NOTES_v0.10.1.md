# LLM_SEM v0.10.1

## Relation Teaching Integration

This release completes the relation-memory path integrated from LLM_TRY.

### Added

- persistent structured relation memory
- relation parsing from trusted /teach-answer responses
- contextual relation persistence
- duplicate suppression
- unified-memory rebuild support

### Files

- relation_memory_v0101.py
- run_relation_memory_regression_v0101.py
- data/relation_memory_v0101.jsonl

### Verification

```powershell
python run_relation_memory_regression_v0101.py
python build_unified_semantic_answer_memory_v0100.py
python chat.py
```

After the unified-memory rebuild, relation-backed concepts become available through Semantic Answer Memory and Answer-Aware Gate.
