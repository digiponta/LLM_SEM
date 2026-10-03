# LLM_SEM v0.10.8

## Internal Knowledge Probe

Adds:

```text
/internal <query>
```

This command bypasses:

- Semantic Memory
- Answer Memory
- Relation Memory

and probes only the currently loaded internal checkpoint.

Example:

```text
/internal 文学とは
```

Output includes:

```text
INTERNAL> bypass=SemanticMemory,AnswerMemory,RelationMemory
INTERNAL> label=... sim=... margin=... mode=semantic-guided
AI-INTERNAL> ...
```

This makes it possible to verify whether /sleep actually transferred external knowledge into the model weights, instead of merely observing an Answer Memory hit.

Normal chat behavior is unchanged.
