# LLM_SEM v0.10.19

## Surface Generation Repair

v0.10.18 achieved multi-knowledge internalization with:

```text
文学      similarity=0.937500
量子暗号  similarity=0.891566
mean      similarity=0.914533
```

The remaining issue was surface realization. For example:

```text
量子力学の物理法則を利用して、理論上絶対に盗聴されない安全な通信を実現するなも電力です。
```

The semantic content is mostly correct, but the ending is malformed.

v0.10.19 adds `surface_generation_repair_v01019.py`.

### Repair target

Only mandatory rows whose canonical similarity is:

```text
0.70 <= similarity < 0.95
```

are selected for surface repair.

This avoids retraining rows that are either:
- already clean enough, or
- too semantically wrong to treat as a surface-only problem.

### Trainable parameters

Surface repair freezes the Transformer body and trains only:

```text
final_norm
lm_head
```

with a low learning rate.

Semantic benchmark preservation is retained through cosine-preservation loss.

### Concept balance

Repair rows are grouped by concept and each concept contributes equally to the repair loss.

### Sleep integration

The new stage runs after balanced internal learning and before final retention/promotion gates:

```text
Balanced Sleep
  -> Surface Generation Repair
  -> Semantic Retention
  -> Dataset Answer Retention
  -> Runtime Retention
  -> Multi-Knowledge Internalization
  -> Promotion
```

Default next candidate:

```text
model/model-sem-sleep-v0119.pt
```

Regression:

```powershell
python run_surface_generation_repair_regression_v01019.py
```
