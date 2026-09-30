# LLM_SEM

CUDA/PyTorch semantic-extension version of the homemade LLM project.

This repository is based on the architecture and end-to-end flow proven in
`digiponta/LLM` branch `v0.3`. The original educational virtual-GPU runtime
and hand-written backward propagation are replaced by real PyTorch tensors,
CUDA kernels, and PyTorch autograd.

## Architecture

```text
Japanese corpus
    |
character tokenizer
    |
Embedding
    |
Transformer blocks
    |-- LayerNorm
    |-- single-head causal Self-Attention
    |-- residual connection
    |-- LayerNorm
    |-- FFN (Linear -> GELU -> Linear)
    |-- residual connection
    |
final LayerNorm
    |
lm_head
    |
Cross Entropy
    |
PyTorch autograd
    |
AdamW
    |
CUDA GPU
```

## Files

```text
tokenizer.py          character tokenizer compatible with the original project
dataset.py            next-token dataset / uniform sampled windows
model.py              CUDA-capable Transformer language model
train.py              GPU training loop with progress and ETA
train_corpus.py       corpus training entry point
infer.py              interactive GPU inference
check_gpu.py          CUDA/PyTorch diagnostic
prepare_wikipedia.py  Wikipedia dump -> plain text helper
requirements.txt      Python dependencies
```

## Installation

Install a CUDA-enabled PyTorch build suitable for the NVIDIA driver on the PC,
then install the project requirements.

```powershell
python -m pip install -r requirements.txt
python check_gpu.py
```

A successful setup reports:

```text
CUDA available : True
Device         : cuda
GPU            : NVIDIA ...
VRAM           : ... GiB
```

## Training data

By default the trainer looks for:

```text
data/general-ja.txt
data/data-nagato.txt
```

For convenience, if those files do not exist in this repository, it also
looks for the existing original-project files:

```text
../LLM/data/general-ja.txt
../LLM/data/data-nagato.txt
```

## Train

```powershell
python train_corpus.py
```

Default v0.4 long GPU run:

```text
context length : 64
d_model        : 64
layers         : 2
FFN dimension  : 256
attention heads: 1
batch size     : 64
samples        : 120,000,000
epochs         : 3
learning rate  : 5e-4
```

This sample count is estimated from the measured v0.3 benchmark on an
RTX 3070 Ti: 20,000 samples x 3 epochs took about 6 seconds. Linear scaling
gives approximately 120,000,000 samples x 3 epochs for a ten-hour run.

Actual runtime will vary with GPU clocks, thermals, system load, and I/O.
v0.4 computes sample positions on demand instead of allocating a huge Python
list, and DataLoader shuffling is disabled because the dataset itself uses a
deterministic pseudo-random corpus traversal.

Checkpoint:

```text
model/model-gpu-v0.4.pt
```

## Inference

```powershell
python infer.py
```

Generation uses temperature, top-k sampling, repetition penalty, and a bounded
context window.

## Relationship to the original v0.3

The original repository verified the complete path:

```text
corpus -> tokenizer -> model forward -> loss -> backward
       -> optimizer -> checkpoint -> reload -> inference
```

LLM_GPU preserves that path while moving tensor computation and gradient
calculation to a physical CUDA GPU.


## Semantic extension

LLM_SEM adds an explicit semantic representation on top of the trained
Transformer hidden states.

```text
Text
  |
Tokenizer
  |
Token IDs
  |
Embedding
  |
Transformer blocks
  |
Final contextual hidden states [batch, time, d_model]
  |
Mean pooling
  |
Semantic Vector [batch, d_model]
  |
SemanticData
```

### Semantic API

`model.py` now provides:

```python
hidden = model.encode_hidden(token_ids)
semantic_vector = model.encode_semantic(token_ids)
```

With the current default model, the semantic vector dimension is
`d_model = 64`.

`semantic.py` defines the explicit exported structure:

```text
SemanticData
  - text
  - vector
  - dimension
  - token_count
  - model_type
  - confidence
```

It also provides cosine similarity and cosine semantic distance.

### Semantic experiment

After placing the existing tokenizer and trained checkpoint under `model/`:

```powershell
python semantic_demo.py
```

The demo converts several Japanese sentences into semantic vectors and
compares them using cosine similarity and semantic distance.

This is the first implementation stage. Meaning, intent, purpose, metadata,
and learned semantic routing are not yet explicitly inferred; the current
SemanticData vector is derived from the Transformer's contextual hidden state.


## Semantic performance evaluation

`semantic_eval.py` evaluates whether the existing trained checkpoint already
contains useful semantic structure. The evaluation does not retrain the model.

The built-in benchmark contains Japanese sentences from several semantic
classes such as animals, weather, computers, food, and transportation.

Run:

```powershell
python semantic_eval.py
```

The experiment measures:

- mean cosine similarity for sentence pairs in the same semantic class
- mean cosine similarity for sentence pairs in different semantic classes
- semantic margin = within-class mean - between-class mean
- leave-one-out 1-nearest-neighbor label accuracy
- pairwise cosine distance

Pairwise results are also written to:

```text
semantic_eval_results.csv
```

A positive semantic margin means that, on average, sentences in the same
semantic group are closer than sentences in different groups.

A custom benchmark can be supplied as JSON:

```powershell
python semantic_eval.py --benchmark-json my_benchmark.json
```

JSON format:

```json
[
  {"label": "animal", "text": "猫は動物です。"},
  {"label": "animal", "text": "犬は動物です。"},
  {"label": "weather", "text": "今日は雨です。"}
]
```

The purpose of this experiment is to determine how much semantic information
can be extracted from the existing next-token language model before adding
semantic-specific training objectives.


## Semantic pooling comparison

The semantic evaluator can now compare five pooling strategies without
retraining the language model:

- `mean`
- `last`
- `bos`
- `max`
- `attention`

Run all methods on the repository benchmark:

```powershell
python semantic_eval.py --benchmark my_benchmark.csv
```

The output includes a comparison table with:

- within-class cosine similarity
- between-class cosine similarity
- semantic margin
- 1-NN label accuracy

Detailed pairwise results are written to:

```text
semantic_eval_results.csv
```

The method comparison is written to:

```text
semantic_eval_summary.csv
```

To evaluate one pooling method only:

```powershell
python semantic_eval.py --benchmark my_benchmark.csv --pooling attention
```

The `attention` method uses the final Transformer block's self-attention
weights to weight the final contextual token states. The `bos` method is
included as an experimental baseline; because this model uses causal
attention, the BOS position cannot attend to later tokens.


## Hybrid semantic vector

LLM_SEM now supports a hybrid representation:

```text
Hybrid = alpha * AttentionVector + (1 - alpha) * LastTokenVector
```

Use `semantic_hybrid_eval.py` to sweep alpha from 0.0 to 1.0 without
retraining the model:

```powershell
python semantic_hybrid_eval.py --benchmark my_benchmark.csv
```

Default sweep:

```text
alpha = 0.0, 0.1, 0.2, ... 1.0
```

The script reports the within-class similarity, between-class similarity,
semantic margin, and 1-NN accuracy for every alpha. It separately reports
the alpha with the highest 1-NN accuracy and the alpha with the largest
semantic margin.

Results are saved to:

```text
semantic_hybrid_summary.csv
```

A finer search can be run with:

```powershell
python semantic_hybrid_eval.py --benchmark my_benchmark.csv --alpha-step 0.05
```


## Normalized hybrid semantic vector

Hybrid evaluation now supports two modes:

```text
raw:
  hybrid = alpha * attention + (1 - alpha) * last

normalized:
  a = L2Normalize(attention)
  l = L2Normalize(last)
  hybrid = L2Normalize(alpha * a + (1 - alpha) * l)
```

Compare both modes over the alpha sweep:

```powershell
python semantic_hybrid_eval.py --benchmark my_benchmark.csv --alpha-step 0.05
```

Run only the normalized hybrid:

```powershell
python semantic_hybrid_eval.py --benchmark my_benchmark.csv --alpha-step 0.05 --mode normalized
```

Run only the original raw hybrid:

```powershell
python semantic_hybrid_eval.py --benchmark my_benchmark.csv --alpha-step 0.05 --mode raw
```

The summary CSV now includes a `mode` column so raw and normalized hybrid
results can be compared directly.


## Default semantic representation

Based on the current benchmark experiments, the default semantic representation
is now:

```text
pooling          = hybrid
alpha            = 0.35
normalize_hybrid = False
```

That is:

```text
SemanticVector = 0.35 * AttentionVector + 0.65 * LastTokenVector
```

This configuration produced the largest semantic margin in the current raw
hybrid sweep while retaining high 1-NN accuracy.

## Semantic routing

`semantic_router.py` builds one semantic centroid per label from
`my_benchmark.csv` and routes new text to the closest centroid using cosine
similarity.

Interactive mode:

```powershell
python semantic_router.py
```

Single-text mode:

```powershell
python semantic_router.py --text "明日の東京の気温を知りたいです"
```

The router prints the selected route and the top candidate routes with
similarity and semantic distance.

The current routing classes are derived from the benchmark labels:

```text
animal
weather
computer
food
transport
science
```

This is a prototype of the Semantic OS routing path:

```text
Input text
  |
LLM_SEM semantic encoder
  |
64-D raw hybrid semantic vector
  |
cosine similarity to route centroids
  |
Semantic route / VM candidate
```


## Leave-one-out routing evaluation

The semantic router can evaluate all benchmark samples with leave-one-out
centroid routing:

```powershell
python semantic_router.py --evaluate
```

This reports:

- overall routing accuracy
- per-category accuracy
- confusion matrix
- mean Top-1 similarity
- mean Top-1 / Top-2 margin
- candidate thresholds for an Unknown route

The candidate Unknown thresholds are estimated from the lower 10% region of
correctly routed samples. A sample can be treated as Unknown when either its
Top-1 similarity or its Top-1 / Top-2 margin falls below the corresponding
threshold.

Per-sample evaluation results are saved to:

```text
semantic_router_eval.csv
```


## Unknown / Discovery route evaluation

A separate unknown-category benchmark is included:

```text
unknown_benchmark.csv
```

It contains examples from categories that are not part of the six known
routing classes.

Run:

```powershell
python semantic_router.py --evaluate-unknown
```

The evaluation derives the Unknown thresholds from leave-one-out routing on
the known benchmark, then evaluates the separate unknown benchmark.

Reported metrics:

- Known routing accuracy
- Known accept rate
- Unknown detection rate
- False Unknown rate
- False Known rate
- Unknown detection rate by unknown category

Per-sample results are saved to:

```text
semantic_unknown_eval.csv
```

The current candidate decision rule is:

```text
Unknown if:
  Top-1 similarity < similarity threshold
  OR
  Top-1/Top-2 margin < margin threshold
```

This provides a prototype path from ordinary semantic routing to an
Unknown / Discovery route.


## Threshold sweep without Known-sample leakage

The Known/Unknown evaluation now uses leave-one-out centroids for every known
sample. This removes the earlier optimistic bias caused by evaluating a known
sample against a centroid that included that same sample.

Run the corrected Known/Unknown evaluation:

```powershell
python semantic_router.py --evaluate-unknown
```

The report now includes:

- Known recall
- Known accept rate
- Unknown detection rate
- False Unknown rate
- False Known rate
- Balanced accuracy

To search similarity and margin thresholds:

```powershell
python semantic_router.py --sweep-thresholds
```

The sweep compares threshold pairs using:

```text
Balanced Accuracy =
    (Known Recall + Unknown Detection Rate) / 2
```

The top threshold combinations are printed, and all evaluated combinations are
saved to:

```text
semantic_unknown_threshold_sweep.csv
```

The best threshold pair is selected by Balanced Accuracy, with Unknown
Detection Rate and Known Recall used as secondary tie-break criteria.


## Routing policies

The semantic router now supports three automatic Known/Unknown threshold
policies derived from the threshold sweep:

```text
known-first
balanced
discovery-first
```

Policy selection criteria:

```text
known-first:
  maximize Known Recall first

balanced:
  maximize Balanced Accuracy first

discovery-first:
  maximize Unknown Detection Rate first
```

Apply a policy to a single input:

```powershell
python semantic_router.py --policy balanced --text "明日の東京の気温を知りたいです"
```

Examples:

```powershell
python semantic_router.py --policy known-first --text "猫は動物です"
python semantic_router.py --policy balanced --text "株価について調べたいです"
python semantic_router.py --policy discovery-first --text "ピアノで和音を演奏します"
```

The router prints the automatically selected similarity and margin thresholds,
their expected benchmark metrics, and then routes the input either to a known
semantic class or to:

```text
unknown
```

Interactive routing can also use a policy:

```powershell
python semantic_router.py --policy discovery-first
```

This provides an operational prototype for switching Semantic OS behavior
between preserving known routes and aggressively forwarding uncertain semantic
tasks to an Unknown / Discovery VM.


## Constrained balanced policy

The `balanced` policy now requires a minimum Known Recall before maximizing
Balanced Accuracy.

Default constraint:

```text
Known Recall >= 70%
```

Run:

```powershell
python semantic_router.py --policy balanced --text "明日の東京の気温を知りたいです"
```

A custom minimum Known Recall can also be supplied:

```powershell
python semantic_router.py --policy balanced --balanced-min-known-recall 0.75 --text "明日の東京の気温を知りたいです"
```

Selection logic:

```text
known-first:
  maximize Known Recall

balanced:
  require Known Recall >= minimum
  then maximize Balanced Accuracy

discovery-first:
  maximize Unknown Detection Rate
```

If no threshold pair satisfies the requested Known Recall constraint, the
balanced policy falls back to the threshold pair with the highest Known Recall.


## v0.3 Adaptive Semantic Learning

Branch `v0.3` adds a first adaptive-learning loop to LLM_SEM without
retraining the frozen Transformer for every correction.

The key idea is to keep the base semantic encoder frozen and append manually
taught utterances to a persistent semantic memory. The centroid router is then
refit immediately from the original benchmark plus the adaptive examples.

```text
User utterance
      |
Frozen semantic encoder
      |
Semantic vector
      |
Centroid router
      |
+-----+-------------------+
|                         |
Known / ACCEPT        Unknown / Review
|                         |
route to class         /teach <label>
                          |
                  semantic_memory.jsonl
                          |
                    refit centroids
                          |
                 next query can be Known
```

New files:

```text
adaptive_semantic_learning.py             persistent labeled semantic memory
chat.py                                   interactive adaptive semantic shell
run_adaptive_semantic_learning_regression_v03.py
                                          memory/deduplication regression tests
```

Run the regression first:

```powershell
python run_adaptive_semantic_learning_regression_v03.py
```

Then start the adaptive semantic shell:

```powershell
python chat.py
```

Interactive commands:

```text
/learn on|off|status
/teach <label>
/memory
/quit
```

Example learning cycle:

```text
You> 明日の大阪は傘が必要ですか
SEM> UNKNOWN_KNOWLEDGE ...

You> /teach weather
Learned: label='weather', text='明日の大阪は傘が必要ですか'

You> 明日の大阪は傘が必要ですか
SEM> ACCEPT  label=weather ...
```

This is intentionally a semantic-memory update rather than full model
fine-tuning. It provides a low-risk first step for Adaptive Semantic Learning:
unknown or ambiguous semantic regions can be corrected interactively while the
base language model remains unchanged.


## License

This project is licensed under the **Apache License 2.0**.

You may use, modify, and distribute this software in accordance with the terms of the Apache License, Version 2.0.

See the `LICENSE` file for the full license text.

Apache License 2.0  
Copyright (c) Hirofumi Inomata


## Semantic Data v2.0 Runtime Integration on v0.3

Branch `v0.3` now also includes the Semantic Data v2.0 representation and
runtime adapter while preserving the existing Adaptive Semantic Learning flow.

Added/updated files:

```text
semantic.py                    Semantic Data v2.0 + legacy compatibility
semantic_runtime_v2.py         adaptive-runtime -> SemanticDataV2 adapter
semantic_runtime_v2_demo.py    runtime integration example
```

The existing v0.3 semantic-memory runtime can remain the decision source:

```text
User Query
   |
Frozen Semantic Encoder
   |
Adaptive Semantic Memory / Router
   |
memory / local / base / gate signals
   |
semantic_runtime_v2.py
   |
SemanticDataV2
   |
   +-- Global Vector
   +-- Concept Vectors[]
   +-- Purpose Vector
   +-- Relations[]
   +-- Context
   +-- Confidence / Uncertainty
```

Typical integration:

```python
from semantic_runtime_v2 import from_runtime_dict

runtime = {
    "memory_label": memory_label,
    "memory_similarity": memory_similarity,
    "memory_margin": memory_margin,
    "local_majority": local_majority,
    "local_purity": local_purity,
    "local_k": local_k,
    "base_label": base_label,
    "base_similarity": base_similarity,
    "gate_state": gate_state,
    "selected_label": selected_label,
}

semantic_v2 = from_runtime_dict(
    model,
    tokenizer,
    user_text,
    runtime,
    concept_texts=detected_concepts,
    purpose_text=purpose_text,
    intent=intent,
)
```

Existing SemanticData-v1 callers remain compatible through:

```python
legacy = semantic_v2.as_legacy()
vector = semantic_v2.primary_vector()
```

This allows the v0.3 Adaptive Semantic Learning branch to serve as the
experimental runtime for Semantic Data v2.0 without replacing the existing
router or memory implementation.


## Native SemanticDataV2 integration in chat.py

The `v0.3` branch now generates a `SemanticDataV2` object automatically
for every normal user query in `chat.py`.

The runtime flow is now:

```text
User query
   |
Semantic Router
   |
Known / Review / Unknown gate
   |
runtime signals
   |
semantic_runtime_v2.py
   |
SemanticDataV2
   |
automatic V2> display
```

The native v0.3 chat integration uses only signals that actually exist in the
current runtime:

```text
Top-1 selected label / similarity
Top-2 decision margin
gate state
top route candidates
exact adaptive-memory match, when present
adaptive sample count
adaptive label count
policy/runtime metadata
```

Signals that are not implemented in the v0.3 runtime, such as local-majority
purity, are not synthesized.

Example output:

```text
You> 量子状態の意味を教えて
SEM> GATE_REVIEW  label=science sim=0.883911 margin=0.008111
V2> schema=2.0 confidence=0.883911 uncertainty=0.837780
V2> concepts=science
V2> purpose='量子状態の意味を教えて'
V2> context gate=GATE_REVIEW decision_margin=0.008111
SEM> Ambiguous semantic region. Review or teach a better label.
```

Semantic Data v2 output is enabled by default. It can be controlled at startup:

```powershell
python chat.py --semantic-v2
python chat.py --no-semantic-v2
```

or interactively:

```text
/semantic on
/semantic off
/semantic status
```

Existing adaptive-learning commands remain available:

```text
/learn on|off|status
/teach <label>
/memory
/quit
```

The integration remains backward compatible with the existing semantic router
and memory system while making structured semantic state visible at runtime.


## v0.3.1 Runtime Evidence Visualization

The `v0.3` branch now includes the v0.3.1 runtime-evidence view in
`chat.py`. The goal is to make the reasoning signals behind each
SemanticDataV2 object visible rather than showing only the final selected
semantic class.

For every normal query, the chat runtime now evaluates both:

```text
Adaptive Router = base benchmark + semantic memory
Base Router     = base benchmark only
```

and displays the top adaptive candidates together with memory/base evidence.

Example:

```text
SEM> GATE_REVIEW  label=science sim=0.801727 margin=0.012000
V2> schema=2.0 confidence=0.801727 uncertainty=0.760000
V2> concepts=science
V2> purpose='暗号とは'
V2> context gate=GATE_REVIEW decision_margin=0.012000
V2> candidates:
    1. science      0.801727
    2. computer     0.789727
    3. weather      0.742100
V2> memory=(none) base=science (0.798400) selected=science
V2> disagreement=False
V2> gate_reason=margin 0.012000 < threshold 0.020000
```

After an exact `/teach` memory entry exists, the view can expose a
memory/base disagreement explicitly:

```text
V2> memory=computer base=science (...) selected=computer
V2> disagreement=True
```

This distinction is important because it separates:

```text
what the frozen/base semantic space suggests
from
what adaptive semantic memory has taught the runtime
```

The v0.3.1 view uses real runtime evidence only. It does not synthesize
signals that are not implemented in v0.3.

The gate reason is also printed explicitly:

```text
UNKNOWN_KNOWLEDGE -> similarity below similarity threshold
GATE_REVIEW       -> Top-1 / Top-2 margin below margin threshold
ACCEPT            -> both threshold conditions satisfied
```

This provides the observation layer needed for the next experiment:
measuring SemanticDataV2 changes before and after `/teach`.


## v0.3.2 Teaching Effect Evaluation

The `v0.3` branch now evaluates the effect of `/teach` immediately after a
new semantic-memory entry is added.

For every normal query, `chat.py` stores a pre-teaching snapshot containing:

```text
gate
selected label
Top-1 similarity
Top-1 / Top-2 margin
Top-3 candidates
exact memory label
base-only label / similarity
base-memory disagreement
```

When the user executes:

```text
/teach <label>
```

the adaptive router and policy thresholds are rebuilt, the previous utterance
is routed again, and the before/after states are compared automatically.

Example workflow:

```text
You> 暗号
SEM> UNKNOWN_KNOWLEDGE  label=science ...

You> /teach computer
Learned: label='computer', text='暗号' ...

TCH> Teaching Effect Evaluation
TCH> text='暗号' taught_label=computer
TCH> before gate=UNKNOWN_KNOWLEDGE selected=science ...
TCH> after  gate=... selected=...
TCH> delta_similarity=...
TCH> delta_margin=...
TCH> gate_transition=UNKNOWN_KNOWLEDGE->...
TCH> label_transition=science->...
TCH> memory_before=(none) memory_after=computer
TCH> base=science (...) disagreement_after=True
TCH> candidates_before:
     ...
TCH> candidates_after:
     ...
```

When Semantic Data v2 output is enabled, the post-teaching state is also
converted immediately into a new `SemanticDataV2` object and displayed.

This experiment separates three effects:

```text
Frozen/base semantic space
        vs
Adaptive router after teaching
        vs
Persistent exact semantic memory
```

The reported deltas are descriptive runtime measurements. Cosine similarity is
not treated as a calibrated probability.

Because the adaptive sample is included when policy thresholds are rebuilt,
the before/after comparison measures the complete runtime effect of teaching,
including both centroid changes and any resulting threshold changes.


## v0.3.3 Memory-Aware Gate / Explicit Teaching Override

The `v0.3` branch now gives an exact explicit semantic-memory match authority
over the ordinary centroid similarity thresholds.

Decision order:

```text
1. Exact semantic-memory match?
       |
       +-- yes --> ACCEPT_MEMORY
       |           selected = taught memory label
       |           decision_source = explicit_memory
       |
       +-- no  --> ordinary centroid gate
                   ACCEPT / GATE_REVIEW / UNKNOWN_KNOWLEDGE
```

This solves the v0.3.2 observation where a manually taught exact utterance
could still remain `UNKNOWN_KNOWLEDGE` because its adaptive-centroid
similarity was below the normal Known threshold.

The memory override does not erase conflicting base evidence. When the frozen
base router and explicit semantic memory disagree, the runtime records:

```text
memory_exact=True
decision_source=explicit_memory
disagreement=True
review_recommended=True
```

Example:

```text
You> 暗号
SEM> ACCEPT_MEMORY  label=computer ...

V2> memory=computer base=science (...) selected=computer
V2> disagreement=True
V2> memory_exact=True decision_source=explicit_memory review_recommended=True
V2> gate_reason=exact semantic-memory match overrides centroid thresholds

SEM> Accepted by exact semantic memory: computer
SEM> Base/memory disagreement detected; review is recommended.
```

The SemanticDataV2 runtime adapter now stores explicit memory provenance in
both context and relations. Exact memory matches use the relation predicate:

```text
query --explicit_memory_match--> <label>
```

rather than the weaker `memory_candidate` relation.

This design separates two questions:

```text
"What label was explicitly taught for this exact utterance?"
from
"What label does the frozen/base semantic space currently suggest?"
```

The centroid similarity is still retained as evidence and is not replaced by
a synthetic probability. `ACCEPT_MEMORY` therefore means authoritative
retrieval of an explicit teaching record, not probabilistic certainty that the
teaching itself is objectively correct.
