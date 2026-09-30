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


## v0.3.4 Purpose / Intent Extraction

The `v0.3` branch now separates the semantic topic from user intent and
purpose before constructing `SemanticDataV2`.

A lightweight rule-based extractor is implemented in:

```text
semantic_intent_v034.py
```

The extractor produces:

```text
Concept(s)
Intent
Purpose
Extraction rule
Extraction confidence
```

Examples:

```text
Input   : CPUとは
Concept : CPU
Intent  : definition
Purpose : explain_definition(CPU)

Input   : GPUについて教えて
Concept : GPU
Intent  : explain
Purpose : explain(GPU)
```

When no reliable rule matches, the extractor falls back conservatively:

```text
Concept : original input
Intent  : unspecified
Purpose : original input
Rule    : fallback
```

The chat runtime now combines the extracted concept with the selected semantic
route label. This preserves both the user-level concept and the routing-level
semantic class inside `SemanticDataV2`.

Example output:

```text
You> CPUとは
SEM> ACCEPT  label=computer ...

V2> concepts=CPU, computer
V2> purpose='explain_definition(CPU)' intent=definition
V2> extraction rule=definition-pattern confidence=0.950
```

For an explicitly taught query such as `暗号`, the memory-aware gate from
v0.3.3 remains active. Purpose/intent extraction is therefore orthogonal to
routing authority and provenance.

Current design:

```text
User Input
   |
   +--> Purpose / Intent Extractor
   |       +-- Concept
   |       +-- Intent
   |       +-- Purpose
   |
   +--> Semantic Router / Memory-Aware Gate
           +-- semantic class
           +-- provenance
           +-- confidence / uncertainty
   |
   +--------------------------+
                              |
                              v
                       SemanticDataV2
```

v0.3.4 intentionally uses transparent rules rather than a learned intent head.
This provides a reproducible baseline for later comparison with learned
purpose/intent extraction.


## v0.3.5 Automatic Relation Generation

The `v0.3` branch now generates explicit semantic relations automatically
from the Concept / Intent / Purpose structure produced by v0.3.4.

New file:

```text
semantic_relations_v035.py
```

The relation generator currently maps intents as follows:

```text
definition  -> query --requests_definition_of--> concept
explain     -> query --requests_explanation_of--> concept
how_to      -> query --requests_how_to_for--> concept
why         -> query --requests_reason_for--> concept
query       -> query --requests_information_about--> concept
unspecified -> query --mentions--> concept
```

It also emits purpose bindings:

```text
query --has_purpose--> purpose
purpose --targets_concept--> concept
```

Example:

```text
Input   : CPUとは
Concept : CPU
Intent  : definition
Purpose : explain_definition(CPU)

Generated relations:

query --requests_definition_of--> CPU
query --has_purpose--> explain_definition(CPU)
explain_definition(CPU) --targets_concept--> CPU
```

These semantic relations are merged with the existing runtime/provenance
relations:

```text
query --selected_label--> computer
query --base_candidate--> computer
query --route_candidate--> computer
...
```

The two relation families remain conceptually distinct:

```text
semantic relations  = what the user means/wants
runtime relations   = how the router/memory system decided
```

### Concept / routing-evidence separation

v0.3.5 also removes routing labels from `Concept Vectors[]`.

Before:

```text
V2> concepts=暗号, science, animal
```

After:

```text
V2> concepts=暗号
```

Routing labels are retained in `relations[]` and `context` instead of being
misclassified as semantic concepts.

This produces a cleaner structure:

```text
SemanticDataV2
  |
  +-- Concept Vectors[]       semantic concepts only
  +-- Purpose / Intent        requested action
  +-- Semantic Relations[]    meaning structure
  +-- Runtime Relations[]     routing / memory provenance
  +-- Context                 gate / thresholds / runtime state
```

The chat runtime prints all relations with their confidence values.

A small regression test is included:

```powershell
python run_semantic_relations_regression_v035.py
```

It covers definition, explanation, how-to, and fallback relation generation.


## v0.3.5.1 Intent Rule Priority Fix

The rule order in the Purpose / Intent extractor has been corrected so that
specific action patterns are evaluated before generic explanation patterns.

Before:

```text
Pythonの使い方を教えて
-> intent=explain
-> concept=Pythonの使い方を
```

After:

```text
Pythonの使い方を教えて
-> intent=how_to
-> concept=Python
-> purpose=how_to(Python)
-> relation=query --requests_how_to_for--> Python
```

The priority is now:

```text
how_to
definition
why
explain
fallback
```

The regression suite has also been expanded to cover:

```text
CPUとは
GPUについて教えて
Pythonの使い方を教えて
CUDAの設定方法を教えて
なぜGPUは高速ですか
暗号
```

Run:

```powershell
python run_semantic_relations_regression_v035.py
```

The expected result is:

```text
Result: 6/6 passed
```


## v0.3.6 Proposition / Subject-Predicate Relation Extraction

The `v0.3` branch now extracts a lightweight proposition structure from
Japanese concept phrases before building `SemanticDataV2`.

New file:

```text
semantic_proposition_v036.py
```

Primary example:

```text
Input       : なぜGPUは高速ですか
Concept     : GPUは高速
Intent      : why
Proposition : GPU --has_property--> 高速
Purpose     : explain_reason(GPU, 高速)
```

The proposition extractor uses transparent Japanese particle rules. The first
baseline supports:

```text
は / が -> subject + predicate phrase
を      -> implicit_subject --acts_on--> object
に      -> implicit_subject --targets--> object
で      -> implicit_subject --context_of_action--> object
と      -> implicit_subject --related_with--> object
```

For topic/subject forms, simple property expressions such as `高速`,
`安全`, `重要`, `可能`, etc. are normalized as:

```text
subject --has_property--> property
```

More general predicate phrases are retained as:

```text
subject --has_predicate--> predicate_phrase
```

Example:

```text
CPUは命令を実行する
-> CPU --has_predicate--> 命令を実行する
```

These proposition relations are merged with the existing two relation families:

```text
semantic relations   = what the user asks for
proposition relations= what semantic statement is being referred to
runtime relations    = how the router/memory system decided
```

For `why` queries, Purpose is refined when a proposition is available:

```text
Before:
explain_reason(GPUは高速)

After:
explain_reason(GPU, 高速)
```

A regression test is included:

```powershell
python run_semantic_proposition_regression_v036.py
```

Expected result:

```text
Result: 3/3 passed
```

The extractor is intentionally conservative. If no proposition is recognized,
the existing v0.3.5 semantic relation behavior is preserved.


## v0.3.6.1 Refined Purpose Consistency + v0.3.7 Proposition-Aware Concepts

Two related improvements are now integrated.

First, the refined Purpose produced by proposition extraction is reused
consistently by the Semantic Relations layer.

Example:

```text
Input    : なぜGPUは高速ですか
Purpose  : explain_reason(GPU, 高速)
```

The same refined Purpose is now used by:

```text
SemanticDataV2.purpose
query --has_purpose--> explain_reason(GPU, 高速)
explain_reason(GPU, 高速) --targets_concept--> ...
```

This removes the previous split between:

```text
explain_reason(GPU, 高速)
and
explain_reason(GPUは高速)
```

Second, v0.3.7 uses proposition structure to split Concept Vectors[] into
semantic components.

Examples:

```text
GPUが高速
Before concepts : [GPUが高速]
After concepts  : [GPU, 高速]

CPUは命令を実行する
Before concepts : [CPUは命令を実行する]
After concepts  : [CPU, 命令を実行する]
```

The proposition itself remains explicit:

```text
GPU --has_property--> 高速
CPU --has_predicate--> 命令を実行する
```

This is the first stage where Semantic Data v2.0 uses multiple concept vectors
derived from the internal proposition structure rather than merely storing one
vector for the full phrase.

The proposition regression now verifies:

```text
- proposition extraction
- refined Purpose
- Purpose/Relation consistency
- proposition-aware Concept separation
```

Run:

```powershell
python run_semantic_proposition_regression_v036.py
```

Expected result:

```text
Result: 3/3 passed
```


## v0.3.9 Proposition Vector / Relation Vector

The v0.4 branch now extends Semantic Data v2.0 so that structural relations
and proposition nodes carry their own semantic vectors.

The representation now includes:

```text
Global Vector
Concept Vectors[]
Purpose Vector
Relation Vectors[]
Proposition Vectors[]
Context
Confidence / Uncertainty
```

### Relation vectors

Every SemanticRelation is now encoded from its canonical textual form:

```text
subject predicate object
```

Example:

```text
GPU has_property 高速
```

This produces a relation vector using the same frozen LLM_SEM encoder as the
global, concept, and purpose vectors.

### Proposition vectors

A proposition node now has an explicit vectorized representation:

```text
proposition_1
  subject   = GPU
  predicate = has_property
  object    = 高速
  vector    = SemanticVector(...)
```

The proposition vector is encoded from:

```text
GPU has_property 高速
```

The structural graph and the vector representation therefore coexist:

```text
query --requests_reason_for--> proposition_1
proposition_1 --subject--> GPU
proposition_1 --predicate--> has_property
proposition_1 --object--> 高速
GPU --has_property--> 高速
```

### chat.py output

The chat runtime now displays vector dimensions for both relation and
proposition vectors:

```text
V2> relations:
    GPU --has_property--> 高速 [0.920] vec=64

V2> propositions:
    proposition_1: GPU --has_property--> 高速 vec=64
```

### Regression

Run:

```powershell
python run_semantic_vector_regression_v039.py
```

Expected:

```text
[PASS] global-vector
[PASS] concept-vectors
[PASS] purpose-vector
[PASS] relation-vectors
[PASS] proposition-vector
Result: 5/5 passed
```

This is the first Semantic Data v2.0 stage where topic, concepts, purpose,
relations, and propositions are all represented independently in vector space
while remaining connected through the semantic graph.


## v0.4.0 Semantic Structural Consistency Evaluation

The v0.4 branch now includes a structural consistency experiment for the
Semantic Data v2.0 proposition representation.

New script:

```powershell
python run_semantic_structural_consistency_v040.py
```

The experiment asks whether an independently composed semantic vector built
from:

```text
Subject Vector
Predicate Vector
Object Vector
```

is closer to the correct proposition vector than to counterfactual
propositions in which one structural element is replaced.

For example:

```text
Positive:
GPU --has_property--> 高速

Counterfactual subject:
CPU --has_property--> 高速

Counterfactual predicate:
GPU --has_predicate--> 高速

Counterfactual object:
GPU --has_property--> 低速
```

For each case the script reports:

```text
Positive similarity
Counterfactual subject similarity
Counterfactual predicate similarity
Counterfactual object similarity
Best counterfactual similarity
Structural margin
PASS / FAIL
```

The structural margin is:

```text
positive_similarity - max(counterfactual_similarities)
```

A positive margin means the frozen semantic encoder preserves enough
structure for the correct proposition to be more compatible with the
independently encoded subject/predicate/object components than all tested
one-element counterfactuals.

The script also prints component-alignment diagnostics for Subject,
Predicate, and Object versus the positive proposition vector.

This experiment is diagnostic. Similarity values are not calibrated
probabilities, and a PASS does not imply logical correctness. The goal is to
measure whether structural distinctions are present in the current frozen
semantic space strongly enough to support future proposition-aware routing,
gating, or projection learning.


## v0.4.1 Structural Role Encoding / Projection

v0.4.1 adds trainable role-specific projections for Subject, Predicate, and
Object while keeping the base LLM_SEM semantic encoder frozen.

The structural composition changes from a simple role-agnostic mean:

```text
mean(Subject, Predicate, Object)
```

to:

```text
normalize(
    W_subject(Subject)
  + W_predicate(Predicate)
  + W_object(Object)
)
```

Each projection is a learnable 64 -> 64 linear transformation initialized as
identity.

Training uses the same positive/counterfactual structure introduced in v0.4.0.
For every proposition, the projected composition is optimized so that its
similarity to the correct proposition exceeds the similarity to one-element
counterfactuals by a target margin.

The loss is a margin-ranking objective:

```text
max(0, target_margin - sim(projected, positive) + sim(projected, counterfactual))
```

The base Transformer remains completely frozen. Only the three structural role
projection matrices are trained.

Run:

```powershell
python run_structural_role_projection_v041.py
```

The experiment currently uses:

```text
Seeds         : 1, 2, 3, 4, 5
Epochs        : 250
Learning rate : 1e-3
Target margin : 0.10
```

The output reports, for every seed:

```text
before positive-margin cases / mean margin
after  positive-margin cases / mean margin
minimum post-training margin
```

The best seed is selected lexicographically by:

```text
positive-margin case count
mean structural margin
minimum structural margin
```

The best projection is saved as:

```text
model/structural-role-projection-v041.pt
```

The primary evaluation target is to improve the v0.4.0 baseline from:

```text
Positive-margin cases : 2/4
Mean structural margin: -0.014815
```

toward:

```text
Positive-margin cases : 4/4
Mean structural margin: > 0
```

This experiment tests whether explicit role encoding can recover structural
distinctions that are only partially represented in the raw frozen semantic
space.


## v0.4.2 Structural Generalization / Holdout Evaluation

v0.4.2 adds a strict train/holdout protocol to test whether Structural Role
Projection learns reusable role binding rather than memorizing the propositions
used during optimization.

New script:

```powershell
python run_structural_generalization_holdout_v042.py
```

### Protocol

The experiment separates propositions into:

```text
TRAIN   : 8 propositions
HOLDOUT : 6 propositions
```

Holdout propositions are never included in the optimizer loss.

The base LLM_SEM encoder remains frozen. Only the Subject / Predicate / Object
projection matrices are trained.

The training loss combines structural ranking and semantic preservation:

```text
Loss =
    Margin Ranking Loss
  + lambda * Preservation Loss
```

where:

```text
Margin Ranking Loss
= max(0, target_margin - sim(projected, positive)
                     + sim(projected, counterfactual))

Preservation Loss
= 1 - sim(projected, positive proposition)
```

Current settings:

```text
Seeds               : 1, 2, 3, 4, 5
Epochs              : 300
Learning rate       : 5e-4
Target margin       : 0.08
Preservation lambda : 0.35
Initialization      : identity + small seed-dependent noise
```

Unlike v0.4.1, seed-dependent initialization makes the multi-seed experiment
non-degenerate.

### No holdout model selection

The best seed is selected using TRAIN metrics only:

```text
positive-margin case count
mean structural margin
minimum structural margin
mean semantic preservation
```

HOLDOUT metrics are never used to select or train the model.

This avoids leaking evaluation information into model selection.

### Output

The script reports before/after metrics for both TRAIN and HOLDOUT:

```text
Positive-margin cases
Mean structural margin
Minimum structural margin
Mean semantic preservation
```

The strongest evidence for structural learning is:

```text
TRAIN improves
AND
HOLDOUT also improves
```

If TRAIN improves strongly while HOLDOUT does not, the result remains
compatible with memorization or overfitting.

The best train-selected model is saved as:

```text
model/structural-role-projection-v042.pt
```


## v0.4.3 Relation-Unseen Structural Generalization

v0.4.3 strengthens the structural generalization test by holding out entire
relation labels from training.

New script:

```powershell
python run_relation_unseen_generalization_v043.py
```

### Train relations

The projection is trained only with:

```text
has_property
targets
related_with
```

### Unseen relations

The following relation labels are never used in optimization:

```text
is_a
has_predicate
used_for
part_of
causes
```

### Three evaluation splits

The experiment reports three increasingly strict conditions:

```text
A) Seen relation / unseen proposition
B) Unseen relation / seen concepts
C) Unseen relation / unseen concepts
```

A tests whether the model generalizes to new propositions using familiar
relation labels.

B tests whether role projection can handle a relation label never observed
during training while reusing familiar concepts.

C is the strictest test. Both the relation label and the core concepts are
unseen during training.

### Leakage control

Holdout examples are excluded from:

```text
optimizer loss
seed selection
checkpoint selection
```

The best seed is selected using TRAIN metrics only.

### Interpretation

Strong evidence for structural role learning requires improvement not only in A
but also in B, and preferably C.

If only A improves, the projection may still depend strongly on learned
relation labels.

If B improves, there is evidence that Subject / Predicate / Object role
encoding transfers to unseen predicates.

If C also improves, the result is stronger evidence that the projection has
learned reusable structural composition rather than memorized specific
relations or concept combinations.

The selected checkpoint is saved as:

```text
model/structural-role-projection-v043.pt
```


## v0.4.4 Role Ablation / Causal Contribution Analysis

v0.4.4 evaluates the causal contribution of the learned Subject, Predicate,
and Object pathways in the v0.4.3 Structural Role Projection.

New script:

```powershell
python run_role_ablation_v044.py
```

The analysis loads the trained checkpoint:

```text
model/structural-role-projection-v043.pt
```

and performs no retraining.

The following variants are compared:

```text
Full
  Ws(S) + Wp(P) + Wo(O)

No Subject
  Wp(P) + Wo(O)

No Predicate
  Ws(S) + Wo(O)

No Object
  Ws(S) + Wp(P)
```

Each ablation is evaluated on the same three holdout splits introduced in
v0.4.3:

```text
A) Seen relation / unseen proposition
B) Unseen relation / seen concepts
C) Unseen relation / unseen concepts
```

For each split the script reports:

```text
Positive-margin cases
Mean structural margin
Minimum margin
Mean semantic preservation
Delta margin versus Full
Lost positive-margin cases
```

The causal contribution of one role is defined as:

```text
Role contribution
= Full mean structural margin
  - Ablated mean structural margin
```

Interpretation:

```text
positive contribution
  -> removing the role hurts structural discrimination

near zero
  -> role is largely redundant under the current evaluation

negative contribution
  -> removing the role improves margin, suggesting interference or
     over-reliance elsewhere
```

A final cross-split table reports the Subject, Predicate, and Object
mean-margin drops across A/B/C and identifies the largest average contributor.

Results are saved to:

```text
results/role_ablation_v044.pt
```


## v0.4.5 Role Interaction / Synergy Analysis

v0.4.5 evaluates interaction effects among Subject, Predicate, and Object
pathways using the fixed v0.4.3 projection checkpoint.

New script:

```powershell
python run_role_synergy_v045.py
```

No retraining is performed.

The following seven non-empty role combinations are compared:

```text
Subject only
Predicate only
Object only

Subject + Predicate
Subject + Object
Predicate + Object

Subject + Predicate + Object
```

The same three holdout splits are used:

```text
A) Seen relation / unseen proposition
B) Unseen relation / seen concepts
C) Unseen relation / unseen concepts
```

### Pair synergy

For a role pair A+B:

```text
pair_synergy(A,B)
= margin(A+B) - max(margin(A), margin(B))
```

Positive pair synergy means the combination provides more structural margin
than either constituent role alone.

### Triple synergy

For the Full model:

```text
triple_synergy
= margin(Subject+Predicate+Object)
  - max(
      margin(Subject+Predicate),
      margin(Subject+Object),
      margin(Predicate+Object)
    )
```

Positive triple synergy means all three roles together outperform every
two-role configuration.

### Output

For every split the script reports:

```text
Positive-margin cases
Mean structural margin
Minimum margin
Mean semantic preservation
```

for all seven variants, followed by:

```text
Subject + Predicate synergy
Subject + Object synergy
Predicate + Object synergy
Triple synergy
```

A final cross-split table reports the mean synergy across A/B/C.

Results are saved to:

```text
results/role_synergy_v045.pt
```


## v0.6.1 Unified Semantic Runtime

v0.6.1 integrates the previously separate semantic subsystems into the
interactive `chat.py` runtime.

Integrated path:

```text
User utterance
    |
    +--> Fixed Base Router
    |
    +--> Adaptive Semantic Memory
    |      +--> Multi-Prototype memory
    |      +--> Local k-NN evidence
    |      +--> Conditional adaptive override
    |
    +--> Semantic Data v2.0
           +--> concept vectors
           +--> purpose / intent
           +--> semantic relations
           +--> proposition vectors
           +--> confidence / uncertainty
           +--> runtime provenance
```

The base Transformer remains frozen. Adaptive decisions use the v0.4.6 policy:

```text
normal accept:
  memory_similarity >= 0.80
  AND adaptive label == base label

adaptive override:
  adaptive label != base label
  AND memory_similarity >= 0.92
  AND local majority == adaptive label
  AND local purity >= 0.60
```

Default runtime parameters:

```text
prototypes / label : 2
local k            : 3
local purity       : 0.60
memory similarity  : 0.80
override similarity: 0.92
```

New runtime command:

```text
/runtime
```

This prints the active unified-runtime policy.

The normal chat path now reports both the semantic routing result and local
adaptive evidence. A high-confidence disagreement supported by local evidence
is surfaced as:

```text
ACCEPT_ADAPTIVE
```

Semantic Data v2 continues to record concepts, purpose, relations,
propositions, confidence, uncertainty, gate state, decision margin, and
runtime provenance.

### Regression

A fast integration regression was added:

```powershell
python run_v061_regression.py
```

It does not load the full language-model checkpoint. It verifies:

```text
chat.py syntax
v0.6.1 integration markers
Semantic Data v2 wiring
relation / proposition wiring
multi-prototype adaptive runtime wiring
local-evidence ACCEPT behavior
conditional ADAPTIVE_OVERRIDE behavior
safe fallback / review behavior
required runtime files
```

A successful run ends with:

```text
RESULT: PASS
```

This branch therefore establishes a single runtime path from adaptive semantic
memory through structured Semantic Data, providing the baseline for the next
stage: Semantic Composition.


## v0.6.2 Semantic Composition Experiment

The next experiment moves from structural-role detection to proposition-level
semantic composition.

New script:

```powershell
python run_semantic_composition_v062.py
```

The experiment compares four composition methods:

```text
1) simple_mean
   mean(normalized Subject, Predicate, Object)

2) role_aware
   frozen v0.4.3 Subject / Predicate / Object role projection

3) relation_weighted
   frozen role projection with stronger Predicate/Relation contribution

4) learned_comp
   train-only neural composition head over frozen role-projected vectors
```

### Evaluation protocol

The existing v0.4.3 splits are reused without modification:

```text
TRAIN : seen relations / training propositions
A     : seen relation / unseen proposition
B     : unseen relation / seen concepts
C     : unseen relation / unseen concepts
```

Holdout leakage is explicitly avoided:

```text
A/B/C in optimization : False
A/B/C in seed select  : False
```

The learned composition head is trained only on TRAIN cases. The best seed is
selected only from TRAIN metrics.

### Main metric

For each proposition, the composed vector is compared with:

```text
positive proposition
subject counterfactual
predicate counterfactual
object counterfactual
```

The structural counterfactual margin is:

```text
margin
= similarity(composed, positive)
  - max(similarity(composed, counterfactuals))
```

Positive margin means the composed semantic representation is closer to the
correct proposition than to all single-role replacements.

Reported metrics:

```text
positive-margin cases
mean margin
minimum margin
semantic preservation
```

A final cross-split table aggregates A/B/C and is intended to distinguish:

```text
memorized composition
vs.
transferable semantic composition
```

The result is saved to:

```text
results/semantic_composition_v062.pt
```

Interpretation:

```text
TRAIN gain only
  -> likely proposition-pattern overfitting

A gain
  -> unseen proposition composition

B gain
  -> unseen relation transfer

C gain
  -> strongest evidence for reusable semantic composition
```


## v0.6.3 Adaptive Semantic Composition Gate

v0.6.3 tests whether proposition composition should change according to
semantic novelty instead of using one fixed Subject / Predicate / Object
weighting for every input.

New script:

```powershell
python run_adaptive_semantic_composition_v063.py
```

The experiment is an exploratory follow-up to v0.6.2.

### Fixed composition modes

```text
balanced
  S=0.333  P=0.333  O=0.333

relation_aware
  S=0.25   P=0.50   O=0.25

concept_aware
  S=0.40   P=0.20   O=0.40
```

### Adaptive gate

The gate uses only novelty relative to TRAIN_CASES:

```text
known relation
  -> balanced

unseen relation + subject/object both seen in TRAIN
  -> relation_aware

unseen relation + at least one novel concept
  -> balanced
```

The gate does not inspect holdout margins, positive vectors, or
counterfactual scores when selecting a mode.

Compared methods:

```text
simple_mean
balanced
relation_aware
concept_aware
adaptive_gate
```

The same A/B/C holdouts are retained.

For every adaptive decision the script prints:

```text
relation seen/unseen
subject seen/unseen
object seen/unseen
selected composition mode
decision reason
```

Results are saved to:

```text
results/adaptive_semantic_composition_v063.pt
```

### Scientific status

This experiment is explicitly marked:

```text
exploratory / post-hoc
```

because its rule is motivated by the v0.6.2 observations. If the adaptive gate
improves the existing holdout result, a new independently designed holdout set
will be required for confirmatory evaluation.


## v0.6.4 Confirmatory Adaptive Semantic Composition

v0.6.4 freezes the v0.6.3 adaptive composition rule and evaluates it on new
confirmatory holdout cases.

New script:

```powershell
python run_adaptive_composition_confirmatory_v064.py
```

The v0.6.3 gate is not modified:

```text
known relation
  -> balanced

unseen relation + subject/object both seen in TRAIN
  -> relation_aware

otherwise
  -> balanced
```

No gate training, composition-weight tuning, threshold tuning, or seed
selection is performed.

### New confirmatory splits

```text
D) known relation / new concepts
E) unseen relation / seen concepts
F) unseen relation / partially seen concepts
G) unseen relation / new concepts
```

Compared methods:

```text
balanced
relation_aware
adaptive_gate
```

The primary confirmatory criterion is:

```text
adaptive_gate positive cases >= balanced positive cases
AND
adaptive_gate mean margin >= balanced mean margin
```

The script reports:

```text
Positive-case delta
Mean-margin delta
Confirmed: True / False
```

A `Confirmed=True` result means the frozen v0.6.3 policy replicated on the
newly introduced holdout set. A `Confirmed=False` result means the v0.6.3
gain remains exploratory and should not be promoted to the runtime.

Results are saved to:

```text
results/adaptive_composition_confirmatory_v064.pt
```


## v0.6.5 Adaptive Composition Runtime Integration

v0.6.5 promotes the adaptive composition policy confirmed in v0.6.4 from an
experiment into the interactive runtime.

New runtime module:

```text
adaptive_composition_runtime_v065.py
```

The confirmed policy remains frozen:

```text
known relation
  -> balanced
  -> S=0.333 P=0.333 O=0.333

unseen relation + subject/object both seen in TRAIN
  -> relation_aware
  -> S=0.25 P=0.50 O=0.25

otherwise
  -> balanced
```

### Runtime path

```text
extracted proposition
    |
    +--> subject vector
    +--> predicate vector
    +--> object vector
    |
    +--> frozen v0.4.3 structural role projection
    |
    +--> v0.6.5 novelty gate
    |
    +--> balanced / relation_aware composition
    |
    +--> adaptive proposition vector
    |
    +--> SemanticDataV2.propositions[]
```

`chat.py` now enables adaptive proposition composition by default.

Relevant options:

```text
--composition / --no-composition
--composition-checkpoint model/structural-role-projection-v043.pt
```

Each proposition records provenance:

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

The interactive Semantic Data v2 summary also displays the composition mode,
weights, novelty state, vector role, and composition confidence.

Semantic Data v2 now accepts a precomputed proposition vector through the
proposition specification. If no precomputed vector is supplied, the previous
text-encoded proposition behavior remains unchanged for backward
compatibility.

### Regression

Run:

```powershell
python run_v065_regression.py
```

The regression checks:

```text
chat.py integration
composition runtime import
composition CLI flag
proposition enrichment
precomputed vector support
composition provenance export
frozen confirmed gate behavior
known-relation balanced mode
unseen-relation/seen-concepts relation-aware mode
partial/new-concept balanced fallback
```

A successful run ends with:

```text
RESULT: PASS
```


## v0.6.6 Proposition Runtime Validation

v0.6.6 validates the actual runtime path that was not exercised by
definition-only chat inputs:

```text
proposition extraction
    -> adaptive composition
    -> adaptive proposition vector
    -> SemanticDataV2
```

New script:

```powershell
python run_composition_runtime_validation_v066.py
```

Representative cases include:

```text
GPUは高速
  -> has_property
  -> balanced

GPUはcomputer
  -> has_predicate
  -> relation_aware

GPUは計算装置
  -> has_predicate
  -> balanced
```

The script verifies proposition extraction, expected predicate,
composition mode, generated vector length, SemanticDataV2 proposition
creation, vector role, model type, and provenance.

A successful run ends with:

```text
RESULT: PASS
```

Also, remaining v0.6.1 labels in `chat.py` help/description text were updated
to v0.6.5.


## v0.6 Series Summary

The v0.6 series establishes Semantic Composition as a validated runtime
capability rather than only an offline experiment.

### Progression

```text
v0.6.1  Unified Semantic Runtime
        Adaptive Memory + Local Evidence + Semantic Data v2

v0.6.2  Semantic Composition
        Compare simple mean, role-aware, relation-weighted,
        and learned proposition composition

v0.6.3  Adaptive Composition
        Select composition policy from semantic novelty

v0.6.4  Confirmatory Holdout
        Freeze the v0.6.3 gate and verify it on new D/E/F/G cases

v0.6.5  Runtime Promotion
        Move the confirmed adaptive composition policy into chat.py

v0.6.6  Runtime Validation
        Validate proposition extraction -> adaptive composition
        -> adaptive proposition vector -> SemanticDataV2
```

### Key measured results

v0.6.2 established that structural role-aware composition generalized better
than a simple vector mean and better overall than the train-only learned
composer on the original A/B/C holdouts:

```text
Cross-split A/B/C

simple_mean      4/11   mean margin -0.009362
role_aware      10/11   mean margin +0.012679
relation_weighted
                 9/11   mean margin +0.006192
learned_comp     9/11   mean margin +0.010585
```

v0.6.3 then introduced the adaptive gate. On the same exploratory A/B/C
evaluation it preserved 10/11 positive cases and increased mean margin:

```text
balanced       10/11   +0.012679
adaptive_gate  10/11   +0.013768
```

Because that rule was designed after inspecting v0.6.2, v0.6.4 evaluated the
frozen policy on new confirmatory D/E/F/G cases:

```text
balanced       10/13   +0.005945
adaptive_gate  10/13   +0.006517

Positive-case delta : +0
Mean-margin delta   : +0.000572
Confirmed           : True
```

This confirmed that the adaptive gate could improve structural margin without
reducing the number of positive-margin cases on the new holdout.

### Confirmed runtime policy

The runtime policy promoted in v0.6.5 is:

```text
known relation
  -> balanced
  -> S=0.333 P=0.333 O=0.333

unseen relation + subject/object both seen in TRAIN
  -> relation_aware
  -> S=0.25 P=0.50 O=0.25

otherwise
  -> balanced
```

The policy remains deliberately interpretable and non-neural.

### Current runtime architecture

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

### Regression and runtime validation

The following checks are available:

```powershell
python run_v061_regression.py
python run_v065_regression.py
python run_composition_runtime_validation_v066.py
```

Observed results:

```text
v0.6.1 regression : PASS
v0.6.5 regression : PASS
v0.6.6 runtime validation : PASS
```

The v0.6.6 validation exercised actual proposition inputs and verified both
composition branches:

```text
GPUは高速
  -> has_property
  -> balanced

GPUはcomputer
  -> has_predicate
  -> relation_aware

GPUは計算装置
  -> has_predicate
  -> balanced
```

### v0.6 conclusion

The v0.6 series supports the following implementation-level conclusion:

```text
Semantic meaning is not represented only as one vector.

It is represented as:
  semantic components
  + structural roles
  + explicit relations
  + proposition structure
  + adaptive composition policy
  + composed proposition vectors
  + provenance
```

This provides the baseline for v0.7, where the next research target is to move
from single-proposition composition to multi-proposition semantic structure and
Semantic Graph processing.
