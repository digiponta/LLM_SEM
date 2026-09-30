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


## Semantic Data v2.0

Branch:

```text
v0.5-semantic-data-v2
```

Semantic Data v2.0 extends the original single-vector representation into a
structured semantic object.

```text
Semantic Data v2.0
  |
  +-- global_vector
  |
  +-- concept_vectors[]
  |     +-- name
  |     +-- vector
  |     +-- role
  |     +-- attributes
  |
  +-- purpose
  |     +-- text
  |     +-- intent
  |     +-- vector
  |
  +-- relations[]
  |     +-- subject
  |     +-- predicate
  |     +-- object
  |
  +-- context
  |     +-- text
  |     +-- optional context vector
  |     +-- attributes
  |
  +-- confidence
  +-- uncertainty
  +-- metadata
```

The design separates several semantic signals that should not be collapsed
into one embedding:

- `global_vector` is the document/query-level vector used by the current
  semantic router.
- `concept_vectors[]` preserves multiple concept/entity representations.
- `purpose.vector` represents intended action/goal separately from topic.
- `relations[]` stores explicit directional structure and bindings.
- `context` stores execution/domain/state information and may also have a
  vector.
- `confidence` and `uncertainty` expose quality information for gates and
  QHA scheduling.

Relations are structural data in v2.0. They are not forced into a relation
embedding yet. This keeps the first implementation aligned with the
structured-composition experiments and leaves relation-vector learning as a
separate future experiment.

### Backward compatibility

The original API remains available:

```python
from semantic import encode_text

legacy = encode_text(model, tokenizer, text)
```

New code can produce Semantic Data v2.0:

```python
from semantic import SemanticRelation, encode_semantic_v2

semantic = encode_semantic_v2(
    model,
    tokenizer,
    text,
    concepts=["GPU", "ニューラルネットワーク"],
    purpose_text="高速化する方法を教えて",
    intent="explain",
    relations=[
        SemanticRelation(
            subject="GPU",
            predicate="accelerates",
            object="ニューラルネットワーク",
        )
    ],
    context_text="CUDAを利用する計算環境",
)
```

Existing router/evaluation code can consume the v2 object through its global
vector without modification:

```python
legacy = semantic.as_legacy()
vector = semantic.primary_vector()
```

This makes the migration incremental:

```text
LLM_SEM
   |
Semantic Data v2.0
   |-- global vector --------> existing Semantic Router
   |-- concept vectors ------> semantic structure / memory
   |-- purpose vector -------> QHA task/purpose scheduling
   |-- relations/context ---> Semantic Shell / QHA Semantic OS
   +-- confidence -----------> gate / Unknown / review
```

The encoder is intentionally extraction-agnostic. Concept, purpose, relation,
and context extraction can be supplied by the current adaptive-memory runtime,
rule-based logic, or future learned semantic heads without changing the v2.0
data contract.


## v0.5.1 Semantic Data v2 Runtime Integration

Branch:

```text
v0.5.1-semantic-data-v2-runtime
```

v0.5.1 connects the adaptive semantic runtime signals to the structured
Semantic Data v2.0 contract.

The current runtime already produces signals such as:

```text
memory label / similarity / margin
local majority / purity / k
base label / similarity
prototype labels / scores
gate state
selected label
adaptive sample counts
```

These values are now mapped as follows:

```text
Adaptive Runtime
    |
    +-- memory candidate -----------+
    +-- local majority -------------+
    +-- base candidate -------------+--> relations[]
    +-- selected label -------------+
    +-- prototype labels -----------+
    |
    +-- gate/runtime state ------------> context.attributes
    |
    +-- similarity / purity / margin --> confidence + uncertainty
    |
    +-- query text --------------------> global_vector
    |
    +-- concept labels ----------------> concept_vectors[]
    |
    +-- task/purpose ------------------> purpose.vector
```

The adapter is implemented in:

```text
semantic_runtime_v2.py
```

Primary types/functions:

```python
RuntimeSemanticSignals
build_runtime_semantic_v2(...)
from_runtime_dict(...)
runtime_summary(...)
```

### Minimal chat runtime integration

After the v0.4.x runtime has calculated its normal decision values:

```python
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
    "adaptive_enabled": True,
}
```

convert them with:

```python
from semantic_runtime_v2 import from_runtime_dict

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

The existing router remains unchanged because:

```python
legacy = semantic_v2.as_legacy()
```

still exposes the v2 global vector through the original SemanticData
interface.

### Confidence and uncertainty

v0.5.1 deliberately uses a conservative runtime estimate rather than treating
raw similarity as calibrated probability.

The default order is:

```text
explicit confidence
    -> memory similarity x local purity
    -> base similarity
    -> neutral fallback
```

A small memory margin increases uncertainty, and a gate state containing
`REVIEW` forces elevated uncertainty. These are runtime heuristics, not
probabilistic calibration.

This separation allows later calibration experiments without changing the
Semantic Data v2.0 schema.

### Current integration boundary

`chat_v0412.py` is not currently present in the GitHub repository, so v0.5.1
adds a non-invasive adapter rather than modifying the local chat runtime
directly.

The intended local integration point is immediately after the existing
memory/local/base/gate decision has been calculated. The resulting
`SemanticDataV2` object can then be passed to QHA, Semantic Shell, logging,
or future adaptive-learning stages.
