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


## Independent holdout validation

Branch `v0.1` includes independent validation data that are not used for
threshold selection:

```text
holdout_benchmark.csv
unknown_holdout.csv
```

Run:

```powershell
python semantic_validation.py
```

The validation flow is:

```text
Development data
  my_benchmark.csv
  unknown_benchmark.csv
        |
        v
select policy thresholds
        |
        v
freeze thresholds
        |
        v
Independent holdout data
  holdout_benchmark.csv
  unknown_holdout.csv
        |
        v
final validation metrics
```

The holdout set is never used to optimize the similarity or margin thresholds.

Reported metrics include:

- Known routing accuracy before Unknown rejection
- Known Recall after Unknown rejection
- Known accept rate
- Unknown detection rate
- False Unknown rate
- False Known rate
- Balanced accuracy
- per-category Known and Unknown results

Per-sample results are saved to:

```text
semantic_validation_results.csv
```

Policy examples:

```powershell
python semantic_validation.py --policy known-first
python semantic_validation.py --policy balanced
python semantic_validation.py --policy discovery-first
```

For the constrained balanced policy:

```powershell
python semantic_validation.py --policy balanced --balanced-min-known-recall 0.70
```


## Class-specific semantic radius

Branch `v0.1` now includes an open-set detector based on one semantic radius
per known class.

The radii are learned from the known development benchmark only, using
leave-one-out distances to each class centroid:

```text
class centroid
    |
    +-- animal radius
    +-- weather radius
    +-- computer radius
    +-- food radius
    +-- transport radius
    +-- science radius
```

A new input is classified as Unknown when its distance from the nearest
predicted class centroid exceeds that class's learned radius:

```text
distance(query, nearest centroid) > class radius
    -> Unknown
```

The implementation is in:

```text
semantic_radius.py
```

Independent validation now uses the class-radius detector by default:

```powershell
python semantic_validation.py
```

Equivalent explicit command:

```powershell
python semantic_validation.py --detector class-radius
```

The default radius parameters are:

```text
quantile = 0.90
scale    = 1.00
```

They can be changed without using holdout data for fitting:

```powershell
python semantic_validation.py --detector class-radius --radius-quantile 0.90 --radius-scale 1.10
```

The previous global similarity/margin detector remains available for
comparison:

```powershell
python semantic_validation.py --detector global --policy balanced
```

This allows a direct comparison between global-threshold open-set detection
and class-specific semantic-radius detection on the same independent holdout
benchmarks.


## Semantic space analysis

Branch `v0.1` now includes:

```text
semantic_space_analysis.py
```

Run:

```powershell
python semantic_space_analysis.py
```

The analyzer measures the geometry of the known semantic classes:

- leave-one-out routing accuracy per class
- class-specific semantic radius
- mean and maximum within-class distance
- centroid-to-centroid cosine distance matrix
- nearest competing semantic class
- separation ratio
- overlap risk between class-radius regions
- most common misclassification destination

The separation ratio is defined as:

```text
separation_ratio =
    centroid_distance(class A, class B)
    /
    (radius_A + radius_B)
```

Interpretation:

```text
separation_ratio < 1.0
    -> class-radius regions overlap

separation_ratio >= 1.0
    -> class-radius regions are geometrically separated
```

Output files:

```text
semantic_space_summary.csv
semantic_centroid_distance_matrix.csv
semantic_space_errors.csv
```

These diagnostics are intended to identify which known semantic classes are
poorly separated before changing the semantic-vector representation or adding
semantic-specific training.


## Semantic projection head

Branch `v0.1` now includes a trainable semantic projection stage above the
frozen base language model.

Architecture:

```text
Frozen LLM
  |
64-D raw hybrid semantic vector
  |
SemanticProjectionHead
  64 -> 128 -> 64
  |
L2-normalized projected semantic vector
```

Files:

```text
semantic_projection.py
semantic_projection_train.py
semantic_projection_eval.py
```

The projection head uses a residual MLP and is trained with supervised
contrastive loss. The base LLM checkpoint is never updated.

Train:

```powershell
python semantic_projection_train.py
```

Default output:

```text
model/semantic-projection-v0.1.pt
```

Evaluate before/after projection:

```powershell
python semantic_projection_eval.py
```

The evaluation reports:

- development leave-one-out routing accuracy
- within-class cosine similarity
- between-class cosine similarity
- semantic margin
- mean nearest-centroid distance
- independent holdout Known routing accuracy
- Known Recall
- Unknown Detection Rate
- False Unknown / False Known rates
- Balanced Accuracy

The intended goal is to reduce overlap between known semantic classes while
preserving or improving independent open-set detection.


## Regularized semantic projection training

The projection trainer now includes geometry preservation and early stopping
to reduce overfitting to the small development benchmark.

New training objective:

```text
Total Loss =
    Supervised Contrastive Loss
    +
    preservation_lambda * Preservation Loss
```

where:

```text
Preservation Loss =
    mean(1 - cosine(projected_vector, original_vector))
```

New default settings:

```text
hidden_dim            = 64
epochs                = 100
learning_rate         = 1e-4
preservation_lambda   = 1.0
patience              = 20
min_delta             = 1e-4
```

Train:

```powershell
python semantic_projection_train.py
```

Then evaluate:

```powershell
python semantic_projection_eval.py
```

The trainer prints total loss, contrastive loss, preservation loss, and the
best epoch. Early stopping restores the best projection checkpoint before
saving.

Custom regularization can be tested without modifying source code:

```powershell
python semantic_projection_train.py --preservation-lambda 2.0 --epochs 200 --patience 30
```


## Preservation lambda sweep

Branch `v0.1` now includes a development-only sweep for the projection
geometry-preservation weight:

```text
semantic_projection_sweep.py
```

Run:

```powershell
python semantic_projection_sweep.py
```

Default values:

```text
lambda = 0.0, 0.5, 1.0, 2.0, 5.0
```

For every lambda, the script trains a fresh projection head with the same seed
and reports:

- development leave-one-out routing accuracy
- within-class similarity
- between-class similarity
- semantic margin
- geometry drift from the original semantic vector
- best training epoch
- contrastive and preservation losses

The sweep deliberately does not load either holdout benchmark. Candidate
selection is based only on development metrics:

```text
1. highest LOO routing accuracy
2. highest semantic margin
3. lowest geometry drift
```

Results are saved to:

```text
semantic_projection_sweep.csv
```

Each trained candidate is also saved under:

```text
model/projection_sweep/
```

A custom sweep can be run with:

```powershell
python semantic_projection_sweep.py --lambdas 0.25,0.5,1.0,2.0,4.0
```

The existing independent holdout sets should not be used to choose lambda.
After development-only selection, confirm the chosen configuration with a new
independent test set.


## Multi-seed preservation sweep

Branch `v0.1` now includes a development-only multi-seed stability test:

```text
semantic_projection_multiseed.py
```

Run:

```powershell
python semantic_projection_multiseed.py
```

Default experiment:

```text
lambda = 0.5, 1.0, 2.0, 5.0
seed   = 1, 2, 3, 4, 5
```

For each lambda/seed pair, the script trains a fresh projection head and
measures:

- leave-one-out routing accuracy
- semantic margin
- geometry drift
- best epoch
- contrastive and preservation losses

It then reports, for each lambda:

- mean LOO accuracy
- standard deviation of LOO accuracy
- mean semantic margin
- standard deviation of semantic margin
- mean geometry drift
- standard deviation of geometry drift

Candidate selection is development-only and stability-aware:

```text
1. highest mean LOO accuracy
2. lowest LOO accuracy standard deviation
3. prefer mean geometry drift <= 0.05
4. highest mean semantic margin
```

Outputs:

```text
semantic_projection_multiseed_detail.csv
semantic_projection_multiseed_summary.csv
```

The script intentionally does not load any holdout benchmark. The selected
configuration should be confirmed only with a fresh independent test set.


## v0.2 reproducible projection workflow

The base language model remains frozen. Only the semantic projection head is
trained again.

Recommended sequence:

```powershell
python semantic_projection_multiseed.py
python semantic_projection_train.py --preservation-lambda 1.0 --seed 42 --output model/semantic-projection-v0.2.pt
python semantic_projection_eval.py --projection model/semantic-projection-v0.2.pt
python semantic_infer.py --projection model/semantic-projection-v0.2.pt
```

The model artifacts are intentionally separated:

```text
model/
  tokenizer.json
  model-gpu-v0.4.pt
  semantic-projection-v0.2.pt
```

The operational semantic path is:

```text
Input text
   |
Tokenizer
   |
Frozen LLM_GPU v0.4
   |
64-D raw hybrid semantic vector
   |
LLM_SEM v0.2 SemanticProjectionHead
   |
64-D projected semantic vector
   |
Projected class centroids
   |
Class-specific semantic radius
   |
Known semantic class / Unknown
```

`semantic_infer.py` loads the frozen base checkpoint and the projection
checkpoint separately. It rebuilds projected class centroids and class radii
from the development benchmark, then classifies an input without modifying the
base LLM.

Single-input example:

```powershell
python semantic_infer.py --text "明日の東京の気温を知りたいです"
```

This separation makes it possible to continue semantic experiments without
retraining or overwriting the base LLM checkpoint.


## Unknown stress test

LLM_SEM v0.2 includes a multi-category out-of-domain stress test:

```text
semantic_unknown_stress.py
unknown_stress_benchmark.csv
```

The supplied benchmark contains unseen categories such as finance, music, law,
history, art, and sports. None of these categories are used as known routing
classes.

Run:

```powershell
python semantic_unknown_stress.py
```

The test uses the frozen base LLM and the trained semantic projection, then
measures each stress sample against the projected known-class centroids and
class-specific radii.

Key metric:

```text
distance_radius_ratio = distance_to_nearest_known_centroid / class_radius
```

Interpretation:

```text
ratio <= 1.0  -> accepted inside a known-class radius
ratio >  1.0  -> detected as Unknown
```

The report includes:

- overall Unknown Detection Rate
- per-category Unknown Detection Rate
- nearest known-class attraction counts
- mean/minimum distance-to-radius ratio
- per-sample distance, radius, similarity, and decision

Detailed results are written to:

```text
semantic_unknown_stress_results.csv
```


## Integrated Known + Unknown stress test

LLM_SEM v0.2 also provides a combined evaluation of independent Known samples
and out-of-domain Unknown samples:

```text
semantic_integrated_stress.py
```

Run:

```powershell
python semantic_integrated_stress.py
```

The test uses:

```text
development known classes -> my_benchmark.csv
independent Known samples  -> holdout_benchmark.csv
Unknown stress samples     -> unknown_stress_benchmark.csv
```

All inputs are passed through the frozen base LLM and the trained v0.2
projection head. The projected development benchmark defines known-class
centroids and class-specific semantic radii.

The report includes:

- Known base routing accuracy
- Known Recall after Unknown rejection
- False Unknown Rate
- Unknown Detection Rate
- False Known Rate
- Balanced Accuracy
- Known and Unknown distance/radius distributions
- maximum Known ratio and minimum Unknown ratio
- separation gap and overlap status
- per-category Known and Unknown results

The core separation diagnostic is:

```text
separation_gap =
    min(Unknown distance/radius)
    -
    max(Known distance/radius)
```

Interpretation:

```text
separation_gap > 0
    -> no observed overlap between Known and Unknown ratio distributions

separation_gap <= 0
    -> observed overlap; radius policy requires further study
```

Detailed results are saved to:

```text
semantic_integrated_stress_results.csv
```


## Radius scale sweep

LLM_SEM v0.2 includes a class-radius scale sweep that does not retrain either
the base LLM or the semantic projection head:

```text
semantic_radius_sweep.py
```

Run:

```powershell
python semantic_radius_sweep.py
```

Default sweep:

```text
scale = 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5
```

For every radius scale the script reports:

- Known Recall
- Known Accept Rate
- False Unknown Rate
- Unknown Detection Rate
- False Known Rate
- Balanced Accuracy
- maximum Known distance/radius ratio
- minimum Unknown distance/radius ratio
- separation gap

Candidate selection uses:

```text
1. highest Balanced Accuracy
2. highest Known Recall
3. highest Unknown Detection Rate
4. lowest False Unknown Rate
```

Results are saved to:

```text
semantic_radius_sweep.csv
```

A custom sweep can be run with:

```powershell
python semantic_radius_sweep.py --scales 0.9,1.0,1.05,1.1,1.15,1.2
```


## Projection architecture sweep

LLM_SEM v0.2 includes a sweep across projection hidden size and geometry
preservation strength:

```text
semantic_projection_arch_sweep.py
```

Run:

```powershell
python semantic_projection_arch_sweep.py
```

Default candidates:

```text
hidden_dim = 32, 64, 128, 256
lambda     = 0.5, 1.0, 2.0
```

The base LLM remains frozen for every candidate. Each projection head is
trained from scratch with the same seed and evaluated with radius_scale=1.0.

Reported metrics include:

- development leave-one-out accuracy
- geometry drift
- independent Known base accuracy
- Known Recall after radius rejection
- False Unknown Rate
- Unknown stress detection rate
- Balanced Accuracy

Candidate selection prioritizes:

```text
1. highest Balanced Accuracy
2. highest Known base accuracy
3. highest development LOO accuracy
4. highest Unknown Detection Rate
5. lowest geometry drift
```

Results are saved to:

```text
semantic_projection_arch_sweep.csv
```

A smaller custom sweep can be run with:

```powershell
python semantic_projection_arch_sweep.py --hidden-dims 64,128 --lambdas 0.5,1.0
```
