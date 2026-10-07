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


## v0.15.7.5 — Semantic Memory Internalization completed

The v0.15.7.x experiments demonstrate a complete path from external Semantic
Memory to a promoted model checkpoint whose decoder behavior reflects the new
knowledge while preserving the source model's trajectory within the defined
validation thresholds.

The experimental progression was:

```text
Semantic Memory
    |
latent internalization
    |  target NLL improves and weights change
    |
decoder preference shift
    |
decoder boundary crossing
    |  canonical continuation > confuser continuation
    |
trajectory preservation
    |  source behavior remains within preservation thresholds
    |
qualified candidate
    |
promotion gate
    |
model/model-sem-internalized-v01575.pt
```

### Final promoted checkpoint

```text
model/model-sem-internalized-v01575.pt
```

Promotion provenance is written to:

```text
results/promotion_manifest_v01575.json
```

### Boundary-crossing result

Prompt-aware decoder validation against the protected source checkpoint
`model/model-sem-diagnostic-v0154.pt` produced:

```text
Mean margin BEFORE : -1.807448
Mean margin AFTER  : +0.436255
Mean margin gain   : +2.243703
Boundary crossed   : 4/4
RESULT             : PROMPT_AWARE_PASS
```

This means the canonical quantum-sensor continuation moved from below the
confuser continuation to above it for every prompt-aware probe.

### Preservation result

The selected decoder candidate used:

```text
final_norm learning rate : 5e-4
lm_head learning rate    : 2e-4
```

and passed trajectory-preservation validation:

```text
Mean prompt JS           : 0.042576
Prompt top1 retention    : 83.3%
Mean trajectory JS       : 0.042938
Mean trajectory top1     : 76.6%
Mean source NLL delta    : +0.117602
Boundary check           : PASS
Trajectory preservation : PASS
QUALIFIED                : YES
```

A stronger candidate using `1e-3 / 5e-4` crossed the decoder boundary but
failed preservation, confirming that successful internalization requires a
balance between target acquisition and source-behavior retention.

### Final validation and promotion

The promoted checkpoint was re-evaluated after promotion metadata was added and
retained the same boundary and trajectory metrics. The final promotion gate
reported:

```text
Boundary             : PROMPT_AWARE_PASS
Trajectory JS        : 0.042938
Trajectory top1      : 76.6%
Source NLL delta     : +0.117602
RESULT               : PROMOTED
```

The final verification commands are:

```powershell
python .\run_decoder_boundary_diagnostic_v01561.py `
  --before .\model\model-sem-diagnostic-v0154.pt `
  --after .\model\model-sem-internalized-v01575.pt `
  --tokenizer .\model\tokenizer.json

python .\run_trajectory_preservation_v01574.py `
  --source .\model\model-sem-diagnostic-v0154.pt `
  --candidate .\model\model-sem-internalized-v01575.pt `
  --tokenizer .\model\tokenizer.json
```

### Experimental conclusion

v0.15.7.5 establishes the following experimental result:

> Semantic knowledge held in external Semantic Memory can be internalized into
> the model weights, moved across the decoder decision boundary, and promoted
> as an internalized checkpoint while retaining the protected source trajectory
> within the defined validation thresholds.

This is an experimental result for the current model, dataset, prompts, and
validation policy. It should not be interpreted as a general proof that all
semantic memories can be safely internalized under arbitrary conditions.

See `RELEASE_NOTES_v0.15.7.5.md` for the experiment history and validation
summary.


## v0.18.1: NDC-based domain classification

LLM_SEM now uses the Nippon Decimal Classification (NDC) as the common domain
taxonomy for Semantic Memory metadata.

- The historical labels `computer / science / animal / weather / food / transport`
  map to `007 / 400 / 480 / 451 / 596 / 680`.
- New Semantic Memory rows persist `ndc_code`, `ndc_main`, `ndc_name`,
  `classification_state`, and `ndc_source`.
- Existing prompt/answer-only memory files remain compatible: NDC metadata is
  attached at load time without rewriting the source file.
- `UNKNOWN` is deliberately separate from NDC `000`; code `000` remains the
  valid NDC class for General Works.
- NDC is a domain axis only. Semantic vectors, proposition structure,
  internalized gating, protected knowledge, and `/sleep` continue to operate
  independently.

Representative mapping:

| Domain | NDC |
|---|---:|
| Information science / AI / LLM / CPU / GPU | 007 |
| Mathematics | 410 |
| Physics / quantum mechanics | 420 |
| Astronomy / space science | 440 |
| Meteorology | 451 |
| Zoology | 480 |
| Food / cooking | 596 |
| Transportation | 680 |
| Language | 800-series |
| Literature | 900-series |

Regression:

```powershell
python .\run_ndc_classification_v0181.py
```


## v0.18.2: Semantic Vector + NDC centroid experiment

This branch adds a training-free semantic NDC router on top of the v0.18.1
keyword baseline.

Pipeline:

```text
text
  -> LLM_SEM encode_semantic()
  -> L2-normalized semantic vector
  -> NDC 0-9 centroids
  -> cosine top-1 / top-2 margin
  -> similarity threshold
  -> ACCEPT or UNKNOWN
```

The first experiment deliberately routes only the ten NDC main classes.  It
does not yet classify three-digit NDC codes.  This keeps the experiment focused
on whether the current semantic space separates broad knowledge domains.

Files:

- `ndc_semantic_router_v0182.py`
  - curated centroid seeds for NDC 0-9
  - semantic-vector encoding
  - cosine centroid routing
  - ACCEPT / UNKNOWN state
- `run_ndc_semantic_centroid_v0182.py`
  - 30 held-out known queries
  - 12 ambiguous/nonsense unknown probes
  - threshold sweep
  - raw routing accuracy
  - known acceptance
  - unknown rejection
  - balanced threshold score

Run:

```powershell
python .\run_ndc_semantic_centroid_v0182.py
```

Optional pooling comparison:

```powershell
python .\run_ndc_semantic_centroid_v0182.py --pooling mean
python .\run_ndc_semantic_centroid_v0182.py --pooling hybrid
python .\run_ndc_semantic_centroid_v0182.py --pooling attention
```

UNKNOWN remains a classifier confidence state and is not NDC 000.


## v0.18.3: Multi-Prototype NDC Semantic Router

v0.18.2 showed that one centroid per NDC main class over-compressed the current
semantic space. Mean pooling was best, but raw accuracy was 53.33%, while hybrid
and attention were lower.

v0.18.3 therefore keeps multiple semantic prototypes per NDC main class.

Pipeline:

```text
text
  -> mean-pooled semantic vector
  -> cosine similarity to all NDC prototypes
  -> per-class top-k aggregation
  -> top-1 class / top-2 class margin
  -> similarity + margin calibration
  -> ACCEPT / UNKNOWN
```

The experiment searches:

- top_k = 1, 2, 3
- nearest-prototype weight = 0.40, 0.60, 0.80, 1.00
- similarity threshold = 0.60 .. 0.95
- margin threshold = 0.00 .. 0.08

Run:

```powershell
python .\run_ndc_semantic_prototypes_v0183.py
```

The model remains frozen. This branch tests whether local prototype structure
can recover NDC separability before any learned projection is introduced.


## v0.18.4: NDC-Specific Semantic Projection Head

v0.18.3 established the multi-prototype baseline:

- raw accuracy: 70.00%
- known acceptance: 50.00%
- unknown rejection: 100.00%

v0.18.4 adds a learned NDC-specific projection space while keeping the base
LLM completely frozen.

Architecture:

```text
Frozen Base LLM
    |
    v
Base Semantic Vector (d_model=256)
    |
    v
Linear 256 -> 128
    |
   GELU
    |
 LayerNorm
    |
    v
Linear 128 -> 64
    |
 L2 normalize
    |
    v
Projected NDC Space
    |
    +--> Multi-Prototype Routing
    |
    +--> Similarity + Margin Unknown Gate
```

Only the projection module and its temporary cosine-classifier prototypes are
trained. The base LLM parameters are explicitly set to `requires_grad=False`.

Training objective:

```text
loss =
    cosine prototype cross entropy
  + 0.25 * same-class compactness
  + 0.10 * inter-class separation
```

The experiment runs five initialization seeds by default and automatically
selects the best projected router after re-calibrating:

- prototype top-k
- nearest-prototype weight
- similarity threshold
- class-margin threshold

Run:

```powershell
git switch v0.18.4
python .\run_ndc_projection_v0184.py
```

Output checkpoint:

```text
model/ndc-projection-v0184.pt
```

Target criteria:

| Metric | v0.18.3 baseline | v0.18.4 target |
|---|---:|---:|
| Raw accuracy | 70.00% | >= 80% |
| Known accept | 50.00% | >= 65% |
| Unknown reject | 100.00% | >= 90% |

Files:

- `ndc_projection_v0184.py` — projection head and checkpoint I/O
- `run_ndc_projection_v0184.py` — training, multi-seed evaluation, threshold calibration

This stage is intentionally limited to NDC main classes 0-9. Three-digit NDC
routing will be attempted only after the projected main-class space is stable.


## v0.18.6: Leakage-Free Residual NDC Projection

v0.18.5 preserved the v0.18.3 raw routing baseline (70.00%) but final known
acceptance collapsed to 3.33%. The cause was calibration leakage: runtime
prototypes were built from all six NDC seed texts per class, while the final two
texts per class were also used as DEV samples. DEV therefore contained its own
prototype texts and reported an artificial 100% routing/acceptance score.

v0.18.6 fixes the protocol:

```text
40 TRAIN samples
   |
   +--> projection training
   |
   +--> runtime NDC prototypes (TRAIN ONLY)

20 DEV samples
   |
   +--> seed/model selection
   +--> top-k / weight / similarity / margin calibration

30 FINAL TEST samples
   |
   +--> untouched until the final report
```

Unknown probes are also split into DEV and FINAL subsets.

The residual projection remains near identity:

```text
64 -> 32 -> 64 residual
z = normalize(x + alpha * delta(x))
alpha = 0.25
```

Run:

```powershell
git switch v0.18.6
python .\run_ndc_projection_v0186.py
```

Output checkpoint:

```text
model/ndc-projection-v0186.pt
```

The first goal is not to exceed the v0.18.3 raw baseline immediately, but to
verify that a leakage-free calibration can preserve approximately 70% raw
accuracy while recovering useful known acceptance without sacrificing unknown
rejection.


## v0.18.7: Confusion-Aware Prototype Augmentation

v0.18.6 produced a trustworthy leakage-free result:

- raw accuracy: 66.67%
- known acceptance: 40.00%
- unknown rejection: 100.00%

This confirmed that the residual projection preserves the original semantic
geometry but does not improve routing beyond the v0.18.3 multi-prototype
baseline. The dominant remaining errors are concentrated in NDC 2/3/4 and
8/9, with selected NDC 7 confusions.

v0.18.7 therefore returns to the strongest idea from v0.18.3: local semantic
prototypes. The experiment adds targeted representative phrases for the
confused regions while keeping the base model frozen.

To keep evaluation valid, v0.18.7 does NOT reuse the previous final test as a
fresh final benchmark. Instead it introduces a NEW FINAL-V2 holdout and compares:

```text
A) TRAIN-ONLY BASELINE
   40 original prototypes

B) CONFUSION-AWARE AUGMENTED
   40 original train-only prototypes
   + targeted extra prototypes
```

Calibration uses only the original DEV subset and DEV unknown probes.

Run:

```powershell
git switch v0.18.7
python .\run_ndc_confusion_prototypes_v0187.py
```

The script reports FINAL-V2 raw accuracy, known acceptance, unknown rejection,
balanced score, per-class results, and the delta between augmented and baseline
routing.


## v0.18.8: Dual-Router Consensus Gate

v0.18.7 improved raw routing from 63.33% to 66.67% on FINAL-V2, but global
threshold calibration reduced known acceptance from 33.33% to 13.33% while
raising unknown rejection to 100%.

v0.18.8 therefore separates routing from unknown rejection. It runs both:

- TRAIN-only baseline prototype router
- confusion-aware augmented prototype router

and accepts a classification only when both routers predict the same NDC main
class and pass a relaxed similarity/margin gate.

```text
Semantic Vector
   |--------------------|
   v                    v
Baseline Router    Augmented Router
   |                    |
   +------ agreement ---+
             |
      same NDC class?
         /       \
       yes       no
       |          |
 relaxed gate   UNKNOWN
       |
     ACCEPT
```

Because v0.18.8 was designed after observing FINAL-V2, it introduces a NEW
FINAL-V3 holdout and does not reuse FINAL-V2 as a fresh benchmark.

Run:

```powershell
git switch v0.18.8
python .\run_ndc_consensus_v0188.py
```

The experiment compares baseline, augmented, and consensus routing on the same
FINAL-V3 set and reports the consensus delta in known acceptance, unknown
rejection, and balanced score.


## v0.18.9: Separated Classification / Unknown Detection

v0.18.8 showed that the augmented router can reach 90.00% raw classification
accuracy, while a single global threshold suppresses known acceptance.

v0.18.9 therefore separates the two tasks:

```text
Semantic Vector
   |
   +--> Augmented NDC Router ----> NDC 0-9 label
   |
   +--> Evidence Gate -----------> ACCEPT / UNKNOWN
```

The NDC label always comes from the confusion-aware augmented router. The
independent unknown gate combines:

- baseline/augmented router agreement
- augmented nearest similarity
- baseline nearest similarity
- augmented class margin
- baseline class margin

The gate uses a calibrated evidence score rather than one global cosine
threshold.

To preserve evaluation independence, this branch introduces a new FINAL-V4
known/unknown holdout.

Run:

```powershell
git switch v0.18.9
python .\run_ndc_separated_gate_v0189.py
```

Target criteria:

- raw augmented accuracy >= 80%
- known acceptance >= 50%
- unknown rejection >= 80%


## v0.18.10: Contrastive Unknown Gate

v0.18.9 demonstrated that the NDC classifier itself is now strong:

- FINAL-V4 raw augmented accuracy: 93.33%
- known acceptance: 86.67%
- unknown rejection: 16.67%

The remaining problem is therefore unknown detection, not NDC classification.

v0.18.10 freezes the classifier design and replaces the gate with a direct
known-vs-unknown semantic comparison.

```text
Semantic Vector
   |
   +--> Augmented NDC prototypes
   |       -> nearest known-domain similarity
   |
   +--> UNKNOWN prototypes
           -> nearest unknown-pattern similarity

contrast = known_similarity - unknown_similarity

gate_score =
    contrast_weight * contrast
  + margin_weight * NDC_margin
  + known_similarity_weight * known_similarity
```

The NDC label still comes from the augmented router. The contrastive gate only
decides ACCEPT vs UNKNOWN.

Evaluation uses a new FINAL-V5 holdout to avoid reusing FINAL-V4 after observing
its errors.

Run:

```powershell
git switch v0.18.10
python .\run_ndc_unknown_gate_v01810.py
```

Target criteria:

- raw augmented accuracy >= 80%
- known acceptance >= 60%
- unknown rejection >= 80%


## v0.18.11: Stable NDC Runtime Integration

v0.18.10 established the current stable NDC architecture:

- raw augmented NDC accuracy: 93.33%
- known acceptance: 93.33%
- unknown rejection: 100.00%
- balanced score: 96.67%

The classifier and unknown detector are intentionally separated:

```text
Frozen semantic encoder
        |
        +--> Augmented NDC prototype classifier --> NDC main class
        |
        +--> Contrastive unknown prototypes
                    |
                    v
          known_similarity - unknown_similarity
                    |
                    v
               ACCEPT / UNKNOWN
```

v0.18.11 packages this architecture into `StableNDCRouter` and integrates it
into `chat.py` as an independent runtime command.

Files:

- `ndc_runtime_v01811.py` — reusable stable runtime router
- `verify_ndc_runtime_v01811.py` — known/unknown regression
- `chat.py` — adds `/ndc <text>`

The NDC runtime uses its own frozen checkpoint by default:

```text
model/model-sem-internalized-v01575.pt
```

This keeps NDC classification stable even when the live chat model changes
through `/sleep`, `/repair`, or reload operations.

Run regression:

```powershell
python .\verify_ndc_runtime_v01811.py
```

Run chat:

```powershell
python .\chat.py
```

Examples:

```text
/ndc 化学物質の反応を観察する
/ndc 近代日本の歴史を学ぶ
/ndc それについてお願いします
```

UNKNOWN remains a separate classification state and is never mapped to NDC 000.
