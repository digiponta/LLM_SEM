# LLM_SEM v0.10.44

## Batched Sleep Optimization

v0.10.40 fixed a correctness bug by making teacher forcing use the same rolling
context window as runtime generation. The first implementation evaluated each
answer token with a separate Transformer forward pass.

That was correct but very slow.

For one row, one sleep attempt may use 120-240 epochs, and the one-by-one
scheduler can try five configurations:

```text
120 + 160 + 200 + 240 + 240 = 960 epochs / row
```

With multiple remaining rows, protected repair, pairwise fallback, semantic
preservation and runtime validation, /sleep can therefore take a long time.

## Optimization

v0.10.44 keeps exactly the same runtime-visible prefix for every answer token,
but groups positions with the same prefix length into one GPU batch.

Old:

```text
answer token 1 -> forward
answer token 2 -> forward
answer token 3 -> forward
...
```

New:

```text
equal-length context windows
        -> one batched forward
```

No padding is used, so each sample sees exactly the same tokens as before.

## Correctness

The regression compares the old per-token reference implementation against the
new batched implementation and checks both:

- loss equivalence
- gradient equivalence

Run:

```powershell
python run_batched_context_loss_regression_v01044.py
```

## Expected benefit

The speed-up depends on prompt and answer length. Once the rolling prefix
reaches context_length, many answer positions share the same context length and
can be evaluated together, substantially reducing Transformer forward calls.

The sleep acceptance policy, retry schedule, retention checks, PARTIAL commit,
and model architecture are unchanged.

## Default candidate

```text
model/model-sem-sleep-v0144.pt
```
