# LLM_SEM v0.10.9

## Sleep Answer Memorization

v0.10.8 proved that the sleep checkpoint was promoted but the internally generated answer for taught knowledge was still gibberish.

v0.10.9 strengthens the sleep QA stage.

### Changes

- Learned Answer Memory rows are always placed in the training split.
- Relation Memory canonical rows are always placed in the training split.
- Base QA rows keep normal train/holdout behavior.
- Learned rows use sleep weight 6.
- Relation canonical rows use sleep weight 8.
- Sleep QA defaults increase to 240 epochs.
- The last two Transformer blocks plus final norm and LM head are trainable during sleep QA.
- Sleep QA uses stronger learning rates while preserving semantic geometry.
- Mandatory external knowledge is generated after training and checked with string similarity.
- Promotion requires mandatory generation similarity >= 0.35.
- The next sleep checkpoint is `model/model-sem-sleep-v0109.pt`, avoiding overwrite of the current active `v0106` checkpoint.

### Verification

```powershell
python run_sleep_memorization_regression_v0109.py
python chat.py
```

Then:

```text
/sleep
```

After a successful promotion and restart:

```text
/internal 文学とは
```

The final sleep log prints generation probes such as:

```text
Generation probe details
[0.xxx] 文学とは
  expected : ...
  generated: ...
```

A sleep candidate with gibberish mandatory generations is not promoted.
