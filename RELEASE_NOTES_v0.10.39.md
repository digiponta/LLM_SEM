# LLM_SEM v0.10.39

## Pairwise Consolidation

v0.10.38 showed that repeated one-sided protected repair does not remove the persistent conflict.

The representative conflict stayed effectively fixed through all repair rounds:

```text
量子通信を説明して
  target: 0.259259 -> 0.285714

量子暗号とは
  protected drop: 0.078379
  allowed drop  : 0.05
```

The issue is therefore treated as interference between two nearby concepts rather than insufficient repair duration.

## New fallback

When:
- the new target has visible runtime progress;
- protected repair has exhausted all rounds;
- one or more protected rows still fail;

v0.10.39 builds a pairwise dataset.

```text
must_train:
  new target
  conflicting protected row(s)

protected:
  all other known/accepted rows
```

For the current case this means conceptually:

```text
量子通信 ...      must_train
量子暗号とは      must_train
文学 ...          protected
```

The pair is optimized together from the committed source checkpoint rather than continuing a one-sided repair chain.

## Pairwise settings

```text
epochs                    : 180
learning rate             : 0.50x
LM head learning rate     : 0.50x
protected replay weight   : 4.0
new knowledge weight      : 4.0
concept-balanced training : ON
```

## Acceptance

Pairwise consolidation is accepted only when:
- the new target keeps at least +0.02 visible canonical gain, or reaches canonical >= 0.70;
- every existing protected anchor satisfies the original retention conditions.

If either side fails, the original committed source remains active.

## Default candidate

```text
model/model-sem-sleep-v0139.pt
```

## Regression

```powershell
python run_pairwise_consolidation_regression_v01039.py
```
