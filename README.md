# N-Gram Language Models on the Penn Treebank

N-gram language models written from scratch in Python (no NLP libraries): unsmoothed maximum likelihood, Add-1 (Laplace) smoothing,
linear interpolation with weights tuned on the dev set, and Stupid Backoff. They're trained on the Penn Treebank and compared by
test-set perplexity, and the best model generates sample sentences.

## Results (Penn Treebank test set, perplexity, lower is better)

| Model | 1-gram | 2-gram | 3-gram | 4-gram |
|---|---|---|---|---|
| Maximum likelihood (no smoothing) | 652.16 | ∞ | ∞ | ∞ |
| Add-1 smoothing | 652.74 | 748.12 | 3,278.52 | 6,129.72 |

| Model | Perplexity |
|---|---|
| Linear interpolation, trigram (λ = 0.33 / 0.33 / 0.34, chosen on the dev set) | 188.73 |
| **Stupid Backoff, trigram (α = 0.4)** | **185.70** |

These numbers were reproduced by re-running `code.py`.

**What the results show**

- **Unsmoothed higher-order models fail completely.** Any test n-gram never seen in training gets probability 0, so the log-likelihood
  is −∞ and perplexity is infinite.
- **Add-1 smoothing fixes the zeros but gets worse as n grows.** It spreads probability mass over every possible n-gram
  (10,000⁴ = 10¹⁶ possible 4-grams with a 10,000-word vocabulary, against about 887,000 training tokens), so real continuations get too little.
- **Mixing orders works.** Interpolation and backoff use the trigram when it has evidence and fall back to shorter contexts when it doesn't,
  which cuts perplexity by about 3.5× compared with the best Add-1 model.

**Caveat on Stupid Backoff.** Stupid Backoff (Brants et al., 2007) returns *scores*, not a normalized probability distribution, so its
"perplexity" isn't strictly comparable to that of a proper probability model like interpolation. The two are close (185.70 vs 188.73);
the fair conclusion is that both clearly beat Add-1, not that backoff is better. A normalized alternative such as Katz backoff or
Kneser-Ney smoothing would settle it.

`report.md` has the full write-up (preprocessing, why each model behaves as it does, generated sentences and a fluency assessment),
and `questions&answers.md` answers the assignment's analysis questions.

> Note: parts of `report.md` describe the Stupid Backoff model as a 4-gram. The implementation in `code.py` backs off
> trigram → bigram → unigram (`StupidBackoffModel.n = 3`), and the numbers above come from that code.

## How it works

| Component | In `code.py` |
|---|---|
| Sentence padding with `<s>` / `</s>`, n-gram and context counting | `NGramsProcessor`, `NGramsModel` |
| Unsmoothed MLE and Add-1 models for n = 1–4 | `MLEModel`, `AddOneSmoothingModel` |
| Linear interpolation of unigram, bigram and trigram, with a grid search over λ on `ptb.valid.txt` | `LinearInterpolationModel`, `LanguageModelEvaluator.tune_interpolation_weights` |
| Stupid Backoff with α = 0.4 | `StupidBackoffModel` |
| Perplexity on `ptb.test.txt` | `LanguageModelEvaluator` |
| Sampling sentences from the best model | `TextGenerator` |

The Penn Treebank files (`ptb.train.txt`, `ptb.valid.txt`, `ptb.test.txt`) are the standard pre-tokenized version with a 10,000-word vocabulary
and rare words replaced by `<unk>`.

## Run it

```bash
git clone https://github.com/HarshitGadge/N_gram_Model_.git
cd N_gram_Model_
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install numpy
python code.py
```

Requires Python 3.8+. The script prints every model's perplexity, the dev-set λ search, and generated sentences.

## Files

```
code.py                 all models, evaluation and text generation
ptb.train.txt / ptb.valid.txt / ptb.test.txt    Penn Treebank splits
report.md               full analysis report
questions&answers.md    answers to the assignment's analysis questions
```

Possible extensions: Kneser-Ney smoothing, a λ search that isn't limited to five hand-picked combinations, and comparing against a small
neural language model (for example an LSTM) on the same splits.
