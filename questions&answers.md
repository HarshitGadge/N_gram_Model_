# Section 4: Analysis and Discussion - Questions and Answers

---

## 4.1 Pre-processing and Vocabulary Decisions

### Question:
**Explain your tokenization, sentence boundary and other preprocessing strategies.**

### Answer:

**Tokenization:**
- Penn Treebank comes pre-tokenized with words separated by spaces
- Punctuation treated as separate tokens
- Case preserved for proper nouns

**Sentence Boundaries:**
- `<s>` token marks sentence start
- `</s>` token marks sentence end
- One sentence per line in dataset

**Other Preprocessing:**
- `<unk>` replaces rare/out-of-vocabulary words
- `N` replaces numeric values to reduce vocabulary size
- No additional preprocessing needed (PTB is already preprocessed)

**Vocabulary:**
- Built from all unique tokens in training corpus
- Estimated size: ~10,000 tokens
- Includes words, special tokens, punctuation, and placeholders

---

## 4.2 Impact of N-gram Order

### Questions:
1. **Compare the perplexity results for all models.**
2. **Discuss the trend you observe and explain this phenomenon in terms of the Markov Assumption and Data Sparsity.**

### Answer:

**Perplexity Comparison:**

| Model Type | 1-gram | 2-gram | 3-gram | 4-gram |
|------------|--------|---------|---------|---------|
| **Pure MLE** | 652.16 | INF | INF | INF |
| **Add-1 Smoothing** | 652.74 | 748.12 | 3,278.52 | 6,129.72 |
| **Linear Interpolation** | - | - | 188.73 | - |
| **Stupid Backoff** | - | - | - | 185.70 |

**Trends Observed:**
- Pure MLE: Higher-order models show INF perplexity
- Add-1: Performance degrades with higher n-gram orders
- Advanced models: Much better performance (~185-188)

**Markov Assumption:**
- In theory: Longer context (higher n) = better predictions
- Assumes P(word | all previous words) ≈ P(word | last n-1 words)
- Should improve with more context

**Data Sparsity Problem:**
- Possible n-grams grow exponentially: V, V², V³, V⁴
- With V ≈ 10,000: 10¹² possible 4-grams but only 887K training words
- Higher-order n-grams more likely to be unseen in training
- Test set contains novel combinations → zero probabilities

**Why MLE Shows INF:**
- Unseen n-gram → count = 0 → P = 0
- log(0) = -∞ in perplexity calculation
- Result: Infinite perplexity

**Why Add-1 Gets Worse:**
- Adds 1 to all possible n-grams (including billions of unseen ones)
- Massive probability redistribution away from observed sequences
- Over-smoothing: too much probability to impossible sequences
- Unrealistic uniform prior assumption

**Conclusion:** Without proper smoothing, data sparsity overwhelms the benefits of additional context.

---

## 4.3 Comparison of Smoothing/Backoff Strategies

### Questions:
1. **Why are the perplexity scores for the unsmoothed models so high/infinite, and how did smoothing correct this?**
2. **Compare the performance of the backoff/interpolation strategies. Which one performed the best and why?**

### Answer:

**Why Unsmoothed Models Have INF Perplexity:**

**The Zero Probability Problem:**
- Unseen n-gram in test set → P = 0
- log(0) = -∞ → infinite perplexity
- Inevitable with limited training data

**How Smoothing Fixes This:**

**Linear Interpolation:**
```
P(w | context) = λ₁·P_1gram + λ₂·P_2gram + λ₃·P_3gram
```
- Blends multiple n-gram orders
- If 3-gram unseen, 2-gram and 1-gram provide backup
- Never assigns zero (unigram always > 0)
- Optimal λs: (0.33, 0.33, 0.34) from dev set

**Stupid Backoff:**
```
If 4-gram seen: use it
Else: α × (try 3-gram)
Else: α × (try 2-gram)
Else: α × (use 1-gram)
```
- Recursive backoff chain
- Always reaches unigram fallback
- α = 0.4 penalty at each backoff level

**Performance Comparison:**

| Model | Perplexity | Order |
|-------|------------|-------|
| Linear Interpolation | 188.73 | 3-gram |
| **Stupid Backoff** | **185.70** ✓ | 4-gram |

**Why Stupid Backoff Won:**

1. **Higher-order base:** Uses 4-grams vs 3-grams → more specific predictions when context available
2. **Selective smoothing:** Only backs off when necessary, preserves sharp probabilities for frequent sequences
3. **Less aggressive:** Linear Interpolation always blends all orders, even when longer context is reliable
4. **Computational efficiency:** No normalization required, faster computation

**When Linear Interpolation Might Be Better:**
- Very sparse training data
- Need calibrated probabilities (not just scores)
- Theoretical guarantees required

---

## 4.4 Qualitative Analysis (Generated Text)

### Questions:
1. **Generate at least 5 distinct sentences and include them in your report.**
2. **How "fluent" or "human-like" does the generated text appear?**
3. **Provide a brief explanation of how the model manages to generate these sequences.**

### Answer:

**Generated Sentences (Stupid Backoff):**

1. "Hahn carry it off in the executive if there month ends the two executives considered."
2. "To oust unk on the unk leader arthur unk such a high of n transactions."
3. "Miller says in his alleged unk and several n n schedule could be more brand."
4. "K."
5. "S ad unit to lead of some concern to the fact at the appropriate time."

**Fluency Assessment: LOW TO MODERATE (3/10)**

**Positive Observations:**
- Local coherence in short phrases: "in the executive", "at the appropriate time"
- Some syntactic awareness: determiners before nouns, basic SVO structure
- Domain vocabulary: business terms ("executive", "transactions") reflect PTB corpus

**Limitations:**
- Frequent `<unk>` tokens break readability
- Number placeholders ("n n") appear literally
- Semantic drift: no coherent meaning across clauses
- Grammatical errors: "Hahn carry" (agreement error)
- Fragments: sentence 4 is just "K."
- No global coherence or consistent topic

**How the Model Generates Text:**

**Process:**
1. Start with `<s>` token
2. For each position:
   - Build context window (previous 3 words for 4-gram)
   - Look up P(word | context)
   - If unseen → backoff: 4-gram → 3-gram → 2-gram → 1-gram (multiply by α=0.4 each time)
   - Sample next word from probability distribution
   - Update context window
3. Stop when `</s>` sampled or max length reached

**Why It Looks This Way:**
- **Limited context:** Only sees 3 words of history, cannot maintain long-range coherence
- **No semantics:** Purely statistical, no understanding of meaning or causality
- **Training data reflection:** Business vocabulary from Wall Street Journal articles
- **Markov property:** Can't remember sentence topic from 10 words ago

**Comparison to Modern Models:**
- N-gram: 3-4 words context, statistical patterns only, local syntax
- Transformers: 100+ words context, semantic understanding, global coherence

**Conclusion:** Model generates by following learned statistical patterns with probabilistic sampling and backoff, but lacks semantic understanding, resulting in locally plausible but globally incoherent text.

---
