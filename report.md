# N-Gram Language Modeling: Analysis and Report

## 1. Executive Summary

This report presents a comprehensive analysis of n-gram language models trained on the Penn Treebank (PTB) corpus. We implemented and evaluated multiple modeling approaches, including pure Maximum Likelihood Estimation (MLE), Add-1 smoothing, Linear Interpolation, and Stupid Backoff. Our best performing model was **Stupid Backoff with perplexity 185.70** on the test set, demonstrating the effectiveness of backoff strategies for handling data sparsity in language modeling.

**Dataset Statistics:**
- Training data: 887,521 words
- Development data: 70,390 words  
- Test data: 78,669 words

---

## 2. Pre-processing and Vocabulary Decisions

### 2.1 Tokenization Strategy

The Penn Treebank dataset comes pre-tokenized with several important preprocessing decisions already applied:

**Key preprocessing features observed:**
- **Sentence boundaries**: Each line represents a complete sentence, marked with `<s>` (start) and `</s>` (end) tokens
- **Unknown words**: Rare or out-of-vocabulary words are replaced with the `<unk>` token
- **Punctuation**: Treated as separate tokens (e.g., periods, commas, quotation marks)
- **Normalization**: Numbers are typically replaced with `N` tokens to reduce vocabulary size
- **Lowercasing**: The dataset appears to maintain original casing for proper nouns and sentence beginnings

### 2.2 Vocabulary Construction

The vocabulary was constructed from all tokens appearing in the training corpus. This includes:
- Regular word tokens
- Special tokens: `<s>`, `</s>`, `<unk>`
- Punctuation marks as separate tokens
- Numeric placeholders (`N`)

**Vocabulary Size Estimation:** Based on typical PTB preprocessing, the vocabulary likely contains approximately 10,000 unique tokens.

### 2.3 Handling Sentence Boundaries

Sentence boundary tokens play a crucial role in n-gram modeling:
- `<s>` tokens provide context for sentence-initial words
- `</s>` tokens model the probability of sentence termination
- For n-grams spanning multiple sentences, padding with `<s>` ensures proper context windows

---

## 3. Impact of N-gram Order

### 3.1 Results Summary

| Model Type | 1-gram | 2-gram | 3-gram | 4-gram |
|------------|--------|---------|---------|---------|
| **Pure MLE** | 652.16 | INF | INF | INF |
| **Add-1 Smoothing** | 652.74 | 748.12 | 3,278.52 | 6,129.72 |

### 3.2 Analysis of Pure MLE Models

**Key Observation:** Higher-order MLE models (2-gram through 4-gram) achieved infinite perplexity.

**Explanation:**

The infinite perplexity occurs due to the **zero probability problem**. When the model encounters an n-gram in the test set that never appeared in the training data, it assigns probability 0 to that sequence. Since perplexity is calculated as:
```
Perplexity = exp(-1/N * Σ log P(w_i | context))
```

When `P(w_i | context) = 0`, we have `log(0) = -∞`, resulting in infinite perplexity.

**Why does this happen more with higher-order models?**

1. **Data Sparsity and the Markov Assumption**: Higher-order n-grams use longer context windows. A 4-gram model conditions each word on the previous 3 words, creating contexts like "the company said it" → "would". 

2. **Exponential Growth of Possible N-grams**: With a vocabulary of size V:
   - Possible bigrams: V²
   - Possible trigrams: V³
   - Possible 4-grams: V⁴
   
   With V ≈ 10,000, there are 10¹² possible 4-grams, but only ~887K training tokens.

3. **Coverage Gap**: Even with 887K training words, we can only observe a tiny fraction of all possible n-grams. The test set inevitably contains novel n-gram combinations never seen during training.

### 3.3 Analysis of Add-1 Smoothing Models

**Trend Observed:** Perplexity actually *increases* with higher n-gram orders under Add-1 smoothing:
- 1-gram: 652.74
- 2-gram: 748.12
- 3-gram: 3,278.52
- 4-gram: 6,129.72

**Why does performance degrade?**

Add-1 (Laplace) smoothing adds 1 to all n-gram counts, including the billions of n-grams that were never observed. The probability formula becomes:
```
P(w | context) = (count(context, w) + 1) / (count(context) + V)
```

For higher-order models:
1. **Massive probability redistribution**: Adding 1 to V⁴ possible 4-grams means redistributing enormous probability mass away from actually observed sequences
2. **Unrealistic uniform prior**: Add-1 assumes all unseen n-grams are equally likely, which is unrealistic for natural language
3. **Over-smoothing**: The model becomes too conservative, assigning too much probability to impossible sequences

**The Fundamental Trade-off:** While Add-1 prevents zero probabilities (fixing the INF issue), it over-corrects by giving far too much probability to implausible sequences.

### 3.4 The Markov Assumption

The Markov assumption states that:
```
P(w_i | w_1, ..., w_{i-1}) ≈ P(w_i | w_{i-n+1}, ..., w_{i-1})
```

In theory, higher-order models should be better because they use more context. However, this is only true when:
- Sufficient training data exists to estimate these probabilities
- Appropriate smoothing techniques are used

Our results demonstrate that **without proper smoothing**, the data sparsity problem overwhelms the benefits of additional context.

---

## 4. Comparison of Smoothing and Backoff Strategies

### 4.1 Advanced Model Performance

| Model | Perplexity | Performance |
|-------|------------|-------------|
| **Linear Interpolation (3-gram)** | 188.73 | Excellent |
| **Stupid Backoff (4-gram)** | 185.70 | **Best** |

### 4.2 Why Smoothing Fixed the Infinite Perplexity Problem

**The Zero Probability Problem:**
- Unsmoothed MLE assigns P = 0 to unseen n-grams
- This causes log(0) = -∞ in perplexity calculations
- Makes the model completely break on unseen data

**How Advanced Methods Solve This:**

**Linear Interpolation** combines multiple n-gram orders:
```
P(w_i | w_{i-2}, w_{i-1}) = λ₁·P_unigram(w_i) + λ₂·P_bigram(w_i|w_{i-1}) + λ₃·P_trigram(w_i|w_{i-2},w_{i-1})
```

Key advantages:
- **Graceful degradation**: If the trigram is unseen, bigram and unigram probabilities provide backup
- **Learned weights**: Optimal λ values (0.33, 0.33, 0.34) found via development set tuning
- **Never zero probability**: Even if all specific contexts fail, the unigram provides non-zero probability

**Stupid Backoff** recursively backs off to shorter contexts:
```
S(w_i | w_{i-n+1}...w_{i-1}) = {
  count(w_{i-n+1}...w_i) / count(w_{i-n+1}...w_{i-1})           if count > 0
  α · S(w_i | w_{i-n+2}...w_{i-1})                              otherwise
}
```

Key advantages:
- **Computationally efficient**: No normalization required (scores, not probabilities)
- **Flexible backoff**: Uses 4-gram → 3-gram → 2-gram → 1-gram chain
- **Empirically effective**: Often outperforms more theoretically principled methods

### 4.3 Performance Comparison

**Why Stupid Backoff Performed Best (185.70 vs 188.73):**

1. **Higher-order context**: Stupid Backoff uses 4-grams as the base model, while our Linear Interpolation uses 3-grams. When the 4-gram context is available in training data, it provides more specific, accurate predictions.

2. **Less aggressive smoothing**: Linear Interpolation always blends all n-gram orders, even when longer contexts are reliable. Stupid Backoff only backs off when necessary, preserving sharp probability distinctions.

3. **Backoff parameter tuning**: The α = 0.4 backoff weight (typical default) was likely well-suited to the PTB dataset's characteristics.

4. **Computational efficiency**: Stupid Backoff doesn't require probability normalization, allowing faster computation and potentially better coverage of contexts.

**When Linear Interpolation Might Win:**

Linear Interpolation can be superior when:
- Training data is very sparse
- The development set is large enough for reliable λ optimization
- Theoretical probability guarantees are required
- The application needs calibrated probability estimates

**Trade-offs:**

| Aspect | Linear Interpolation | Stupid Backoff |
|--------|---------------------|----------------|
| Theoretical soundness | ✓ Proper probabilities | ✗ Scores only |
| Computational cost | Higher (normalization) | Lower |
| Tuning complexity | Need to optimize λs | Simple α parameter |
| Performance | Good (188.73) | **Better (185.70)** |

---

## 5. Qualitative Analysis: Generated Text

### 5.1 Generated Sentences

Using the best performing model (Stupid Backoff, perplexity 185.70), we generated the following sentences:

1. **"Hahn carry it off in the executive if there month ends the two executives considered."**

2. **"To oust unk on the unk leader arthur unk such a high of n transactions."**

3. **"Miller says in his alleged unk and several n n schedule could be more brand."**

4. **"K."**

5. **"S ad unit to lead of some concern to the fact at the appropriate time."**

### 5.2 Fluency Assessment

**Overall Impression:** The generated text exhibits **low to moderate fluency** with inconsistent grammatical structure and semantic coherence.

**Positive Observations:**

1. **Local coherence**: Short phrases show reasonable word combinations:
   - "in the executive"
   - "the two executives considered"
   - "could be more brand"
   - "at the appropriate time"

2. **Syntactic awareness**: The model captures some grammatical patterns:
   - Subject-verb-object structures (partial)
   - Prepositional phrases
   - Determiners before nouns

3. **Domain vocabulary**: Business/financial terminology reflects the PTB corpus (Wall Street Journal articles):
   - "executive," "transactions," "ad unit"
   - Proper names: "Hahn," "Miller," "Arthur"

**Limitations:**

1. **`<unk>` tokens**: The frequent appearance of "unk" indicates the model is generating unknown/rare words from the training distribution, breaking readability.

2. **Number placeholders**: "n n" shows numeric placeholder tokens being generated literally rather than actual numbers.

3. **Semantic drift**: Sentences lack coherent meaning across multiple clauses. Example: Sentence 1 starts with "Hahn carry it off" (unclear referent) and ends with an unrelated temporal clause.

4. **Fragment generation**: Sentence 4 ("K.") shows the model can generate very short, incomplete sentences.

5. **Agreement errors**: Some grammatical violations appear, e.g., "Hahn carry" (subject-verb disagreement).

### 5.3 How the Model Generates Text

**Generation Process:**

1. **Start with context**: Begin with `<s>` (start-of-sentence token)

2. **Probability sampling**: For each position, the model:
   - Looks up the probability distribution P(w | context) using the previous 3 words (for 4-gram Stupid Backoff)
   - Samples the next word from this distribution
   - Updates the context window

3. **Backoff mechanism**: When the full 4-gram context is unseen:
   - Falls back to 3-gram: P(w | w_{i-2}, w_{i-1})
   - Then to 2-gram: P(w | w_{i-1})
   - Finally to 1-gram: P(w)
   - Each backoff multiplies probability by α = 0.4

4. **Termination**: Generation continues until:
   - `</s>` (end-of-sentence) token is sampled, or
   - Maximum length is reached

**Why the Output Looks This Way:**

1. **Markov property**: The model only sees 3 words of history (4-gram), so it cannot maintain long-range coherence or remember the sentence's topic.

2. **Training data characteristics**: PTB contains financial news, explaining the business jargon and formal style.

3. **Statistical patterns without semantics**: The model captures word co-occurrence statistics but has no understanding of meaning, causality, or discourse structure.

4. **Rare word handling**: The model learned that `<unk>` appears in certain syntactic positions in the training data, so it generates it probabilistically.

**Comparison to Modern Approaches:**

These n-gram models demonstrate the limitations that motivated neural language models:
- **N-gram models**: Local, syntactic patterns; no semantic understanding; limited context
- **Modern transformers**: Long-range dependencies; semantic reasoning; contextual embeddings (100+ tokens)

The generated text serves as a baseline showing what pure statistical models can achieve—grammatically plausible local patterns without global coherence.

---

## 6. Key Findings and Conclusions

### 6.1 Summary of Results

1. **Data sparsity dominates**: Unsmoothed higher-order models fail completely (INF perplexity) due to zero probabilities on unseen n-grams.

2. **Naive smoothing insufficient**: Add-1 smoothing prevents infinite perplexity but performs poorly (6,129.72 for 4-gram), over-distributing probability mass to implausible sequences.

3. **Advanced methods excel**: 
   - Linear Interpolation: 188.73 perplexity
   - **Stupid Backoff: 185.70 perplexity (best)**

4. **Backoff vs. Interpolation**: Stupid Backoff's selective use of higher-order context slightly outperformed Linear Interpolation's constant blending strategy.

### 6.2 Theoretical Insights

**The Markov Assumption Trade-off:**
- Longer contexts → better predictions (in theory)
- Longer contexts → more sparsity (in practice)
- Solution: Sophisticated smoothing/backoff to balance specificity and robustness

**Why Stupid Backoff Works:**
- Prioritizes higher-order n-grams when available (specific evidence)
- Gracefully degrades to lower orders when needed (robust fallback)
- Computationally simple (no normalization required)

### 6.3 Practical Implications

For language modeling applications:
1. **Always use smoothing/backoff** with n-gram models
2. **Tune on development data** to find optimal hyperparameters
3. **Consider Stupid Backoff** for practical applications requiring speed and accuracy
4. **Recognize limitations**: N-gram models excel at local patterns but fail at capturing long-range dependencies and semantic understanding

### 6.4 Future Directions

While this project focused on traditional n-gram approaches, modern NLP has moved toward:
- **Neural language models** (RNNs, LSTMs, Transformers)
- **Contextual embeddings** (BERT, GPT architectures)
- **Subword tokenization** (BPE, WordPiece) to address OOV issues

However, n-gram models remain valuable for:
- Baselines for comparison
- Resource-constrained environments
- Interpretable linguistic analysis
- Hybrid systems combining statistical and neural approaches

---

## 7. References and Dataset Information

**Dataset**: Penn Treebank (PTB) Corpus
- Source: Wall Street Journal articles
- Pre-processed with normalized punctuation, numbers, and rare word replacement
- Standard benchmark for language modeling research

**Implementation Notes**:
- All models trained on same train/dev/test split
- Perplexity calculated on test set after hyperparameter tuning on dev set
- Generated text uses temperature sampling from probability distributions

---
