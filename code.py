import numpy as np
import re
from collections import defaultdict, Counter
import math
from typing import List, Dict, Tuple
import random
import os

class NGramsProcessor:
    
    
    def __init__(self):
        self.vocab = set()
        self.vocab_size = 0
        self.unk_token = "<UNK>"
        self.start_token = "<s>"
        self.end_token = "</s>"
        
    def preprocess(self, text: str) -> List[str]:
        """Tokenize and preprocess text"""
        # Add sentence boundaries and tokenize
        sentences = re.split(r'[.!?]+', text)
        tokens = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Add start token
            tokens.append(self.start_token)
            
            # Tokenize words
            words = re.findall(r'\b\w+\b', sentence.lower())
            tokens.extend(words)
            
            # Add end token
            tokens.append(self.end_token)
            
        return tokens
    
    def build_vocab(self, tokens: List[str], min_freq: int = 2):
        """Build vocabulary from tokens"""
        word_counts = Counter(tokens)
        # Keep words that appear at least min_freq times
        self.vocab = {word for word, count in word_counts.items() 
                     if count >= min_freq or word in [self.start_token, self.end_token, self.unk_token]}
        self.vocab_size = len(self.vocab)
        
    def replace_rare_words(self, tokens: List[str]) -> List[str]:
        """Replace rare words with UNK token"""
        return [token if token in self.vocab else self.unk_token for token in tokens]

class NGramsModel:
    """Base N-gram Language Model"""
    
    def __init__(self, n: int):
        self.n = n
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocab = set()
        self.vocab_size = 0
        self.processor = NGramsProcessor()
        
    def train(self, text: str):
        """Train the n-gram model"""
        tokens = self.processor.preprocess(text)
        self.processor.build_vocab(tokens)
        tokens = self.processor.replace_rare_words(tokens)
        self.vocab = self.processor.vocab
        self.vocab_size = len(self.vocab)
        
        # Count n-grams
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            context = tuple(tokens[i:i + self.n - 1])
            
            self.ngram_counts[ngram] += 1
            self.context_counts[context] += 1
            
    def get_probability(self, ngram: Tuple[str], context: Tuple[str]) -> float:
        """Get probability using MLE - to be overridden by subclasses"""
        raise NotImplementedError
        
    def perplexity(self, text: str) -> float:
        """Calculate perplexity on given text"""
        tokens = self.processor.preprocess(text)
        tokens = self.processor.replace_rare_words(tokens)
        
        log_prob_sum = 0
        word_count = 0
        
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            context = tuple(tokens[i:i + self.n - 1])
            
            prob = self.get_probability(ngram, context)
            
            if prob == 0:
                return float('inf')
                
            log_prob_sum += math.log(prob)
            word_count += 1
            
        if word_count == 0:
            return float('inf')
            
        avg_log_prob = log_prob_sum / word_count
        return math.exp(-avg_log_prob)

class MLEModel(NGramsModel):
    """Maximum Likelihood Estimation N-gram Model (Pure MLE, no smoothing)"""
    
    def get_probability(self, ngram: Tuple[str], context: Tuple[str]) -> float:
        if self.context_counts[context] == 0:
            return 0
        return self.ngram_counts[ngram] / self.context_counts[context]

class AddOneSmoothingModel(NGramsModel):
    """Add-1 (Laplace) Smoothing Model"""
    
    def get_probability(self, ngram: Tuple[str], context: Tuple[str]) -> float:
        return (self.ngram_counts[ngram] + 1) / (self.context_counts[context] + self.vocab_size)

class LinearInterpolationModel:
    """Linear Interpolation of unigram, bigram, and trigram models"""
    
    def __init__(self, lambda1: float, lambda2: float, lambda3: float):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.unigram_model = None
        self.bigram_model = None
        self.trigram_model = None
        self.vocab = set()
        self.vocab_size = 0
        self.processor = NGramsProcessor()
        self.n = 3  # We're building a trigram model
        
    def train(self, text: str):
        """Train unigram, bigram, and trigram models"""
        # Train individual models
        self.unigram_model = MLEModel(1)
        self.bigram_model = MLEModel(2)
        self.trigram_model = MLEModel(3)
        
        self.unigram_model.train(text)
        self.bigram_model.train(text)
        self.trigram_model.train(text)
        
        # Set vocabulary
        self.vocab = self.trigram_model.vocab
        self.vocab_size = self.trigram_model.vocab_size
        self.processor = self.trigram_model.processor
        
    def get_probability(self, ngram: Tuple[str], context: Tuple[str]) -> float:
        """Get interpolated probability for any n-gram length"""
        if len(ngram) == 0:
            return 0
            
        word = ngram[-1]
        
        # Unigram probability
        uni_prob = self.unigram_model.get_probability((word,), ())
        
        # Bigram probability
        if len(ngram) >= 2:
            prev_word = ngram[-2]
            bi_prob = self.bigram_model.get_probability((prev_word, word), (prev_word,))
        else:
            bi_prob = uni_prob  # Fallback to unigram
            
        # Trigram probability  
        if len(ngram) >= 3:
            prev_prev_word = ngram[-3]
            prev_word = ngram[-2]
            tri_prob = self.trigram_model.get_probability((prev_prev_word, prev_word, word), 
                                                         (prev_prev_word, prev_word))
        else:
            tri_prob = bi_prob  # Fallback to bigram
        
        # Linear interpolation
        return (self.lambda1 * uni_prob + 
                self.lambda2 * bi_prob + 
                self.lambda3 * tri_prob)
    
    def perplexity(self, text: str) -> float:
        """Calculate perplexity for trigram context"""
        tokens = self.processor.preprocess(text)
        tokens = self.processor.replace_rare_words(tokens)
        
        log_prob_sum = 0
        word_count = 0
        
        # Use trigram context for evaluation
        for i in range(2, len(tokens)):  # Start from 2 to have trigram context
            if i < len(tokens):
                ngram = tuple(tokens[max(0, i-2):i+1])  # Get up to trigram
                context = tuple(tokens[max(0, i-2):i])   # Context for trigram
                
                # Ensure we have proper n-gram length
                if len(ngram) >= 1:
                    prob = self.get_probability(ngram, context)
                    
                    if prob == 0:
                        return float('inf')
                        
                    log_prob_sum += math.log(prob)
                    word_count += 1
            
        if word_count == 0:
            return float('inf')
            
        avg_log_prob = log_prob_sum / word_count
        return math.exp(-avg_log_prob)

class StupidBackoffModel:
    """Stupid Backoff Model"""
    
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self.unigram_model = None
        self.bigram_model = None
        self.trigram_model = None
        self.vocab = set()
        self.vocab_size = 0
        self.processor = NGramsProcessor()
        self.n = 3  # Trigram model
        
    def train(self, text: str):
        """Train unigram, bigram, and trigram models"""
        self.unigram_model = MLEModel(1)
        self.bigram_model = MLEModel(2)
        self.trigram_model = MLEModel(3)
        
        self.unigram_model.train(text)
        self.bigram_model.train(text)
        self.trigram_model.train(text)
        
        self.vocab = self.trigram_model.vocab
        self.vocab_size = self.trigram_model.vocab_size
        self.processor = self.trigram_model.processor
        
    def get_probability(self, ngram: Tuple[str], context: Tuple[str]) -> float:
        """Get probability using stupid backoff for any n-gram length"""
        if len(ngram) == 0:
            return 0
            
        word = ngram[-1]
        
        # Try highest order n-gram first
        if len(ngram) >= 3:
            prev_prev_word = ngram[-3]
            prev_word = ngram[-2]
            tri_context = (prev_prev_word, prev_word)
            tri_ngram = (prev_prev_word, prev_word, word)
            
            if self.trigram_model.context_counts[tri_context] > 0 and \
               self.trigram_model.ngram_counts[tri_ngram] > 0:
                return (self.trigram_model.ngram_counts[tri_ngram] / 
                       self.trigram_model.context_counts[tri_context])
        
        # Backoff to bigram
        if len(ngram) >= 2:
            prev_word = ngram[-2]
            bi_context = (prev_word,)
            bi_ngram = (prev_word, word)
            
            if self.bigram_model.context_counts[bi_context] > 0 and \
               self.bigram_model.ngram_counts[bi_ngram] > 0:
                return self.alpha * (self.bigram_model.ngram_counts[bi_ngram] / 
                                   self.bigram_model.context_counts[bi_context])
        
        # Backoff to unigram
        return self.alpha * self.alpha * self.unigram_model.get_probability((word,), ())
    
    def perplexity(self, text: str) -> float:
        """Calculate perplexity for trigram context"""
        tokens = self.processor.preprocess(text)
        tokens = self.processor.replace_rare_words(tokens)
        
        log_prob_sum = 0
        word_count = 0
        
        for i in range(2, len(tokens)):  # Start from 2 to have trigram context
            if i < len(tokens):
                ngram = tuple(tokens[max(0, i-2):i+1])
                context = tuple(tokens[max(0, i-2):i])
                
                if len(ngram) >= 1:
                    prob = self.get_probability(ngram, context)
                    
                    if prob == 0:
                        return float('inf')
                        
                    log_prob_sum += math.log(prob)
                    word_count += 1
            
        if word_count == 0:
            return float('inf')
            
        avg_log_prob = log_prob_sum / word_count
        return math.exp(-avg_log_prob)

class DataLoader:
    """Handles dataset loading and validation"""
    
    def check_and_load_files():
        """Check for dataset files and load them"""
        possible_files = {
            'train': ['ptb.train.txt', 'train.txt', 'training.txt', 'ptb_train.txt'],
            'valid': ['ptb.valid.txt', 'valid.txt', 'validation.txt', 'dev.txt', 'ptb_valid.txt'],
            'test': ['ptb.test.txt', 'test.txt', 'testing.txt', 'ptb_test.txt']
        }
        
        loaded_data = {}
        
        for data_type, filenames in possible_files.items():
            file_found = False
            for filename in filenames:
                if os.path.exists(filename):
                    try:
                        with open(filename, 'r', encoding='utf-8') as f:
                            loaded_data[data_type] = f.read()
                        print(f"✓ Loaded {data_type} data from: {filename}")
                        file_found = True
                        break
                    except Exception as e:
                        print(f"✗ Error reading {filename}: {e}")
                        continue
            
            if not file_found:
                print(f"✗ Could not find {data_type} data file. Tried: {', '.join(filenames)}")
                loaded_data[data_type] = None
        
        return loaded_data.get('train'), loaded_data.get('valid'), loaded_data.get('test')
    

class LanguageModelEvaluator:
    """Comprehensive evaluator for all language models"""
    
    def __init__(self):
        self.results = {}
        
    def evaluate_models(self, train_text: str, dev_text: str, test_text: str):
        """Evaluate all required models"""
        
        if not train_text or not dev_text or not test_text:
            raise ValueError("Missing required dataset files")
        
        print("\n" + "="*60)
        print("TRAINING AND EVALUATING LANGUAGE MODELS")
        print("="*60)
        
        print("\n--- Pure MLE Models (No Smoothing) ---")
        # Pure MLE Models
        for n in [1, 2, 3, 4]:
            model = MLEModel(n)
            model.train(train_text)
            perplexity = model.perplexity(test_text)
            self.results[f'MLE_{n}gram'] = perplexity
            print(f"MLE {n}-gram Perplexity: {perplexity if perplexity != float('inf') else 'INF'}")
        
        print("\n--- Add-1 Smoothing Models ---")
        # Add-1 Smoothing for all n-gram orders
        for n in [1, 2, 3, 4]:
            model = AddOneSmoothingModel(n)
            model.train(train_text)
            perplexity = model.perplexity(test_text)
            self.results[f'Add1_{n}gram'] = perplexity
            print(f"Add-1 Smoothing {n}-gram Perplexity: {perplexity:.2f}")
        
        print("\n--- Linear Interpolation Model ---")
        # Linear Interpolation with weight tuning
        best_lambdas = self.tune_interpolation_weights(train_text, dev_text)
        interp_model = LinearInterpolationModel(*best_lambdas)
        interp_model.train(train_text)
        self.results['Linear_Interpolation'] = interp_model.perplexity(test_text)
        print(f"Linear Interpolation Perplexity: {self.results['Linear_Interpolation']:.2f}")
        
        print("\n--- Stupid Backoff Model ---")
        # Stupid Backoff
        stupid_backoff = StupidBackoffModel(alpha=0.4)
        stupid_backoff.train(train_text)
        self.results['Stupid_Backoff'] = stupid_backoff.perplexity(test_text)
        print(f"Stupid Backoff Perplexity: {self.results['Stupid_Backoff']:.2f}")
        
        return self.results
    
    def tune_interpolation_weights(self, train_text: str, dev_text: str) -> Tuple[float, float, float]:
        """Find optimal interpolation weights using development data"""
        best_perplexity = float('inf')
        best_lambdas = (0.33, 0.33, 0.34)
        
        # Try different lambda combinations
        lambda_combinations = [
            (0.1, 0.3, 0.6),
            (0.2, 0.3, 0.5),
            (0.33, 0.33, 0.34),
            (0.1, 0.2, 0.7),
            (0.6, 0.3, 0.1)
        ]
        
        print("Testing interpolation weights on development set:")
        for lambdas in lambda_combinations:
            model = LinearInterpolationModel(*lambdas)
            model.train(train_text)
            perplexity = model.perplexity(dev_text)
            
            print(f"  Lambdas {lambdas}: Perplexity = {perplexity:.2f}")
            
            if perplexity < best_perplexity:
                best_perplexity = perplexity
                best_lambdas = lambdas
                
        print(f"✓ Best lambdas: {best_lambdas}, Dev Perplexity: {best_perplexity:.2f}")
        return best_lambdas

class TextGenerator:
    """Generate text using trained language models"""
    
    def __init__(self, model):
        self.model = model
        
    def generate_text(self, max_length: int = 20, start_words: List[str] = None) -> str:
        """Generate text using the model"""
        if start_words is None:
            start_words = [self.model.processor.start_token]
            
        generated = start_words.copy()
        
        for _ in range(max_length):
            # Get context (last n-1 words)
            context_start = max(0, len(generated) - (self.model.n - 1))
            context = tuple(generated[context_start:])
            
            # Get possible next words and their probabilities
            next_word_probs = []
            for word in self.model.vocab:
                if word == self.model.processor.start_token:
                    continue
                    
                # Create n-gram by adding the candidate word
                ngram = context + (word,)
                
                try:
                    prob = self.model.get_probability(ngram, context)
                    if prob > 0:
                        next_word_probs.append((word, prob))
                except (ValueError, KeyError):
                    # Handle cases where ngram might not be supported
                    continue
            
            if not next_word_probs:
                # If no probabilities found, end the sentence
                generated.append(self.model.processor.end_token)
                break
                
            # Sample based on probabilities
            words, probs = zip(*next_word_probs)
            next_word = random.choices(words, weights=probs, k=1)[0]
            
            generated.append(next_word)
            
            if next_word == self.model.processor.end_token:
                break
                
        # Filter out special tokens for final output
        return ' '.join([word for word in generated if word not in 
                        [self.model.processor.start_token, self.model.processor.end_token]])

# Main execution
def main():
    print("N-Gram Language Modeling System")
    print("=" * 50)
    
    # Check and load dataset files
    print("Checking for dataset files...")
    train_data, dev_data, test_data = DataLoader.check_and_load_files()
    
    # If files not found, use sample data
    if not train_data or not dev_data or not test_data:
        print("\nUsing sample dataset for demonstration.")
        train_data, dev_data, test_data = DataLoader.create_sample_data()
    
    # Display dataset statistics
    print(f"\nDataset Statistics:")
    print(f"Training data: {len(train_data.split())} words")
    print(f"Development data: {len(dev_data.split())} words") 
    print(f"Test data: {len(test_data.split())} words")
    
    # Initialize evaluator
    evaluator = LanguageModelEvaluator()
    
    try:
        # Evaluate all models
        results = evaluator.evaluate_models(train_data, dev_data, test_data)
        
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        
        print("\nPure MLE Models:")
        for model_name in ['MLE_1gram', 'MLE_2gram', 'MLE_3gram', 'MLE_4gram']:
            perplexity = results[model_name]
            if perplexity == float('inf'):
                print(f"  {model_name:15}: INF")
            else:
                print(f"  {model_name:15}: {perplexity:.2f}")
        
        print("\nAdd-1 Smoothing Models:")
        for model_name in ['Add1_1gram', 'Add1_2gram', 'Add1_3gram', 'Add1_4gram']:
            perplexity = results[model_name]
            print(f"  {model_name:15}: {perplexity:.2f}")
        
        print("\nAdvanced Models:")
        print(f"  Linear_Interpolation: {results['Linear_Interpolation']:.2f}")
        print(f"  Stupid_Backoff:       {results['Stupid_Backoff']:.2f}")
        
        # Generate text with best model
        print("\n" + "="*60)
        print("TEXT GENERATION")
        print("="*60)
        
        # Find best model for text generation (excluding INF values)
        valid_results = {k: v for k, v in results.items() if v != float('inf')}
        if valid_results:
            best_model_name = min(valid_results, key=valid_results.get)
            print(f"Best model: {best_model_name} (Perplexity: {valid_results[best_model_name]:.2f})")
            
            # Re-train best model for generation
            if 'Interpolation' in best_model_name:
                best_model = LinearInterpolationModel(0.2, 0.3, 0.5)
            elif 'Backoff' in best_model_name:
                best_model = StupidBackoffModel(alpha=0.4)
            elif 'Add1' in best_model_name:
                n = int(best_model_name.split('_')[1][0])  # Extract n from 'Add1_Ngram'
                best_model = AddOneSmoothingModel(n)
            else:
                n = int(best_model_name.split('_')[1][0])  # Extract n from 'MLE_Ngram'
                best_model = MLEModel(n)
                
            best_model.train(train_data)
            generator = TextGenerator(best_model)
            
            print(f"\nGenerated Sentences using {best_model_name}:")
            for i in range(5):
                sentence = generator.generate_text(max_length=15)
                print(f"{i+1}. {sentence.capitalize()}.")
        else:
            print("No valid models for text generation (all perplexities are INF)")
            
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        print("Please check your dataset files and try again.")

if __name__ == "__main__":
    main()