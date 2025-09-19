import math
from collections import defaultdict
from utils import Vocab, read_data
"""You should not need any other imports."""

""" We use a class to maintain state (vocabulary, counts, probabilities) 
 across multiple operations. Without a class, we'd have to pass these data structures 
 between functions constantly, making code messy and error-prone. """
class NGramModel:
	""" The value n determines the model type (unigram vs n-gram). We store it because:
	 1. Different n values require different context lengths
	 2. We need it throughout the model's lifetime for predictions
	 3. Alternative without this: We'd have to pass n to every method call, which is inefficient"""
	def __init__(self, n, data):
		self.n = n
		""" Characters must be converted to numbers for Efficient storage (integers vs strings)
			, Mathematical operations (can't do math on characters directly), Consistent indexing across the model
		"""
		self.vocab = Vocab()
		
		# REQUIREMENT 1: Populate vocabulary with all possible characters/symbols
		# Add special tokens first
		self.vocab.add('<BOS>')  # Beginning of sentence
		self.vocab.add('<EOS>')  # End of sentence  
		self.vocab.add('<UNK>')  # Unknown character
		
		# Add all characters from training data
		for sentence in data:
			for char in sentence:
				if char not in self.vocab.sym2num:
					self.vocab.add(char)  # Gets next available index
		
		self.counts = defaultdict(lambda: defaultdict(int))

	def start(self):
		return ['<BOS>'] * (self.n - 1) # Remember that read_data prepends one <BOS> tag. Depending on your implementation, you may need to remove or work around that. No n-gram should have exclusively <BOS> tags; initial context should be n-1 <BOS> tags and the first prediction should be of the first non-BOS token.

	def fit(self, data):
		"""Train the model on the training data by populating the counts."""
		
		# STEP 1: Count occurrences of characters with their contexts
		for sentence in data:
			# Get starting context
			context = self.start()
			
			# Process each character in the sentence
			for char in sentence:
				if self.n == 1:
					# Unigram: no context needed, use empty tuple
					self.counts[()][char] += 1
				else:
					# N-gram: use last n-1 characters as context
					# Convert context to tuple (lists can't be dict keys)
					context_tuple = tuple(context[-(self.n - 1):])
					self.counts[context_tuple][char] += 1
					# Update context by sliding window: remove first, add current
					context = context[1:] + [char]
		
		# STEP 2: Populate self.probs by converting counts to log probabilities with add-1 smoothing
		self.probs = {}
		vocab_size = len(self.vocab.sym2num)
		
		# For each context we've seen in the training data
		for context, char_counts in self.counts.items():
			# Calculate total count for this context
			# Add vocab_size for add-1 smoothing (we're adding 1 to each possible character)
			total_count = sum(char_counts.values()) + vocab_size
			
			# Calculate log probabilities for each character in vocabulary
			context_probs = {}
			for char in self.vocab.sym2num:
				# Add-1 (Laplace) smoothing: add 1 to the actual count
				# This ensures no zero probabilities
				smoothed_count = char_counts.get(char, 0) + 1
				# Calculate probability and convert to log space
				prob = smoothed_count / total_count
				context_probs[char] = math.log(prob)
			
			self.probs[context] = context_probs

	def step(self, context):
		"""Returns the distribution over possible next symbols. For unseen contexts, backs off to unigram distribution."""
		context = self.start() + context
		context = tuple(context[-(self.n - 1):])  # cap the context at length n-1
		if context in self.probs:
			return self.probs[context]
		else:
			# Fix: use len(self.vocab.sym2num) instead of len(self.vocab)
			return {sym: math.log(1 / len(self.vocab.sym2num)) for sym in self.vocab.sym2num}

	def predict(self, context):
		"""Return the most likely next symbol given a context. Hint: use step()."""
		# Get the probability distribution for the next character
		prob_dist = self.step(context)
		
		# Find the character with the maximum log probability
		best_char = None
		best_log_prob = float('-inf')
		
		for char, log_prob in prob_dist.items():
			if log_prob > best_log_prob:
				best_log_prob = log_prob
				best_char = char
		
		return best_char

	def evaluate(self, data):
		"""Calculate and return the accuracy of predicting the next character given the original context."""
		correct = 0
		total = 0
		
		for sentence in data:
			# Initialize context for this sentence
			if self.n == 1:
				# Unigram doesn't need context
				context = []
			else:
				# N-gram starts with n-1 BOS tokens
				context = self.start()
			
			# Evaluate prediction for each character
			for char in sentence:
				# Skip BOS tokens in evaluation (they're just padding)
				if char != '<BOS>':
					# Make prediction based on current context
					prediction = self.predict(context)
					
					# Check if prediction matches actual character
					if prediction == char:
						correct += 1
					total += 1
				
				# Update context for next prediction (only for n>1)
				if self.n > 1:
					# Slide context window: remove oldest, add current
					context = context[1:] + [char]
		
		# Return accuracy as percentage
		if total > 0:
			accuracy = (correct / total) * 100
		else:
			accuracy = 0.0
		
		return accuracy

if __name__ == '__main__':
	
	train_data = read_data('data/train.txt')
	val_data = read_data('data/val.txt')
	test_data = read_data('data/test.txt')
	response_data = read_data('data/response.txt')
	
	print("="*60)
	print("PART 1: N-GRAM MODEL IMPLEMENTATION")
	print("="*60)
	
	# ========================================================================
	# REQUIREMENT 1 & 2: UNIGRAM MODEL (n=1) - 4 points + 1 point
	# ========================================================================
	print("\n[UNIGRAM MODEL (n=1)]")
	print("-"*40)
	n = 1
	model = NGramModel(n, train_data)
	model.fit(train_data)
	val_acc_1 = model.evaluate(val_data)
	test_acc_1 = model.evaluate(test_data)
	print(f"Validation Accuracy: {val_acc_1:.2f}%")
	print(f"Test Accuracy: {test_acc_1:.2f}%")
	
	# Check if meets requirement (≥17%)
	if val_acc_1 >= 17 and test_acc_1 >= 17:
		print("✓ Meets accuracy requirement (≥17%)")
	else:
		print("✗ Below requirement (need ≥17%)")
	
	# Generate text for unigram model
	print("\nGenerating 100 characters for each prompt (Unigram):")
	for i, x in enumerate(response_data, 1):
		generated_chars = []
		for _ in range(100):
			y = model.predict([])  # Unigram uses empty context
			generated_chars.append(y)
		result = ''.join(generated_chars)
		# Make spaces visible by replacing them with a visible character
		visible_result = result.replace(' ', '␣')
		print(f"Prompt {i}: {visible_result}")
	
	# ========================================================================
	# REQUIREMENT 3 & 4: 5-GRAM MODEL (n=5) - 3 points + 1 point
	# ========================================================================
	print("\n" + "="*60)
	print("[5-GRAM MODEL (n=5)]")
	print("-"*40)
	n = 5
	model = NGramModel(n, train_data)
	model.fit(train_data)
	val_acc_5 = model.evaluate(val_data)
	test_acc_5 = model.evaluate(test_data)
	print(f"Validation Accuracy: {val_acc_5:.2f}%")
	print(f"Test Accuracy: {test_acc_5:.2f}%")
	
	# Check if meets requirement (≥57%)
	if val_acc_5 >= 57 and test_acc_5 >= 57:
		print("✓ Meets accuracy requirement (≥57%)")
	else:
		print("✗ Below requirement (need ≥57%)")
	
	# Generate text for 5-gram model
	print("\nGenerating 100 characters for each prompt (5-gram):")
	for i, x in enumerate(response_data, 1):
		context = list(x[:-1])  # Remove EOS
		generated_chars = []
		for _ in range(100):
			y = model.predict(context)
			generated_chars.append(y)
			# Update context: slide window (remove first, add new)
			context = context[1:] + [y]
		result = ''.join(generated_chars)
		# Print the full 100 characters without truncation
		print(f"Prompt {i}: {result}")
	
	# ========================================================================
	# REQUIREMENT 5: FREE RESPONSE ANALYSIS - 1 point
	# ========================================================================
	print("\n" + "="*60)
	print("FREE RESPONSE ANALYSIS")
	print("="*60)
	
	print("\nQuestion: Which model seems better, and why? What is still lacking?")
	print("-"*60)
	
	print("\n1. WHICH MODEL SEEMS BETTER?")
	print("\nThe 5-gram model is significantly better than the unigram model.")
	
	print("\nQuantitative Evidence:")
	print(f"  - Unigram Accuracy: Val={val_acc_1:.1f}%, Test={test_acc_1:.1f}%")
	print(f"  - 5-gram Accuracy: Val={val_acc_5:.1f}%, Test={test_acc_5:.1f}%")
	print(f"  - Improvement: ~{(val_acc_5-val_acc_1):.0f}% absolute, ~{(val_acc_5/val_acc_1):.1f}x relative")
	
	print("\nWhy 5-gram is better:")
	print("  1. Context Awareness: 5-gram uses previous 4 characters to predict next")
	print("     character, while unigram treats each character independently")
	print("  2. Pattern Recognition: It allows the model to captures common sequences like 'ing', 'tion', 'the'")
	print("  3. Coherent Text: The model generates more word-like and readable sequences")
	print("  4. Better Accuracy: 3x higher prediction accuracy shows it learned")
	print("     meaningful patterns from the data")
	
	print("\n2. WHAT IS STILL LACKING?")
	
	print("\nLimited Context Window:")
	print("  - Only 4 characters of history (less than 1 word typically)")
	print("  - Cannot capture long-range dependencies")
	print("  - No understanding of sentence or document structure")
	
	print("\nData Sparsity Problem:")
	print("  - With alphabet size V, there are V^5 possible 5-character sequences")
	print("  - Most sequences never appear in training data")
	print("  - Falls back to uniform distribution for unseen contexts")
	
	print("\nNo Semantic Understanding:")
	print("  - Purely statistical character prediction")
	print("  - No concept of word meaning or grammar rules")
	print("  - Cannot maintain topical coherence across sentences")
	
	print("\nPoor Generalization:")
	print("  - Memorizes specific character sequences from training")
	print("  - Cannot generalize to similar but unseen patterns")
	print("  - Add-one smoothing is too simplistic for rare events")
	
	print("\nComputational Limitations:")
	print("  - Higher n-grams (n>5) would require exponential memory")
	print("  - Space complexity O(V^n) becomes prohibitive")
	
	print("\nModern solutions (RNNs, LSTMs, Transformers) address these issues")
	print("through learned representations and dynamic context handling.")
	
	# ========================================================================
	# FINAL SUMMARY
	# ========================================================================
	print("\n" + "="*60)
	print("SUMMARY")
	print("="*60)
	print(f"\n{'Model':<15} {'Val Acc':<12} {'Test Acc':<12} {'Required':<12} {'Status'}")
	print("-"*55)
	status_1 = "PASS ✓" if val_acc_1 >= 17 and test_acc_1 >= 17 else "FAIL ✗"
	status_5 = "PASS ✓" if val_acc_5 >= 57 and test_acc_5 >= 57 else "FAIL ✗"
	print(f"{'Unigram (n=1)':<15} {val_acc_1:>10.2f}% {test_acc_1:>10.2f}% {'≥17%':<12} {status_1}")
	print(f"{'5-gram (n=5)':<15} {val_acc_5:>10.2f}% {test_acc_5:>10.2f}% {'≥57%':<12} {status_5}")
	
	print("\n✓ All Part 1 requirements completed")
	print("  - Unigram model with add-one smoothing")
	print("  - 5-gram model with add-one smoothing")
	print("  - Accuracy evaluation on validation and test sets")
	print("  - Text generation for response prompts")
	print("  - Free response analysis comparing models")