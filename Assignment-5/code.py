import nltk
from nltk.tokenize import (
    WhitespaceTokenizer,
    WordPunctTokenizer,
    TreebankWordTokenizer,
    TweetTokenizer,
    MWETokenizer
)
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
import string

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

print("="*80)
print("NLP ASSIGNMENT: TOKENIZATION, STEMMING, AND LEMMATIZATION")
print("="*80)

# Sample text for demonstration
sample_text = """Natural Language Processing (NLP) is amazing! 
It's used in chatbots, translation, and sentiment analysis. 
#NLP #AI @TechGuru Check out https://nltk.org for more info. 
The runners are running faster than they ran yesterday."""

tweet_text = "RT @DataScience: Machine learning is revolutionizing AI! 😊 #MachineLearning #DeepLearning https://example.com"

print("\n" + "="*80)
print("ORIGINAL TEXT:")
print("="*80)
print(sample_text)
print("\nTWEET TEXT:")
print(tweet_text)

# ============================================================================
# PART 1: TOKENIZATION
# ============================================================================

print("\n" + "="*80)
print("PART 1: TOKENIZATION TECHNIQUES")
print("="*80)

# 1. Whitespace Tokenization
print("\n1. WHITESPACE TOKENIZATION")
print("-" * 60)
whitespace_tokenizer = WhitespaceTokenizer()
whitespace_tokens = whitespace_tokenizer.tokenize(sample_text)
print(f"Tokens: {whitespace_tokens[:15]}...")
print(f"Total tokens: {len(whitespace_tokens)}")

# 2. Punctuation-based Tokenization (WordPunctTokenizer)
print("\n2. PUNCTUATION-BASED TOKENIZATION")
print("-" * 60)
punct_tokenizer = WordPunctTokenizer()
punct_tokens = punct_tokenizer.tokenize(sample_text)
print(f"Tokens: {punct_tokens[:20]}...")
print(f"Total tokens: {len(punct_tokens)}")

# 3. Treebank Tokenization
print("\n3. TREEBANK TOKENIZATION")
print("-" * 60)
treebank_tokenizer = TreebankWordTokenizer()
treebank_tokens = treebank_tokenizer.tokenize(sample_text)
print(f"Tokens: {treebank_tokens[:20]}...")
print(f"Total tokens: {len(treebank_tokens)}")

# 4. Tweet Tokenization
print("\n4. TWEET TOKENIZATION")
print("-" * 60)
tweet_tokenizer = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=False)
tweet_tokens = tweet_tokenizer.tokenize(tweet_text)
print(f"Tokens: {tweet_tokens}")
print(f"Total tokens: {len(tweet_tokens)}")
print("\nFeatures demonstrated:")
print("  - Preserves @mentions and #hashtags")
print("  - Handles emoticons and URLs")
print("  - Reduces repeated characters (e.g., 'coooool' -> 'cool')")

# 5. Multi-Word Expression (MWE) Tokenization
print("\n5. MULTI-WORD EXPRESSION (MWE) TOKENIZATION")
print("-" * 60)
# First tokenize with basic tokenizer
base_tokens = treebank_tokenizer.tokenize(sample_text)

# Define multi-word expressions
mwe_tokenizer = MWETokenizer([
    ('Natural', 'Language', 'Processing'),
    ('Machine', 'Learning'),
    ('sentiment', 'analysis')
], separator='_')

# Add MWEs and tokenize
mwe_tokens = mwe_tokenizer.tokenize(base_tokens)
print(f"Tokens: {mwe_tokens[:20]}...")
print(f"Total tokens: {len(mwe_tokens)}")
print(f"\nNotice: 'Natural Language Processing' became 'Natural_Language_Processing'")

# ============================================================================
# PART 2: STEMMING
# ============================================================================

print("\n" + "="*80)
print("PART 2: STEMMING TECHNIQUES")
print("="*80)

# Sample words for stemming
words_for_stemming = ['running', 'runs', 'ran', 'runner', 'runners', 
                      'easily', 'fairly', 'learning', 'learned', 'learns',
                      'connection', 'connections', 'connected', 'connecting']

# 1. Porter Stemmer
print("\n1. PORTER STEMMER")
print("-" * 60)
porter = PorterStemmer()
print(f"{'Word':<20} {'Stem':<20}")
print("-" * 40)
for word in words_for_stemming:
    stem = porter.stem(word)
    print(f"{word:<20} {stem:<20}")

# 2. Snowball Stemmer (English)
print("\n2. SNOWBALL STEMMER (English)")
print("-" * 60)
snowball = SnowballStemmer('english')
print(f"{'Word':<20} {'Stem':<20}")
print("-" * 40)
for word in words_for_stemming:
    stem = snowball.stem(word)
    print(f"{word:<20} {stem:<20}")

# Comparison between Porter and Snowball
print("\n3. COMPARISON: PORTER vs SNOWBALL")
print("-" * 60)
print(f"{'Word':<20} {'Porter':<20} {'Snowball':<20}")
print("-" * 60)
for word in words_for_stemming:
    porter_stem = porter.stem(word)
    snowball_stem = snowball.stem(word)
    print(f"{word:<20} {porter_stem:<20} {snowball_stem:<20}")

# ============================================================================
# PART 3: LEMMATIZATION
# ============================================================================

print("\n" + "="*80)
print("PART 3: LEMMATIZATION")
print("="*80)

# WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Words for lemmatization with their POS tags
words_with_pos = [
    ('running', 'v'),  # verb
    ('runs', 'v'),
    ('ran', 'v'),
    ('better', 'a'),   # adjective
    ('geese', 'n'),    # noun
    ('cacti', 'n'),
    ('was', 'v'),
    ('are', 'v'),
    ('fairly', 'r'),   # adverb
]

print("\nWORDNET LEMMATIZER")
print("-" * 60)
print(f"{'Word':<20} {'POS':<10} {'Lemma':<20}")
print("-" * 60)
for word, pos in words_with_pos:
    lemma = lemmatizer.lemmatize(word, pos=pos)
    pos_full = {'n': 'noun', 'v': 'verb', 'a': 'adjective', 'r': 'adverb'}[pos]
    print(f"{word:<20} {pos_full:<10} {lemma:<20}")

# ============================================================================
# PART 4: COMPARISON - STEMMING vs LEMMATIZATION
# ============================================================================

print("\n" + "="*80)
print("PART 4: STEMMING vs LEMMATIZATION COMPARISON")
print("="*80)

comparison_words = ['running', 'better', 'geese', 'connection', 'studying', 'was']

print(f"\n{'Word':<15} {'Porter Stem':<15} {'Snowball Stem':<15} {'Lemma (verb)':<15} {'Lemma (noun)':<15}")
print("-" * 90)
for word in comparison_words:
    porter_stem = porter.stem(word)
    snowball_stem = snowball.stem(word)
    lemma_v = lemmatizer.lemmatize(word, pos='v')
    lemma_n = lemmatizer.lemmatize(word, pos='n')
    print(f"{word:<15} {porter_stem:<15} {snowball_stem:<15} {lemma_v:<15} {lemma_n:<15}")

# ============================================================================
# PART 5: PRACTICAL EXAMPLE - COMPLETE PIPELINE
# ============================================================================

print("\n" + "="*80)
print("PART 5: COMPLETE TEXT PROCESSING PIPELINE")
print("="*80)

demo_sentence = "The runners were running faster than they ran yesterday in the competitions."
print(f"\nOriginal: {demo_sentence}")

# Step 1: Tokenize
tokens = treebank_tokenizer.tokenize(demo_sentence)
print(f"\nTokenized: {tokens}")

# Step 2: Stem with Porter
porter_stems = [porter.stem(token) for token in tokens]
print(f"\nPorter Stemmed: {porter_stems}")

# Step 3: Stem with Snowball
snowball_stems = [snowball.stem(token) for token in tokens]
print(f"\nSnowball Stemmed: {snowball_stems}")

# Step 4: Lemmatize (assuming all verbs for demonstration)
lemmas = [lemmatizer.lemmatize(token, pos='v') for token in tokens]
print(f"\nLemmatized (verb): {lemmas}")