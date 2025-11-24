# St. Carlo Acutis, pray for us!
from curses import beep
from typing import Any
import re

with open ("text.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Split text by GPT-4 style pattern (by words and punctuation)
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
preprocessed = [item for item in preprocessed if item.strip()]

# The vocabulary is made by splitting the text into tokens and giving each token a unique ID
vocab = {token:integer for integer, token in enumerate[str | Any](preprocessed)}

# Implementing the simple tokenizer
class SimpleTokenizerV1:
    def __init__(self, vocab: dict[str | Any, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item for item in preprocessed if item.strip()]
        ids = [self.str_to_int[item] for item in preprocessed]
        return ids

    def decode(self, ids: list[int]) -> str:
        text = " ".join([self.int_to_str[id] for id in ids])
        text = re.sub(r'\s+([,.?!"()\\\'-])', r'\1', text)
        return text

# tokenizer = SimpleTokenizerV1(vocab)

# ids = tokenizer.encode(text)
# print(ids[:100])

# Currently it will break if it encounters a token that is not in the vocabulary
# Solution: add token for unknown and end of text.

all_tokens = sorted(list[str | Any](set[str | Any](preprocessed)))
all_tokens.extend(["<|unk|>", "<|eot|>"])
vocab = {token:integer for integer, token in enumerate[str | Any](all_tokens)}

class SimpleTokenizerV2:
    def __init__(self, vocab: dict[str | Any, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item for item in preprocessed if item.strip()]
        ids = [self.str_to_int.get(item, self.str_to_int["<|unk|>"]) for item in preprocessed]
        return ids

    def decode(self, ids: list[int]) -> str:
        text = " ".join([self.int_to_str.get(id, "<|unk|>") for id in ids])
        text = re.sub(r'\s+([,.?!"()\\\'-])', r'\1', text)
        return text

text1 = "Hello, do you Nice to Nick Land meet you tea?"
tokenizer = SimpleTokenizerV2(vocab)
encoded = tokenizer.encode(text1)
print(encoded)
print(tokenizer.decode(encoded))

"""
LLM from scratch book just installs tiktoken at this point, which is not as fun as building it from scratch

Transformer architecture requires modification to be able to handle raw bytes, making attention extremely expensive. 
Solution: use a byte pair encoder (BPE).

https://www.fast.ai/posts/2025-10-16-karpathy-tokenizers:
"
The algorithm operates on an input sequence—for example, a sequence containing only four vocabulary elements: 
a, b, c, and d. Rather than working with bytes directly, consider this simplified case with a vocabulary size of four.

When a sequence becomes too long and requires compression, the algorithm iteratively identifies the most 
frequently occurring pair of tokens. Once identified, that pair is replaced with a single new token appended to the vocabulary.
...
The algorithm iteratively compresses the sequence while minting new tokens. 
The same approach applies to byte sequences: starting with 256 vocabulary size, we 
identify the most common byte pairs and iteratively mint new tokens, appending them to the 
vocabulary and performing replacements. This produces a compressed training dataset along with an algorithm for 
encoding arbitrary sequences using this vocabulary and decoding them back to strings.
"

"""

# Step 1: get the text
# print("Length of text:", len(text))

# Step 2: Encode the text to UTF-8 bytes and convert to list of integers
tokens = list[int](text.encode("utf-8"))
# print(f"UTF-8 encoded bytes: {tokens[:50]}...")  # Show first 50 bytes
# print(f"Length in bytes: {len(tokens)}")

# Get the most frequently occuring pairs
def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pairs in zip(ids, ids[1:]):
        counts[pairs] = counts.get(pairs, 0) + 1
    return counts

# Step 3: Get the most frequently occuring pairs
stats = get_stats(tokens)
print("Total number of unique pairs:", len(stats))
top_pairs = sorted(stats.items(), key=lambda x: x[1], reverse=True)[:10]
for count, pair in top_pairs:
    print(f"{pair}: {count}")

# Step 4: Get the most frequent pair using max() function
most_frequent_pair = max(stats, key=stats.get)
print(f"Most frequent pair: {most_frequent_pair}")
print(f"Occurs {stats[most_frequent_pair]} times")

# Conver the bytes to characters to see what they are
char1, char2 = most_frequent_pair
print(f"Bytes {char1} and {char2} correspond to characters {chr(char1)} and {chr(char2)}")

# Verify by finding all positions where this pair occurs
occurences = []
for i in range(len(tokens) - 1):
    if tokens[i] == most_frequent_pair[0] and tokens[i+1] == most_frequent_pair[1]:
        occurences.append(i)

print(f"Found {len(occurences)} occurrences of {most_frequent_pair}")

# Step 5: merge the most frequent pair
# BPE starts with 256 bytes (0-255), so new tokens start at 256
new_token_id = 256  # Changed from len(vocab)

# Step 6: iteratively merge most frequent pairs
def merge_pairs(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    """
    In the list of integers (ids), replace all consecutive occurrences 
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # If not at the very last position and the current pair matches the pair to merge
        if ids[i] == pair[0] and i < len(ids) and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2 # skip over the pair
        else:
            newids.append(ids[i])
            i += 1

    return newids

# Test with simple example
test_ids = [5, 6, 6, 7, 9, 1]
result = merge_pairs(test_ids, (6, 7), 99)
# print(f"Original: {test_ids}")
# print(f"After merging (6, 7) -> 99: {result}")

# Step 7: apply the merge to actual tokens
tokens2 = merge_pairs(tokens, most_frequent_pair, new_token_id)

print(f"Original length: {len(tokens)}")
print(f"After merge length: {len(tokens2)}")
print(f"Reduction: {len(tokens) - len(tokens2)} tokens")

# Verify the merge worked
print(f"\nOccurrences of new token {new_token_id}: {tokens2.count(new_token_id)}")
print(f"Occurrences of old pair in original: {sum(1 for i in range(len(tokens)-1) if (tokens[i], tokens[i+1]) == most_frequent_pair)}")


# Verify old pair is gone
old_pair_count = sum(1 for i in range(len(tokens2)-1) if (tokens2[i], tokens2[i+1]) == most_frequent_pair)
print(f"Occurrences of old pair in new tokens: {old_pair_count}")

# Step 8: BPE algorithm - iteratively merge most frequent pairs
current_tokens = tokens2
vocab_size = 256  # Changed from len(vocab) - BPE starts with 256 bytes
for step in range(2, 6):
    # Find most frequent pair
    stats = get_stats(current_tokens)
    if not stats:
        break

    most_frequent_pair = max(stats, key=stats.get)
    new_token_id = vocab_size
    # Merge the most frequent pair
    current_tokens = merge_pairs(current_tokens, most_frequent_pair, new_token_id)

    print(f"Step {step}: {len(current_tokens)} tokens, vocab size: {vocab_size}")
    print(f"  Merged pair: {most_frequent_pair} -> {vocab_size}")

    vocab_size += 1

print(f"\nFinal BPE vocabulary: {vocab_size} tokens")
