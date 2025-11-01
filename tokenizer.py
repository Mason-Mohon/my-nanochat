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
print(tokenizer.decode(tokenizer.encode(text1)))