from typing import Any


import re

with open ("text.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Split text by GPT-4 style pattern (by words and punctuation)
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
preprocessed = [item for item in preprocessed if item.strip()]

# The vocabulary is made by splitting the text into tokens and giving each token a unique ID
vocab = {token:integer for integer, token in enumerate[str | Any](preprocessed)}

for i, item in enumerate[tuple[str | Any, int]](vocab.items()):
    print(item)
    if i >= 50:
        break

# Implementing the simple tokenizer
class SimpleTokenizerv1:
    def __init__(self, vocab: dict[str | Any, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]: