"""Dependency-free BM25 with document-frequency IDF and Unicode tokens."""
from collections import Counter
import math
import re

import numpy as np


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())
    for phrase in re.findall(r"[\u3400-\u9fff]+", text):
        tokens.extend(phrase)
        tokens.extend(phrase[i:i + 2] for i in range(len(phrase) - 1))
    return tokens


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = .75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(document)) for document in documents]
        self.lengths = np.array([sum(doc.values()) for doc in self.docs], dtype=np.float32)
        self.avgdl = float(self.lengths.mean()) if len(self.lengths) else 1.0
        df = Counter(token for doc in self.docs for token in doc)
        n = len(self.docs)
        self.idf = {token: math.log(1 + (n - freq + .5) / (freq + .5)) for token, freq in df.items()}

    def score(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.docs), dtype=np.float32)
        for token in set(tokenize(query)):
            idf = self.idf.get(token, 0)
            if not idf:
                continue
            tf = np.array([doc.get(token, 0) for doc in self.docs], dtype=np.float32)
            norm = self.k1 * (1 - self.b + self.b * self.lengths / max(self.avgdl, 1e-9))
            scores += idf * tf * (self.k1 + 1) / (tf + norm)
        return scores
