"""rank_bm25 기반 BM25 검색기.

순수 파이썬 구현이라 수십만 문서까지가 실용 한계.
코퍼스가 커지면 bm25s 또는 tantivy로 교체할 것 (인터페이스는 동일 유지).
"""

from __future__ import annotations

import numpy as np

from retrieval_bench.retrievers.base import Retriever
from retrieval_bench.tokenizers import Tokenizer


class BM25Retriever(Retriever):
    def __init__(self, tokenizer: Tokenizer, k1: float = 0.9, b: float = 0.4):
        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self._bm25 = None

    def index(self, corpus: dict[str, str]) -> None:
        from rank_bm25 import BM25Okapi

        self.doc_ids = list(corpus.keys())
        tokenized = [self.tokenizer(text) for text in corpus.values()]
        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            raise RuntimeError("index()를 먼저 호출해야 합니다")
        scores = self._bm25.get_scores(self.tokenizer(query))
        top = np.argsort(scores)[::-1][:k]
        return [(self.doc_ids[i], float(scores[i])) for i in top]
