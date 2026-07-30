"""rank_bm25 기반 BM25 검색기.

순수 파이썬 구현이라 수십만 문서까지가 실용 한계.
코퍼스가 커지면 bm25s 또는 tantivy로 교체할 것 (인터페이스는 동일 유지).
"""

from __future__ import annotations

import numpy as np

from retrieval_bench.retrievers.base import Retriever, top_k_indices
from retrieval_bench.tokenizers import Tokenizer


class BM25Retriever(Retriever):
    """주의: rank_bm25의 BM25Okapi는 음수 IDF를 epsilon*평균IDF로 치환한다.
    Lucene/Anserini BM25와 다른 동작이므로 외부 벤치마크 수치와 직접 비교 불가.
    형태소 토큰화는 조사·어미를 고빈도 토큰으로 만들어 음수 IDF 빈도가 높아
    epsilon이 실험 조건에 비대칭으로 작용할 수 있다 — config로 노출한다."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        k1: float = 0.9,
        b: float = 0.4,
        epsilon: float = 0.25,
    ):
        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.doc_ids: list[str] = []
        self._bm25 = None

    def index(self, corpus: dict[str, str]) -> None:
        from rank_bm25 import BM25Okapi

        self.doc_ids = list(corpus.keys())
        tokenized = [self.tokenizer(text) for text in corpus.values()]
        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b, epsilon=self.epsilon)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            raise RuntimeError("index()를 먼저 호출해야 합니다")
        scores = np.asarray(self._bm25.get_scores(self.tokenizer(query)))
        top = top_k_indices(scores, k)
        return [(self.doc_ids[i], float(scores[i])) for i in top]
