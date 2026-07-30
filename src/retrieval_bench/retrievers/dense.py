"""sentence-transformers 기반 벡터 검색기.

- 임베딩은 정규화 후 내적(=코사인 유사도)으로 검색
- 코퍼스 임베딩은 (모델명 + 문서 ID 집합) 해시 키로 디스크 캐시
  — 같은 코퍼스로 dense 단독/RRF 실험을 반복할 때 재인코딩을 피한다
- E5 계열은 query/passage 프리픽스가 필수라 config로 받는다
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from retrieval_bench.retrievers.base import Retriever, top_k_indices


class DenseRetriever(Retriever):
    def __init__(
        self,
        model_name: str,
        query_prefix: str = "",
        passage_prefix: str = "",
        batch_size: int = 64,
        cache_dir: str = ".cache",
    ):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self._model = None
        self.doc_ids: list[str] = []
        self._emb: np.ndarray | None = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _cache_key(self, corpus: dict[str, str]) -> Path:
        # 문서 ID뿐 아니라 텍스트도 해싱한다 — 텍스트 구성이 바뀌었는데
        # (제목 포함 여부, 청킹 등) 옛 임베딩이 조용히 재사용되는 사고 방지.
        # \x00/\x01 구분자는 ["ab","c"] vs ["a","bc"] 류의 키 충돌 방지.
        h = hashlib.sha1()
        h.update(self.model_name.encode())
        h.update(b"\x00")
        h.update(self.passage_prefix.encode())
        for doc_id in sorted(corpus):
            h.update(b"\x00")
            h.update(doc_id.encode())
            h.update(b"\x01")
            h.update(corpus[doc_id].encode())
        return self.cache_dir / f"emb_{h.hexdigest()[:16]}.npz"

    def index(self, corpus: dict[str, str]) -> None:
        # 모델 로드를 인덱싱 단계로 고정 — 캐시 히트 시 첫 search()에서
        # 모델 로드(수 초)가 일어나 쿼리 레이턴시를 오염시키는 문제 방지
        _ = self.model
        self.doc_ids = list(corpus.keys())
        cache = self._cache_key(corpus)
        if cache.exists():
            data = np.load(cache, allow_pickle=False)
            # 캐시의 문서 순서를 그대로 따른다
            self.doc_ids = list(data["doc_ids"])
            self._emb = data["emb"]
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        texts = [self.passage_prefix + corpus[d] for d in self.doc_ids]
        self._emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)
        np.savez(cache, doc_ids=np.array(self.doc_ids), emb=self._emb)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._emb is None:
            raise RuntimeError("index()를 먼저 호출해야 합니다")
        q = self.model.encode(
            [self.query_prefix + query], normalize_embeddings=True
        ).astype(np.float32)[0]
        scores = self._emb @ q
        # BM25와 동일한 top-k 선택 경로 — 레이턴시 비교에서 선택 알고리즘
        # 차이(argpartition vs 전체 정렬)가 교란변수가 되지 않도록 통일
        top = top_k_indices(scores, k)
        return [(self.doc_ids[i], float(scores[i])) for i in top]
