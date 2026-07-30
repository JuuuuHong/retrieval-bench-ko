"""RRF(Reciprocal Rank Fusion) 융합 검색기.

점수 스케일이 다른 검색기들(BM25 점수 vs 코사인 유사도)을 점수 정규화 없이
순위만으로 융합한다: score(d) = Σ_r 1 / (rrf_k + rank_r(d))

rrf_fuse는 순수 함수로 분리해 수계산 유닛테스트 대상으로 둔다.
"""

from __future__ import annotations

from retrieval_bench.retrievers.base import Retriever


def rrf_fuse(
    rankings: list[list[str]],
    rrf_k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """각 검색기의 순위 리스트(문서 ID, 점수 내림차순)를 RRF로 융합.

    rank는 1부터 시작. 반환은 융합 점수 내림차순.
    동점은 문서 ID 사전순으로 안정 정렬해 재현성을 보장한다.
    weights: 검색기별 기여 가중치 — 약한 검색기가 강한 검색기의 상위권을
    끌어내리는 문제(무가중 RRF의 알려진 약점)를 실험하기 위한 축.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(f"weights 수 불일치: {len(weights)} vs {len(rankings)}")
    scores: dict[str, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (rrf_k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


class RRFRetriever(Retriever):
    def __init__(
        self,
        components: list[Retriever],
        rrf_k: int = 60,
        candidate_k: int = 1000,
        weights: list[float] | None = None,
    ):
        if len(components) < 2:
            raise ValueError("RRF는 검색기 2개 이상이 필요합니다")
        self.components = components
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.weights = weights

    def index(self, corpus: dict[str, str]) -> None:
        for c in self.components:
            c.index(corpus)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        rankings = [
            [doc_id for doc_id, _ in c.search(query, self.candidate_k)]
            for c in self.components
        ]
        return rrf_fuse(rankings, self.rrf_k, self.weights)[:k]
