"""검색기 공통 인터페이스와 공용 top-k 선택.

top-k 선택을 공용 헬퍼로 통일하는 이유:
- 검색기마다 선택 알고리즘이 다르면(전체 정렬 vs argpartition) 레이턴시 비교에
  검색 방식 차이가 아닌 구현 차이가 섞인다
- np.argsort 기본(quicksort)은 불안정 정렬이라 동점 순서가 환경에 따라 바뀌고,
  내림차순 뒤집기([::-1])는 동점 그룹 내 순서를 역전시켜 코퍼스 뒤쪽 문서
  (distractor)가 동점인 정답보다 앞서는 체계적 편향을 만든다
→ lexsort로 "점수 내림차순, 동점은 삽입 순서(낮은 인덱스) 우선"을 고정한다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """점수 내림차순 상위 k개 인덱스. 동점은 낮은 인덱스 우선(결정론적)."""
    order = np.lexsort((np.arange(len(scores)), -scores))
    return order[:k]


class Retriever(ABC):
    @abstractmethod
    def index(self, corpus: dict[str, str]) -> None:
        """corpus: 문서 ID -> 문서 텍스트."""

    @abstractmethod
    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """(문서 ID, 점수) 리스트를 점수 내림차순으로 반환."""
