"""검색기 공통 인터페이스.

2주차에 DenseRetriever, RRFRetriever가 같은 인터페이스로 추가된다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Retriever(ABC):
    @abstractmethod
    def index(self, corpus: dict[str, str]) -> None:
        """corpus: 문서 ID -> 문서 텍스트."""

    @abstractmethod
    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """(문서 ID, 점수) 리스트를 점수 내림차순으로 반환."""
