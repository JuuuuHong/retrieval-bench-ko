"""BM25 검색기 end-to-end — 가짜 코퍼스로 정렬·동점 규약을 고정.

리뷰에서 나온 실제 버그의 회귀 테스트: argsort([::-1]) 방식은 동점 그룹 내
순서를 역전시켜, 코퍼스 뒤쪽 문서(distractor)가 동점인 앞쪽 문서(판정 문서)를
구조적으로 이기게 만들었다.
"""

import numpy as np

from retrieval_bench.retrievers.base import top_k_indices
from retrieval_bench.retrievers.bm25 import BM25Retriever
from retrieval_bench.tokenizers import whitespace_tokenizer


def _corpus():
    return {
        "d1": "서울 지하철 노선 안내",
        "d2": "파이썬 백엔드 개발",
        "d3": "김치찌개 만드는 법",
        "d4": "부산 버스 노선",
    }


class TestTopKIndices:
    def test_점수_내림차순(self):
        assert list(top_k_indices(np.array([1.0, 3.0, 2.0]), 3)) == [1, 2, 0]

    def test_동점은_낮은_인덱스_우선(self):
        # 리뷰 재현 케이스: [0,0,5,0,0]에서 옛 구현은 [2,4,3,...]을 반환했다
        assert list(top_k_indices(np.array([0.0, 0.0, 5.0, 0.0, 0.0]), 3)) == [2, 0, 1]

    def test_k가_길이보다_커도_동작(self):
        assert len(top_k_indices(np.array([1.0, 2.0]), 10)) == 2


class TestBM25Retriever:
    def test_관련_문서가_1위(self):
        r = BM25Retriever(tokenizer=whitespace_tokenizer)
        r.index(_corpus())
        top = r.search("지하철 노선", k=2)
        assert top[0][0] == "d1"

    def test_무관한_쿼리의_전원_동점은_삽입순(self):
        r = BM25Retriever(tokenizer=whitespace_tokenizer)
        r.index(_corpus())
        # 코퍼스에 없는 토큰만으로 검색 -> 전원 0점 동점 -> 삽입 순서 유지
        ranked = [doc_id for doc_id, _ in r.search("우주선", k=4)]
        assert ranked == ["d1", "d2", "d3", "d4"]

    def test_doc_ids와_점수_정렬_일치(self):
        r = BM25Retriever(tokenizer=whitespace_tokenizer)
        r.index(_corpus())
        results = r.search("파이썬 백엔드", k=4)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0][0] == "d2"

    def test_index_전_search는_에러(self):
        import pytest

        r = BM25Retriever(tokenizer=whitespace_tokenizer)
        with pytest.raises(RuntimeError):
            r.search("쿼리", k=1)
