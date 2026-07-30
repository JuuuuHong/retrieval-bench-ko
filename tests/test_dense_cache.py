"""DenseRetriever 캐시 키 검증 (모델 로드 불필요 — 키 계산만).

리뷰 HIGH의 회귀 테스트: 캐시 키에 문서 텍스트가 빠지면 청킹/전처리를
바꿔도 옛 임베딩이 재사용되어 "차이 없음"이라는 틀린 결론이 나온다.
"""

from retrieval_bench.retrievers.dense import DenseRetriever


def _r(**kwargs):
    return DenseRetriever(model_name="dummy-model", **kwargs)


class TestCacheKey:
    def test_같은_코퍼스는_같은_키(self):
        corpus = {"a": "텍스트1", "b": "텍스트2"}
        assert _r()._cache_key(corpus) == _r()._cache_key(dict(corpus))

    def test_텍스트가_바뀌면_다른_키(self):
        k1 = _r()._cache_key({"a": "원본 텍스트"})
        k2 = _r()._cache_key({"a": "수정된 텍스트"})
        assert k1 != k2

    def test_문서_순서는_키에_무관(self):
        k1 = _r()._cache_key({"a": "1", "b": "2"})
        k2 = _r()._cache_key({"b": "2", "a": "1"})
        assert k1 == k2

    def test_구분자_모호성_없음(self):
        # id/text 경계가 구분자 없이 이어지면 충돌 가능한 케이스
        k1 = _r()._cache_key({"ab": "c"})
        k2 = _r()._cache_key({"a": "bc"})
        assert k1 != k2

    def test_모델명과_프리픽스가_키에_반영(self):
        corpus = {"a": "텍스트"}
        base = _r()._cache_key(corpus)
        assert DenseRetriever(model_name="other")._cache_key(corpus) != base
        assert _r(passage_prefix="passage: ")._cache_key(corpus) != base
