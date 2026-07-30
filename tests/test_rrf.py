"""RRF 융합 순수 함수 수계산 검증."""

import pytest

from retrieval_bench.retrievers.rrf import rrf_fuse


class TestRrfFuse:
    def test_수계산_대조(self):
        # r1: a(1위), b(2위) / r2: b(1위), a(2위), rrf_k=60
        # a = 1/61 + 1/62, b = 1/62 + 1/61 -> 동점, 사전순 a 먼저
        result = rrf_fuse([["a", "b"], ["b", "a"]], rrf_k=60)
        assert result[0][0] == "a"
        assert result[0][1] == pytest.approx(1 / 61 + 1 / 62)
        assert result[1][1] == pytest.approx(result[0][1])

    def test_양쪽_상위가_한쪽_1위를_이긴다(self):
        # b는 양쪽 모두 2위: 2/(60+2), a와 c는 각각 한쪽 1위: 1/61
        result = rrf_fuse([["a", "b"], ["c", "b"]], rrf_k=60)
        assert result[0][0] == "b"
        assert result[0][1] == pytest.approx(2 / 62)

    def test_한쪽에만_있는_문서도_포함(self):
        result = rrf_fuse([["a"], ["b"]], rrf_k=60)
        assert {d for d, _ in result} == {"a", "b"}

    def test_rrf_k가_클수록_순위차_영향_감소(self):
        # rrf_k -> 무한대면 모든 기여가 균등해짐
        low = rrf_fuse([["a", "b", "c"]], rrf_k=1)
        high = rrf_fuse([["a", "b", "c"]], rrf_k=10_000)
        gap_low = low[0][1] - low[2][1]
        gap_high = high[0][1] - high[2][1]
        assert gap_low > gap_high

    def test_동점은_사전순_재현성(self):
        r1 = rrf_fuse([["x", "y"], ["y", "x"]])
        r2 = rrf_fuse([["y", "x"], ["x", "y"]])
        assert [d for d, _ in r1] == [d for d, _ in r2]
