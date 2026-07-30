"""토큰 비용 메트릭 검증.

tiktoken의 절대 토큰 수는 버전에 따라 달라질 수 있으므로,
정확한 수치 대신 불변 성질(단조성·평균 계산)을 검증한다.
"""

import pytest

from retrieval_bench.eval.cost import count_tokens, mean_topk_tokens


class TestCountTokens:
    def test_빈_문자열은_0(self):
        assert count_tokens("") == 0

    def test_긴_텍스트가_더_많다(self):
        short = count_tokens("서울")
        long = count_tokens("서울 지하철 노선 안내 " * 20)
        assert 0 < short < long


class TestMeanTopkTokens:
    def test_평균_계산(self):
        corpus = {"a": "가", "b": "가", "c": "가"}
        one = count_tokens("가")
        # q1은 상위 2개(a,b), q2는 상위 1개(c)만 -> (2+1)/2 * one
        runs = {"q1": ["a", "b"], "q2": ["c"]}
        assert mean_topk_tokens(runs, corpus, k=2) == pytest.approx(1.5 * one)

    def test_k로_자른다(self):
        corpus = {"a": "가", "b": "가"}
        runs = {"q1": ["a", "b"]}
        assert mean_topk_tokens(runs, corpus, k=1) == pytest.approx(
            count_tokens("가")
        )

    def test_빈_runs면_에러(self):
        with pytest.raises(ValueError):
            mean_topk_tokens({}, {}, k=10)
