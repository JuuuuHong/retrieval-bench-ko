"""메트릭 유닛테스트 — 손으로 계산한 값과 대조.

측정 도구의 신뢰성이 이 프로젝트의 존재 이유이므로,
메트릭 구현은 반드시 수계산 검증을 거친다.
"""

import math

import pytest

from retrieval_bench.eval.metrics import (
    aggregate,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


class TestRecallAtK:
    def test_전부_잡으면_1(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0

    def test_절반만_잡으면_0_5(self):
        assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5

    def test_k_밖의_정답은_무시(self):
        # 정답 b가 3위지만 k=2라서 미포함
        assert recall_at_k(["a", "x", "b"], {"a", "b"}, k=2) == 0.5

    def test_정답이_비면_에러(self):
        with pytest.raises(ValueError):
            recall_at_k(["a"], set(), k=1)


class TestMrrAtK:
    def test_1위_적중이면_1(self):
        assert mrr_at_k(["a", "x"], {"a"}, k=10) == 1.0

    def test_3위_적중이면_1_3(self):
        assert mrr_at_k(["x", "y", "a"], {"a"}, k=10) == pytest.approx(1 / 3)

    def test_k_안에_정답_없으면_0(self):
        assert mrr_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0

    def test_첫_정답_기준(self):
        # 정답이 2위와 4위 -> 첫 정답인 2위 기준 1/2
        assert mrr_at_k(["x", "a", "y", "b"], {"a", "b"}, k=10) == 0.5


class TestNdcgAtK:
    def test_완벽한_순위면_1(self):
        assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=3) == pytest.approx(1.0)

    def test_정답_없으면_0(self):
        assert ndcg_at_k(["x", "y"], {"a"}, k=2) == 0.0

    def test_수계산_대조(self):
        # 정답 a가 2위: DCG = 1/log2(3), IDCG = 1/log2(2) = 1
        expected = (1 / math.log2(3)) / 1.0
        assert ndcg_at_k(["x", "a"], {"a"}, k=2) == pytest.approx(expected)

    def test_idcg는_정답수와_k중_작은값_기준(self):
        # 정답 3개지만 k=2면 IDCG는 상위 2개 기준
        ideal_2 = 1 / math.log2(2) + 1 / math.log2(3)
        assert ndcg_at_k(["a", "b"], {"a", "b", "c"}, k=2) == pytest.approx(
            (1 / math.log2(2) + 1 / math.log2(3)) / ideal_2
        )


class TestAggregate:
    def test_두_쿼리_평균(self):
        runs = {"q1": ["a", "x"], "q2": ["y", "b"]}
        qrels = {"q1": {"a"}, "q2": {"b"}}
        m = aggregate(runs, qrels, k_values=[1])
        # q1: recall@1=1, q2: recall@1=0 -> 평균 0.5
        assert m["recall@1"] == 0.5
        assert m["num_queries"] == 2

    def test_qrels_없는_쿼리는_제외(self):
        runs = {"q1": ["a"], "q2": ["b"]}
        qrels = {"q1": {"a"}}
        m = aggregate(runs, qrels, k_values=[1])
        assert m["num_queries"] == 1
        assert m["recall@1"] == 1.0

    def test_매칭_0건이면_에러(self):
        with pytest.raises(ValueError):
            aggregate({"q1": ["a"]}, {}, k_values=[1])
