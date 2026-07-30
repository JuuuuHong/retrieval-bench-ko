"""코퍼스 서브셋 선택·qrels 정합성 로직 검증 (datasets 다운로드 불필요).

리뷰 BLOCKER의 회귀 테스트 포함: 코퍼스에 없는 정답 docid가 recall 분모에
남으면 모든 recall 수치가 무음으로 왜곡된다.
"""

import pytest

from retrieval_bench.data.miracl import reconcile_qrels, select_corpus_docids


class TestSelectCorpusDocids:
    def test_판정_문서가_정렬순으로_먼저(self):
        result = select_corpus_docids({"b", "a"}, ["a", "b", "c", "d"], 3, seed=1)
        assert result[:2] == ["a", "b"]
        assert len(result) == 3

    def test_같은_시드는_같은_결과(self):
        args = ({"a"}, [f"d{i}" for i in range(100)] + ["a"], 50, 7)
        assert select_corpus_docids(*args) == select_corpus_docids(*args)

    def test_다른_시드는_다른_distractor(self):
        ids = [f"d{i}" for i in range(1000)]
        r1 = select_corpus_docids(set(), ids, 10, seed=1)
        r2 = select_corpus_docids(set(), ids, 10, seed=2)
        assert r1 != r2

    def test_sample_0이면_판정_문서만(self):
        assert select_corpus_docids({"b", "a"}, ["a", "b", "c"], 0, seed=1) == ["a", "b"]

    def test_판정_문서가_목표보다_많으면_에러(self):
        with pytest.raises(ValueError, match="corpus_sample을 키우세요"):
            select_corpus_docids({"a", "b", "c"}, ["a", "b", "c"], 2, seed=1)

    def test_전체_코퍼스보다_큰_목표는_에러(self):
        with pytest.raises(ValueError, match="전체 코퍼스"):
            select_corpus_docids(set(), ["a", "b"], 5, seed=1)

    def test_코퍼스에_없는_판정_문서는_제외(self):
        result = select_corpus_docids({"a", "ghost"}, ["a", "b"], 2, seed=1)
        assert "ghost" not in result
        assert len(result) == 2

    def test_중복_없음(self):
        result = select_corpus_docids({"a"}, ["a"] + [f"d{i}" for i in range(50)], 20, 3)
        assert len(result) == len(set(result)) == 20


class TestReconcileQrels:
    def test_코퍼스에_없는_정답은_분모에서_제외(self):
        # BLOCKER 회귀: 'missing'이 남으면 recall 상한이 조용히 낮아진다
        qrels, queries, stats = reconcile_qrels(
            {"q1": {"a", "missing"}},
            {"q1": "쿼리"},
            corpus_docids={"a"},
        )
        assert qrels == {"q1": {"a"}}
        assert stats["qrels_docs_missing_from_corpus"] == 1

    def test_정답_전멸_쿼리는_드롭(self):
        qrels, queries, stats = reconcile_qrels(
            {"q1": {"missing"}, "q2": {"a"}},
            {"q1": "쿼리1", "q2": "쿼리2"},
            corpus_docids={"a"},
        )
        assert "q1" not in qrels
        assert "q1" not in queries
        assert stats["queries_dropped"] == 1
        assert stats["queries_evaluated"] == 1

    def test_쿼리_텍스트_없는_qrels는_드롭(self):
        qrels, queries, stats = reconcile_qrels(
            {"q1": {"a"}, "q_no_text": {"a"}},
            {"q1": "쿼리"},
            corpus_docids={"a"},
        )
        assert set(qrels) == {"q1"}
        assert stats["queries_dropped"] == 1

    def test_정상_케이스는_무손실(self):
        qrels, queries, stats = reconcile_qrels(
            {"q1": {"a"}}, {"q1": "쿼리"}, corpus_docids={"a", "b"}
        )
        assert qrels == {"q1": {"a"}}
        assert stats == {
            "qrels_docs_missing_from_corpus": 0,
            "queries_dropped": 0,
            "queries_evaluated": 1,
        }
