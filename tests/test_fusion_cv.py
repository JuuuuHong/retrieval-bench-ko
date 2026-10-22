"""교차검증 융합 튜닝 검증 — 튜닝 누수가 없다는 것을 테스트로 고정한다."""

import pytest

from retrieval_bench.fusion_cv import cross_validate, make_folds, select_params


class TestMakeFolds:
    def test_분할은_겹치지_않고_전부_덮는다(self):
        qids = [f"q{i}" for i in range(23)]
        folds = make_folds(qids, 5, seed=0)
        flat = [qid for f in folds for qid in f]
        assert sorted(flat) == sorted(qids)
        assert len(set(flat)) == len(flat)
        assert max(map(len, folds)) - min(map(len, folds)) <= 1

    def test_시드가_같으면_입력_순서와_무관하게_같다(self):
        qids = [f"q{i}" for i in range(10)]
        assert make_folds(qids, 3, 7) == make_folds(list(reversed(qids)), 3, 7)

    def test_쿼리가_fold보다_적으면_실패(self):
        with pytest.raises(ValueError):
            make_folds(["a", "b"], 3, 0)


class TestSelectParams:
    def test_목표_메트릭_최대를_고른다(self):
        grid = {
            (1.0, 60): {"a": {"r": 0.5, "n": 0.0}},
            (3.0, 60): {"a": {"r": 0.9, "n": 0.0}},
        }
        assert select_params(grid, ["a"], "r", "n") == (3.0, 60)

    def test_동점이면_보조_메트릭(self):
        grid = {
            (1.0, 60): {"a": {"r": 0.5, "n": 0.1}},
            (3.0, 60): {"a": {"r": 0.5, "n": 0.4}},
        }
        assert select_params(grid, ["a"], "r", "n") == (3.0, 60)

    def test_완전_동점이면_무가중_k60에_가까운_쪽(self):
        same = {"a": {"r": 0.5, "n": 0.1}}
        grid = {(4.0, 10): same, (1.0, 60): same, (1.5, 60): same}
        assert select_params(grid, ["a"], "r", "n") == (1.0, 60)


class TestCrossValidate:
    def test_평가_fold는_자기_자신으로_튜닝되지_않는다(self):
        # 쿼리 4개, 2-fold. 각 쿼리는 dense 가중치가 높을 때만 정답(d)을 1위로
        # 올리거나(dense형), 낮을 때만 올린다(bm25형).
        # bm25형 쿼리만 모인 fold가 있어도, 그 fold의 설정은 다른 fold에서 정해진다.
        bm25 = {}
        dense = {}
        qrels = {}
        for i in range(4):
            qid = f"q{i}"
            qrels[qid] = {"d"}
            if i < 2:   # bm25형: bm25가 정답을 1위, dense는 꼴찌
                bm25[qid] = ["d", "x", "y"]
                dense[qid] = ["x", "y", "d"]
            else:       # dense형
                bm25[qid] = ["x", "y", "d"]
                dense[qid] = ["d", "x", "y"]
        res = cross_validate(
            [bm25, dense], qrels, [1], weights_grid=[0.25, 4.0], rrf_k_grid=[1],
            n_folds=2, seed=0, objective="recall@1", tiebreak="ndcg@1",
        )
        folds = make_folds(sorted(qrels), 2, 0)
        for info, test in zip(res["folds"], folds):
            train = [qid for f in folds if f is not test for qid in f]
            train_bm25 = sum(1 for qid in train if qid in ("q0", "q1"))
            train_dense = len(train) - train_bm25
            if train_bm25 > train_dense:
                assert info["weight"] == 0.25
            elif train_dense > train_bm25:
                assert info["weight"] == 4.0
        assert set(res["per_query"]) == set(qrels)

    def test_oof_결과는_고른_설정의_쿼리별_점수와_같다(self):
        runs_a = {"q0": ["d", "x"], "q1": ["d", "x"], "q2": ["x", "d"]}
        runs_b = {"q0": ["x", "d"], "q1": ["x", "d"], "q2": ["d", "x"]}
        qrels = {qid: {"d"} for qid in runs_a}
        res = cross_validate(
            [runs_a, runs_b], qrels, [1], [1.0], [60], 3, 0, "recall@1", "ndcg@1",
        )
        # 격자가 1점이면 교차검증 결과 = 그 설정의 결과
        # w=1, k=60: 두 검색기에서 d와 x가 1·2위를 나눠 가져 동점 → 사전순 d 먼저
        assert all(res["per_query"][qid]["recall@1"] == 1.0 for qid in qrels)
