"""RRF 하이퍼파라미터(가중치·rrf_k)의 교차검증 튜닝.

왜 필요한가: 같은 평가 쿼리로 가중치를 고르고 같은 쿼리로 성능을 보고하면,
그 수치는 "평가셋에 맞춘" 낙관적 추정이다. 이 모듈은 쿼리를 K개 fold로 나눠
각 fold의 가중치를 **나머지 fold에서만** 고르고, 고른 가중치로 그 fold를
평가한다. 모든 쿼리가 정확히 한 번, 자기가 튜닝에 쓰이지 않은 설정으로
평가되므로(out-of-fold) 보고 수치에 튜닝 누수가 없다.

MIRACL train 쿼리로 튜닝하지 않는 이유: multilingual-e5는 MIRACL train으로
지도학습되어 train 쿼리에서 dense 성능이 부풀려진다 — 거기서 고른 가중치는
dense 쪽으로 치우친다. dev 내부 교차검증은 이 문제가 없다.

입력은 run.py의 `eval.save_runs`가 저장한 구성 검색기 순위(.cache/runs/*.json.gz).
검색을 다시 돌리지 않고 순위만 융합하므로 격자 전체 탐색이 수십 초에 끝난다.

사용법:
    python -m retrieval_bench.fusion_cv configs/cv/miracl_e5_kiwi.yaml
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import yaml

from retrieval_bench.eval.metrics import per_query_metrics
from retrieval_bench.retrievers.rrf import rrf_fuse

DEFAULT_RRF_K = [5, 10, 20, 40, 60, 100, 200]
DEFAULT_WEIGHTS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]


def load_runs(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def make_folds(qids: list[str], n_folds: int, seed: int) -> list[list[str]]:
    """시드 고정 셔플 후 라운드로빈 분할 — fold 크기 차이는 최대 1."""
    if n_folds < 2:
        raise ValueError("n_folds는 2 이상")
    if len(qids) < n_folds:
        raise ValueError(f"쿼리 {len(qids)}건 < fold {n_folds}개")
    order = sorted(qids)
    random.Random(seed).shuffle(order)
    return [order[i::n_folds] for i in range(n_folds)]


def _mean(per_query: dict[str, dict[str, float]], qids: list[str], metric: str) -> float:
    return sum(per_query[qid][metric] for qid in qids) / len(qids)


def select_params(
    grid_scores: dict[tuple[float, int], dict[str, dict[str, float]]],
    qids: list[str],
    objective: str,
    tiebreak: str,
) -> tuple[float, int]:
    """qids 위 평균 objective가 최대인 (가중치, rrf_k).

    동점이면 tiebreak 메트릭, 그래도 동점이면 무가중(w=1)에 가깝고 rrf_k=60에
    가까운 쪽 — 근거 없는 극단값 선택을 피하는 결정론적 규칙.
    """

    def key(params):
        scores = grid_scores[params]
        w, rk = params
        return (
            -_mean(scores, qids, objective),
            -_mean(scores, qids, tiebreak),
            abs(w - 1.0),
            abs(rk - 60),
        )

    return min(grid_scores, key=key)


def fuse_grid(
    rankings: list[dict[str, list[str]]],
    qrels: dict[str, set[str]],
    qids: list[str],
    k_values: list[int],
    weights_grid: list[float],
    rrf_k_grid: list[int],
) -> dict[tuple[float, int], dict[str, dict[str, float]]]:
    """격자의 모든 (가중치, rrf_k)에 대해 쿼리별 메트릭을 한 번씩 계산."""
    max_k = max(k_values)
    out = {}
    for w in weights_grid:
        for rk in rrf_k_grid:
            fused = {
                qid: [doc for doc, _ in rrf_fuse(
                    [rankings[0][qid], rankings[1][qid]], rrf_k=rk, weights=[1.0, w]
                )[:max_k]]
                for qid in qids
            }
            out[(w, rk)] = per_query_metrics(fused, qrels, k_values)
    return out


def cross_validate(
    rankings: list[dict[str, list[str]]],
    qrels: dict[str, set[str]],
    k_values: list[int],
    weights_grid: list[float],
    rrf_k_grid: list[int],
    n_folds: int,
    seed: int,
    objective: str,
    tiebreak: str,
) -> dict:
    """rankings[0]은 가중치 1 고정, rankings[1]에 weights_grid의 가중치를 준다.

    반환: out-of-fold 쿼리별 메트릭, fold별 선택 파라미터, 격자 전체의
    in-sample 점수(참고용 — 보고 수치로 쓰지 않는다).
    """
    if len(rankings) != 2:
        raise ValueError("현재는 검색기 2개 융합만 지원")
    qids = sorted(qid for qid in qrels if qid in rankings[0] and qid in rankings[1])
    if not qids:
        raise ValueError("공통 평가 쿼리가 없습니다")

    grid_scores = fuse_grid(rankings, qrels, qids, k_values, weights_grid, rrf_k_grid)
    folds = make_folds(qids, n_folds, seed)
    oof: dict[str, dict[str, float]] = {}
    oof_runs: dict[str, list[str]] = {}
    fold_info = []
    for i, test in enumerate(folds):
        train = [qid for j, f in enumerate(folds) if j != i for qid in f]
        w, rk = select_params(grid_scores, train, objective, tiebreak)
        for qid in test:
            oof[qid] = grid_scores[(w, rk)][qid]
            oof_runs[qid] = [doc for doc, _ in rrf_fuse(
                [rankings[0][qid], rankings[1][qid]], rrf_k=rk, weights=[1.0, w]
            )[:max(k_values)]]
        fold_info.append({
            "fold": i,
            "n_test": len(test),
            "weight": w,
            "rrf_k": rk,
            f"train_{objective}": _mean(grid_scores[(w, rk)], train, objective),
        })

    in_sample = {
        f"w={w},k={rk}": _mean(scores, qids, objective)
        for (w, rk), scores in grid_scores.items()
    }
    return {"per_query": oof, "runs": oof_runs, "folds": fold_info, "in_sample_grid": in_sample}


def main() -> None:
    # run.py는 tqdm·datasets 등 무거운 의존성을 끌고 온다 — 유닛테스트(CI)가
    # 이 모듈을 임포트할 때 필요 없도록 실행 시점에만 가져온다
    from retrieval_bench.run import _git_info, _versions

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    loaded = [load_runs(Path(p)) for p in cfg["components"]]
    metas = [x["meta"] for x in loaded]
    if len({json.dumps(m["dataset"], sort_keys=True) for m in metas}) != 1:
        raise ValueError("구성 검색기들의 데이터셋 설정이 다르다 — 같은 코퍼스끼리만 융합")
    qrels = {qid: set(docs) for qid, docs in metas[0]["qrels"].items()}
    k_values = cfg.get("k", [1, 5, 10, 100])
    if max(k_values) > min(m["depth"] for m in metas):
        raise ValueError("평가 k가 저장된 순위 깊이보다 크다")

    res = cross_validate(
        [x["runs"] for x in loaded], qrels, k_values,
        cfg.get("weights", DEFAULT_WEIGHTS), cfg.get("rrf_k", DEFAULT_RRF_K),
        cfg.get("folds", 5), cfg.get("seed", 0),
        cfg.get("objective", "recall@10"), cfg.get("tiebreak", "ndcg@10"),
    )
    oof = res["per_query"]
    names = list(next(iter(oof.values())))
    metrics = {n: sum(v[n] for v in oof.values()) / len(oof) for n in names}
    metrics["num_queries"] = len(oof)

    ts = datetime.now(timezone.utc)
    out = {
        "experiment": cfg["experiment"],
        "timestamp": ts.isoformat(),
        "config": cfg,
        "corpus_size": metas[0]["corpus_size"],
        "dataset": metas[0]["dataset"],
        "components": [m["experiment"] for m in metas],
        "metrics": metrics,
        "per_query": oof,
        "folds": res["folds"],
        "in_sample_grid": res["in_sample_grid"],
        **_git_info(),
        "versions": _versions(),
    }
    # out-of-fold 융합 순위도 저장 — 생성(RAG) 단계가 이 순위를 그대로 쓴다
    from retrieval_bench.run import save_runs

    save_runs(
        Path(".cache/runs") / f"{cfg['experiment']}.json.gz",
        res["runs"],
        {**metas[0], "experiment": cfg["experiment"], "depth": max(k_values)},
    )
    payload = json.dumps(out, ensure_ascii=False, indent=2)
    latest = Path("results") / f"{cfg['experiment']}.json"
    archive = Path("results/history") / f"{cfg['experiment']}_{ts.strftime('%Y%m%dT%H%M%SZ')}.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(payload)
    archive.write_text(payload)
    print(f"결과 저장: {latest}")
    for f in res["folds"]:
        print(f"  fold {f['fold']}: w_dense={f['weight']} rrf_k={f['rrf_k']}")
    print("  " + " · ".join(f"{k}={metrics[k]:.4f}" for k in ("recall@10", "recall@100", "ndcg@10")))


if __name__ == "__main__":
    main()
