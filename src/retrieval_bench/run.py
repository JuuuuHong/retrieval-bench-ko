"""실험 러너.

사용법:
    python -m retrieval_bench.run configs/bm25_baseline.yaml

원칙:
- 결과 JSON에는 config 전문 + git SHA + 라이브러리 버전 + 데이터셋 무결성
  stats를 기록한다 — "재현 가능"은 기록으로 증명한다
- 결과는 최신본(results/{experiment}.json)과 타임스탬프 아카이브
  (results/history/)에 이중 저장 — 재실행이 이전 측정을 지우지 않는다
- print는 flush=True — 백그라운드 실행(stdout 리다이렉트) 시 블록 버퍼링으로
  진행 로그가 안 보이는 문제 방지
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from tqdm import tqdm

from retrieval_bench.data.miracl import load_miracl_ko
from retrieval_bench.eval.cost import mean_topk_tokens
from retrieval_bench.eval.metrics import aggregate
from retrieval_bench.report.markdown import metrics_table
from retrieval_bench.retrievers.bm25 import BM25Retriever
from retrieval_bench.tokenizers import get_tokenizer

_VERSIONED_DEPS = ("numpy", "rank-bm25", "kiwipiepy", "tiktoken", "datasets")
_WARMUP_SEARCHES = 3


def _build_one(r: dict):
    if r["type"] == "bm25":
        return BM25Retriever(
            tokenizer=get_tokenizer(r.get("tokenizer", "whitespace")),
            k1=r.get("k1", 0.9),
            b=r.get("b", 0.4),
            epsilon=r.get("epsilon", 0.25),
        )
    if r["type"] == "dense":
        from retrieval_bench.retrievers.dense import DenseRetriever

        return DenseRetriever(
            model_name=r["model"],
            query_prefix=r.get("query_prefix", ""),
            passage_prefix=r.get("passage_prefix", ""),
            batch_size=r.get("batch_size", 64),
        )
    if r["type"] == "rrf":
        from retrieval_bench.retrievers.rrf import RRFRetriever

        return RRFRetriever(
            components=[_build_one(c) for c in r["components"]],
            rrf_k=r.get("rrf_k", 60),
            candidate_k=r.get("candidate_k", 1000),
        )
    raise ValueError(f"지원하지 않는 retriever: {r['type']}")


def build_retriever(cfg: dict):
    return _build_one(cfg["retriever"])


def percentile(values: list[float], p: float) -> float:
    """nearest-rank 백분위 — 보간·외삽 없이 실측값만 반환한다."""
    if not values:
        raise ValueError("빈 리스트")
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def _git_info() -> dict:
    def run(*args):
        return subprocess.run(
            ["git", *args], capture_output=True, text=True
        ).stdout.strip()

    return {
        "git_sha": run("rev-parse", "HEAD"),
        "git_dirty": bool(run("status", "--porcelain")),
    }


def _versions() -> dict:
    out = {}
    for dep in _VERSIONED_DEPS:
        try:
            out[dep] = importlib.metadata.version(dep)
        except importlib.metadata.PackageNotFoundError:
            out[dep] = None
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    experiment = cfg["experiment"]
    k_values = cfg["eval"]["k"]
    cost_k = cfg["eval"].get("cost_k", 10)
    # 토큰 비용은 검색 결과에서 계산하므로 검색 깊이가 cost_k를 덮어야 한다
    max_k = max(max(k_values), cost_k)

    d = cfg["dataset"]
    if d["name"] != "miracl-ko":
        raise ValueError(
            f"지원하지 않는 데이터셋: {d['name']} (현재 miracl-ko만 지원) — "
            "config에 적힌 이름이 결과 JSON에 박제되므로 무음 대체는 하지 않는다"
        )
    print(f"[{experiment}] 데이터셋 로드 중: {d['name']} ({d['split']})", flush=True)
    dataset = load_miracl_ko(
        split=d["split"],
        corpus_sample=d.get("corpus_sample", 0),
        seed=d.get("seed", 42),
    )
    print(f"  쿼리 {len(dataset.queries)}건 / 코퍼스 {len(dataset.corpus)}건", flush=True)

    retriever = build_retriever(cfg)
    t0 = time.perf_counter()
    retriever.index(dataset.corpus)
    index_sec = time.perf_counter() - t0
    print(f"  인덱싱 완료: {index_sec:.1f}s", flush=True)

    # 워밍업 — 첫 쿼리의 콜드스타트 비용(JIT, 캐시, 지연 초기화)이
    # 레이턴시 분포 꼬리를 오염시키지 않도록 집계에서 제외
    warmup_query = next(iter(dataset.queries.values()))
    for _ in range(_WARMUP_SEARCHES):
        retriever.search(warmup_query, k=max_k)

    runs: dict[str, list[str]] = {}
    latencies: list[float] = []
    for qid, query in tqdm(dataset.queries.items(), desc="검색"):
        t0 = time.perf_counter()
        results = retriever.search(query, k=max_k)
        latencies.append((time.perf_counter() - t0) * 1000)
        runs[qid] = [doc_id for doc_id, _ in results]

    metrics = aggregate(runs, dataset.qrels, k_values)
    metrics["latency_p50_ms"] = percentile(latencies, 50)
    metrics["latency_p95_ms"] = percentile(latencies, 95)

    # 비용 분모를 품질 메트릭과 동일한 쿼리 집합(qrels 보유)으로 통일
    eval_runs = {q: r for q, r in runs.items() if dataset.qrels.get(q)}
    metrics[f"tokens@{cost_k}"] = mean_topk_tokens(eval_runs, dataset.corpus, cost_k)

    timestamp = datetime.now(timezone.utc)
    out = {
        "experiment": experiment,
        "timestamp": timestamp.isoformat(),
        "config": cfg,
        "corpus_size": len(dataset.corpus),
        "dataset_stats": dataset.stats,
        "index_sec": index_sec,
        "metrics": metrics,
        **_git_info(),
        "versions": _versions(),
    }
    payload = json.dumps(out, ensure_ascii=False, indent=2)
    latest = Path("results") / f"{experiment}.json"
    archive = (
        Path("results/history")
        / f"{experiment}_{timestamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(payload)
    archive.write_text(payload)

    print(f"\n결과 저장: {latest} (아카이브: {archive})\n", flush=True)
    print(metrics_table(experiment, metrics), flush=True)


if __name__ == "__main__":
    main()
